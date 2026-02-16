#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# Allow running the script directly from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.informer_gat import DualChannelInformerGAT


WEATHER_FEATURES = [
    "tmax_mean",
    "tmax_std",
    "tmin_mean",
    "tmin_std",
    "tp_mean",
    "tp_std",
    "ssrd_mean",
    "ssrd_std",
    "wind_speed_mean",
    "wind_speed_std",
]

SAT_FEATURES = ["B7", "B8", "B8A", "B12"]


DISTRICT_ALIASES = {
    # Haryana
    "gurgaon": "gurugram",
    "mewat": "nuh",
    "hisar": "hissar",
    "sonipat": "sonepat",
    "yamunanagar": "yamuna nagar",
    # Punjab
    "ferozepur": "firozpur",
    "sas nagar": "mohali",
    "s a s nagar": "mohali",
    "s a s nagar sahibzada ajit singh nagar": "mohali",
    "shaheed bhagat singh nagar": "nawan shehar",
    "shahid bhagat singh nagar": "nawan shehar",
    # Uttar Pradesh
    "budaun": "badaun",
    "kanpur nagar": "kanpur",
    "kheri": "lakhimpur kheri",
    "kushi nagar": "kushinagar",
    "mau": "maunathbhanjan",
    "siddharthnagar": "siddharth nagar",
    "bhadohi": "sant ravi das nagar",
    "sant ravidas nagar": "sant ravi das nagar",
    "sant kabeer nagar": "sant kabir nagar",
}

STATE_ALIASES = {
    "uttar pradesh": "Uttar Pradesh",
    "uttar_pradesh": "Uttar Pradesh",
    "haryana": "Haryana",
    "punjab": "Punjab",
}


@dataclass(frozen=True)
class SeasonSelection:
    season_year: int
    operational_label: str
    target_date: pd.Timestamp
    issue_date: pd.Timestamp
    harvest_date: pd.Timestamp
    horizon_days: int


def _log(msg: str) -> None:
    print(msg, flush=True)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _fmt_seconds(sec: float) -> str:
    sec_i = int(max(0, round(sec)))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class _EpochProgress:
    def __init__(self, total: int, prefix: str, width: int = 28) -> None:
        self.total = max(1, int(total))
        self.prefix = prefix
        self.width = max(10, int(width))
        self.t0 = time.perf_counter()
        self.last_line_len = 0

    def update(self, epoch: int, train_mse: float, val_mse: float) -> None:
        ep = max(0, min(int(epoch), self.total))
        pct = ep / self.total
        filled = int(round(self.width * pct))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.t0
        rate = ep / elapsed if elapsed > 0 and ep > 0 else 0.0
        remain = (self.total - ep) / rate if rate > 0 else 0.0
        line = (
            f"\r{self.prefix} [{bar}] {ep:03d}/{self.total:03d} "
            f"{pct * 100:5.1f}% elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(remain)} "
            f"train_mse={train_mse:.2f} val_mse={val_mse:.2f}"
        )
        pad = max(0, self.last_line_len - len(line))
        print(line + (" " * pad), end="", flush=True)
        self.last_line_len = len(line)

    def close(self) -> None:
        print("", flush=True)


def _norm(text: str) -> str:
    x = str(text).strip().lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _canon_state(text: str) -> str:
    n = _norm(text)
    return STATE_ALIASES.get(n, str(text).strip())


def _canon_district(text: str) -> str:
    n = _norm(text)
    return DISTRICT_ALIASES.get(n, n)


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _load_district_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["district_id", "state_name", "district_name", "district_index"]].copy()
    df = df.sort_values("district_index").reset_index(drop=True)
    df["state_norm"] = df["state_name"].map(_norm)
    df["district_norm"] = df["district_name"].map(_canon_district)
    return df


def _load_yield_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["district_id", "season_start_year", "yield_kg_per_ha", "area_ha"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Yield file missing required columns: {missing}")
    return df[needed].copy()


def _season_split(
    years: Sequence[int],
    mode: str,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    years = sorted(years)
    if mode == "fixed":
        train = [y for y in years if y <= 2020]
        val = [y for y in years if y == 2021]
        test = [y for y in years if y == 2022]
        if not train or not val or not test:
            raise RuntimeError("Fixed split requires seasons including 2017-2022.")
        return train, val, test

    if len(years) < 4:
        raise RuntimeError("Random split requires at least 4 labeled seasons.")
    rng = random.Random(seed)
    shuffled = years.copy()
    rng.shuffle(shuffled)
    val = [shuffled[0]]
    test = [shuffled[1]]
    train = sorted(shuffled[2:])
    return train, val, test


def _load_weather_year(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        path,
        columns=["district_id", "issue_date", "lead_day"] + WEATHER_FEATURES,
    )
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    return df


def _parse_operational_target_date(label: str, season_year: int) -> pd.Timestamp:
    m = re.match(r"^\s*(\d{2})[-/](\d{2})\s*$", str(label))
    if not m:
        raise ValueError(f"Operational date label must be MM-DD. Got: {label}")
    month = int(m.group(1))
    day = int(m.group(2))
    year = season_year if month >= 9 else season_year + 1
    return pd.Timestamp(dt.date(year, month, day))


def _pick_issue_date_for_operational_label(
    weather_year_df: pd.DataFrame,
    season_year: int,
    operational_label: str,
    horizon_days: int,
) -> SeasonSelection:
    issues = pd.to_datetime(sorted(weather_year_df["issue_date"].unique()))
    harvest_date = pd.Timestamp(issues[-1])
    target_date = _parse_operational_target_date(operational_label, season_year=season_year)
    valid = issues[issues <= target_date]
    if len(valid) == 0:
        issue_date = pd.Timestamp(issues[0])
    else:
        issue_date = pd.Timestamp(valid[-1])
    return SeasonSelection(
        season_year=season_year,
        operational_label=operational_label,
        target_date=target_date,
        issue_date=issue_date,
        harvest_date=harvest_date,
        horizon_days=horizon_days,
    )


def _weather_tensor_for_season(
    weather_df: pd.DataFrame,
    district_ids: List[str],
    issue_date: pd.Timestamp,
    horizon_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(district_ids)
    f = len(WEATHER_FEATURES)
    x = np.zeros((n, horizon_days, f), dtype=np.float32)
    m = np.zeros((n, horizon_days), dtype=np.float32)

    sub = weather_df[weather_df["issue_date"] == issue_date].copy()
    sub = sub[sub["lead_day"] <= horizon_days]
    for i, did in enumerate(district_ids):
        rows = sub[sub["district_id"] == did].sort_values("lead_day")
        if rows.empty:
            continue
        lead_map = {int(r.lead_day): r for r in rows.itertuples(index=False)}
        for lead in range(1, horizon_days + 1):
            row = lead_map.get(lead)
            if row is None:
                continue
            vals = [float(getattr(row, c)) for c in WEATHER_FEATURES]
            arr = np.array(vals, dtype=np.float32)
            if np.isnan(arr).any():
                continue
            x[i, lead - 1, :] = arr
            m[i, lead - 1] = 1.0
    return x, m


def _parse_sat_file_key(file_name: str) -> Tuple[str, str]:
    stem = file_name.replace("_remote_sensing_data.csv", "")
    if stem.startswith("Uttar_Pradesh_"):
        return "Uttar Pradesh", stem[len("Uttar_Pradesh_") :]
    if stem.startswith("Haryana_"):
        return "Haryana", stem[len("Haryana_") :]
    if stem.startswith("Punjab_"):
        return "Punjab", stem[len("Punjab_") :]
    raise ValueError(f"Unrecognized satellite merged filename: {file_name}")


def _load_satellite_merged(merged_dir: Path) -> Dict[Tuple[str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for p in sorted(merged_dir.glob("*_remote_sensing_data.csv")):
        state_name, district_name = _parse_sat_file_key(p.name)
        key = (_norm(state_name), _canon_district(district_name))
        df = pd.read_csv(p)
        keep_cols = ["year", "time_step", "end_date"] + SAT_FEATURES
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Satellite file {p} missing columns: {missing}")
        d = df[keep_cols].copy()
        d["end_date"] = pd.to_datetime(d["end_date"])
        for c in SAT_FEATURES:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        out[key] = d
    return out


def _load_sangrur_malerkotla_weights(path: Path) -> Dict[int, Tuple[float, float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[int, Tuple[float, float]] = {}
    if "season_label" not in df.columns:
        return out
    for row in df.itertuples(index=False):
        m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", str(row.season_label))
        if not m:
            continue
        year = int(m.group(1))
        ws = float(getattr(row, "sangrur_area_before", 0.0) or 0.0)
        wm = float(getattr(row, "malerkotla_area_added", 0.0) or 0.0)
        out[year] = (ws, wm)
    return out


def _sat_step_active_mask(season_year: int, seq_len: int, op_date: pd.Timestamp) -> np.ndarray:
    season_start = pd.Timestamp(dt.date(season_year, 10, 1))
    active = np.zeros(seq_len, dtype=np.float32)
    for t in range(seq_len):
        # step represents [start, end) of a 5-day composite.
        step_end = season_start + pd.Timedelta(days=(t + 1) * 5)
        if step_end <= op_date:
            active[t] = 1.0
    return active


def _satellite_tensor_for_season(
    sat_map: Dict[Tuple[str, str], pd.DataFrame],
    district_df: pd.DataFrame,
    season_year: int,
    op_date: pd.Timestamp,
    seq_len: int,
    sangrur_weights: Dict[int, Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(district_df)
    f = len(SAT_FEATURES)
    x = np.zeros((n, seq_len, f), dtype=np.float32)
    m = np.zeros((n, seq_len), dtype=np.float32)

    punjab_key = _norm("Punjab")
    sangrur_key = _canon_district("Sangrur")
    malerkotla_key = _canon_district("Malerkotla")

    for i, row in enumerate(district_df.itertuples(index=False)):
        key = (row.state_norm, row.district_norm)
        sat_df = sat_map.get(key)

        if row.state_norm == punjab_key and row.district_norm == sangrur_key:
            # Merge Malerkotla into Sangrur for feature consistency with yield labels.
            sang = sat_map.get((punjab_key, sangrur_key))
            mal = sat_map.get((punjab_key, malerkotla_key))
            if sang is not None:
                sang = sang[sang["year"] == season_year].copy()
            if mal is not None:
                mal = mal[mal["year"] == season_year].copy()
            if sang is not None and not sang.empty:
                if mal is not None and not mal.empty and season_year in sangrur_weights:
                    ws, wm = sangrur_weights.get(season_year, (1.0, 0.0))
                    ws = float(ws)
                    wm = float(wm)
                    if wm > 0 and ws + wm > 0:
                        merged = sang.merge(
                            mal[["time_step"] + SAT_FEATURES],
                            on="time_step",
                            how="left",
                            suffixes=("_s", "_m"),
                        )
                        for c in SAT_FEATURES:
                            s_val = merged[f"{c}_s"].to_numpy(dtype=np.float32)
                            m_val = merged[f"{c}_m"].to_numpy(dtype=np.float32)
                            m_val = np.where(np.isnan(m_val), s_val, m_val)
                            merged[c] = (s_val * ws + m_val * wm) / (ws + wm)
                        sat_df = merged[["year", "time_step", "end_date"] + SAT_FEATURES].copy()
                    else:
                        sat_df = sang
                else:
                    sat_df = sang
            else:
                sat_df = None

        if sat_df is None:
            continue

        d = sat_df[sat_df["year"] == season_year].copy()
        if d.empty:
            continue
        d = d[d["end_date"] <= op_date].copy()
        if d.empty:
            continue
        d = d.sort_values("time_step")
        for rec in d.itertuples(index=False):
            t = int(rec.time_step)
            if t < 0 or t >= seq_len:
                continue
            vals = [float(getattr(rec, c)) if pd.notna(getattr(rec, c)) else np.nan for c in SAT_FEATURES]
            arr = np.array(vals, dtype=np.float32)
            if np.isnan(arr).any():
                continue
            x[i, t, :] = arr
            m[i, t] = 1.0
    return x, m


def _impute_satellite_by_state_mean(
    sat_x: np.ndarray,
    sat_mask: np.ndarray,
    district_df: pd.DataFrame,
    active_steps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    out = sat_x.copy()
    out_mask = sat_mask.copy()
    n, t, f = out.shape
    assert len(active_steps) == t

    for state in sorted(district_df["state_name"].unique()):
        idx = district_df.index[district_df["state_name"] == state].tolist()
        if not idx:
            continue
        state_x = out[idx, :, :]
        state_m = out_mask[idx, :]
        for ti in range(t):
            if active_steps[ti] == 0:
                continue
            valid_nodes = state_m[:, ti] > 0
            if not np.any(valid_nodes):
                continue
            mean_val = state_x[valid_nodes, ti, :].mean(axis=0)
            miss_nodes = ~valid_nodes
            if np.any(miss_nodes):
                state_x[miss_nodes, ti, :] = mean_val
        out[idx, :, :] = state_x
    return out, out_mask


def _build_adjacency_from_boundaries(
    district_df: pd.DataFrame,
    config_path: Path,
) -> torch.Tensor:
    cfg = _load_yaml(config_path)
    bcfg = cfg["boundaries"]
    path = Path(bcfg["admin2_path"])
    layer = bcfg["admin2_layer"]
    state_field = bcfg["state_field"]
    district_field = bcfg["district_field"]
    country_field = bcfg["country_field"]
    country_iso3 = bcfg["country_iso3"]

    gdf = gpd.read_file(path, layer=layer)
    gdf = gdf[gdf[country_field] == country_iso3].copy()
    gdf["state_name"] = gdf[state_field].astype(str)
    gdf["district_name"] = gdf[district_field].astype(str)
    gdf = gdf[gdf["state_name"].isin(["Punjab", "Haryana", "Uttar Pradesh"])].copy()
    gdf["state_norm"] = gdf["state_name"].map(_norm)
    gdf["district_norm"] = gdf["district_name"].map(_canon_district)

    lookup = {
        (r.state_norm, r.district_norm): r.district_id
        for r in district_df.itertuples(index=False)
    }
    gdf["district_id"] = [
        lookup.get((s, d), None) for s, d in zip(gdf["state_norm"], gdf["district_norm"])
    ]
    gdf = gdf[gdf["district_id"].notna()].copy()
    gdf = gdf.drop_duplicates(subset=["district_id"]).copy()
    gdf = gdf.set_index("district_id")

    ids = district_df["district_id"].tolist()
    n = len(ids)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        adj[i, i] = 1.0
    for i in range(n):
        id_i = ids[i]
        if id_i not in gdf.index:
            continue
        gi = gdf.loc[id_i].geometry
        for j in range(i + 1, n):
            id_j = ids[j]
            if id_j not in gdf.index:
                continue
            gj = gdf.loc[id_j].geometry
            if gi.touches(gj) or gi.intersects(gj):
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Fallback: if a node has only self-loop, connect to nearest nodes in same state by centroid.
    deg = adj.sum(axis=1)
    isolated = np.where(deg <= 1.0)[0]
    if len(isolated) > 0:
        centroids = {}
        for did in ids:
            if did in gdf.index:
                centroids[did] = gdf.loc[did].geometry.centroid
        for idx in isolated:
            did = ids[idx]
            state = district_df.iloc[idx]["state_name"]
            if did not in centroids:
                continue
            candidates = [
                (j, centroids[ids[j]].distance(centroids[did]))
                for j in range(n)
                if j != idx
                and district_df.iloc[j]["state_name"] == state
                and ids[j] in centroids
            ]
            candidates = sorted(candidates, key=lambda x: x[1])[:3]
            for j, _ in candidates:
                adj[idx, j] = 1.0
                adj[j, idx] = 1.0

    return torch.tensor(adj, dtype=torch.float32)


def _fit_scaler(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # values: [S, N, T, F], mask: [S, N, T]
    m = mask.astype(bool)[..., None]
    v = values[m.repeat(values.shape[-1], axis=-1)].reshape(-1, values.shape[-1])
    if v.size == 0:
        mean = np.zeros(values.shape[-1], dtype=np.float32)
        std = np.ones(values.shape[-1], dtype=np.float32)
        return mean, std
    mean = v.mean(axis=0).astype(np.float32)
    std = v.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _apply_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    e = y_pred - y_true
    rmse = float(np.sqrt(np.mean(e**2)))
    mae = float(np.mean(np.abs(e)))
    denom = np.abs(y_true)
    valid = denom > 1e-6
    if np.any(valid):
        mape = float(np.mean(np.abs(e[valid]) / denom[valid]) * 100.0)
    else:
        mape = float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _prediction_rows(
    split_name: str,
    years: Sequence[int],
    district_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    operational_label: str,
) -> pd.DataFrame:
    # y_true/y_pred: [S_split, N]
    rows: List[dict] = []
    district_id = district_df["district_id"].to_numpy()
    state_name = district_df["state_name"].to_numpy()
    district_name = district_df["district_name"].to_numpy()
    for yi, year in enumerate(years):
        for di in range(len(district_df)):
            rows.append(
                {
                    "operational_date": operational_label,
                    "split": split_name,
                    "season_year": int(year),
                    "district_id": str(district_id[di]),
                    "state_name": str(state_name[di]),
                    "district_name": str(district_name[di]),
                    "actual_yield_kg_per_ha": float(y_true[yi, di]),
                    "predicted_yield_kg_per_ha": float(y_pred[yi, di]),
                    "error_kg_per_ha": float(y_pred[yi, di] - y_true[yi, di]),
                    "abs_error_kg_per_ha": float(abs(y_pred[yi, di] - y_true[yi, di])),
                }
            )
    return pd.DataFrame(rows)


def _train_one_operational_day(
    operational_label: str,
    district_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    weather_root: Path,
    sat_map: Dict[Tuple[str, str], pd.DataFrame],
    sangrur_weights: Dict[int, Tuple[float, float]],
    adj: torch.Tensor,
    seasons: List[int],
    split_mode: str,
    split_seed: int,
    horizon_days: int,
    sat_seq_len: int,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    embed_out_dir: Path,
    show_progress: bool,
    log_every: int,
    pred_out_dir: Optional[Path] = None,
    model_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    # Build graph-level tensors (one graph per season year).
    district_ids = district_df["district_id"].tolist()
    season_meta: List[SeasonSelection] = []
    weather_list: List[np.ndarray] = []
    weather_mask_list: List[np.ndarray] = []
    sat_list: List[np.ndarray] = []
    sat_mask_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for year in seasons:
        w_path = weather_root / f"s2s_district_daily_{year}.parquet"
        if not w_path.exists():
            raise FileNotFoundError(f"Missing weather file: {w_path}")
        weather_df = _load_weather_year(w_path)
        sel = _pick_issue_date_for_operational_label(
            weather_year_df=weather_df,
            season_year=year,
            operational_label=operational_label,
            horizon_days=horizon_days,
        )
        season_meta.append(sel)

        wx, wm = _weather_tensor_for_season(
            weather_df=weather_df,
            district_ids=district_ids,
            issue_date=sel.issue_date,
            horizon_days=horizon_days,
        )

        sx, sm = _satellite_tensor_for_season(
            sat_map=sat_map,
            district_df=district_df,
            season_year=year,
            op_date=sel.issue_date,
            seq_len=sat_seq_len,
            sangrur_weights=sangrur_weights,
        )
        active_steps = _sat_step_active_mask(
            season_year=year,
            seq_len=sat_seq_len,
            op_date=sel.issue_date,
        )
        sx, sm = _impute_satellite_by_state_mean(
            sat_x=sx,
            sat_mask=sm,
            district_df=district_df,
            active_steps=active_steps,
        )

        yy = (
            yield_df[yield_df["season_start_year"] == year]
            .set_index("district_id")
            .reindex(district_ids)["yield_kg_per_ha"]
            .to_numpy(dtype=np.float32)
        )
        if np.isnan(yy).any():
            miss_n = int(np.isnan(yy).sum())
            raise RuntimeError(f"Yield labels missing for season {year}, districts missing={miss_n}.")

        weather_list.append(wx)
        weather_mask_list.append(wm)
        sat_list.append(sx)
        sat_mask_list.append(sm)
        y_list.append(yy)

    weather_arr = np.stack(weather_list, axis=0)       # [S, N, Tw, Fw]
    weather_mask_arr = np.stack(weather_mask_list, 0)  # [S, N, Tw]
    sat_arr = np.stack(sat_list, axis=0)               # [S, N, Ts, Fs]
    sat_mask_arr = np.stack(sat_mask_list, 0)          # [S, N, Ts]
    y_arr = np.stack(y_list, axis=0)                   # [S, N]

    train_years, val_years, test_years = _season_split(
        years=seasons,
        mode=split_mode,
        seed=split_seed,
    )
    year_to_idx = {y: i for i, y in enumerate(seasons)}
    idx_train = np.array([year_to_idx[y] for y in train_years], dtype=int)
    idx_val = np.array([year_to_idx[y] for y in val_years], dtype=int)
    idx_test = np.array([year_to_idx[y] for y in test_years], dtype=int)

    # Fit scalers on train split only.
    w_mean, w_std = _fit_scaler(weather_arr[idx_train], weather_mask_arr[idx_train])
    s_mean, s_std = _fit_scaler(sat_arr[idx_train], sat_mask_arr[idx_train])
    weather_arr = _apply_scaler(weather_arr, w_mean, w_std)
    sat_arr = _apply_scaler(sat_arr, s_mean, s_std)

    # To tensors.
    train_w = _to_tensor(weather_arr[idx_train], device)
    train_wm = _to_tensor(weather_mask_arr[idx_train], device)
    train_s = _to_tensor(sat_arr[idx_train], device)
    train_sm = _to_tensor(sat_mask_arr[idx_train], device)
    train_y = _to_tensor(y_arr[idx_train], device)

    val_w = _to_tensor(weather_arr[idx_val], device)
    val_wm = _to_tensor(weather_mask_arr[idx_val], device)
    val_s = _to_tensor(sat_arr[idx_val], device)
    val_sm = _to_tensor(sat_mask_arr[idx_val], device)
    val_y = _to_tensor(y_arr[idx_val], device)

    test_w = _to_tensor(weather_arr[idx_test], device)
    test_wm = _to_tensor(weather_mask_arr[idx_test], device)
    test_s = _to_tensor(sat_arr[idx_test], device)
    test_sm = _to_tensor(sat_mask_arr[idx_test], device)
    test_y = _to_tensor(y_arr[idx_test], device)

    if model_kwargs is None:
        model_kwargs = {}
    model = DualChannelInformerGAT(
        weather_input_dim=len(WEATHER_FEATURES),
        sat_input_dim=len(SAT_FEATURES),
        **model_kwargs,
    ).to(device)
    model_total_params = int(sum(p.numel() for p in model.parameters()))
    model_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state: Optional[dict] = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    epochs_ran = 0

    adj_device = adj.to(device)
    run_t0 = time.perf_counter()
    progress = _EpochProgress(
        total=epochs,
        prefix=f"[op={operational_label}]",
    ) if show_progress else None

    for ep in range(1, epochs + 1):
        epochs_ran = ep
        model.train()
        optimizer.zero_grad()
        pred, _ = model(train_w, train_wm, train_s, train_sm, adj_device)
        loss = criterion(pred, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_w, val_wm, val_s, val_sm, adj_device)
            val_loss = criterion(val_pred, val_y).item()
            if not np.isfinite(val_loss):
                raise RuntimeError(
                    f"Validation loss became non-finite at epoch {ep}. "
                    "Check masks/features for all-zero sequences or NaNs."
                )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if progress is not None and (ep == 1 or ep % max(1, log_every) == 0 or ep == epochs):
            progress.update(ep, float(loss.item()), float(val_loss))
        elif ep == 1 or ep % 20 == 0:
            _log(
                f"[op={operational_label}] epoch={ep:03d} "
                f"train_mse={loss.item():.4f} val_mse={val_loss:.4f}"
            )

    if progress is not None:
        progress.close()

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_pred, _ = model(train_w, train_wm, train_s, train_sm, adj_device)
        val_pred, _ = model(val_w, val_wm, val_s, val_sm, adj_device)
        test_pred, _ = model(test_w, test_wm, test_s, test_sm, adj_device)

    train_y_np = train_y.detach().cpu().numpy()
    val_y_np = val_y.detach().cpu().numpy()
    test_y_np = test_y.detach().cpu().numpy()
    train_pred_np = train_pred.detach().cpu().numpy()
    val_pred_np = val_pred.detach().cpu().numpy()
    test_pred_np = test_pred.detach().cpu().numpy()

    train_metrics = _metrics(train_y_np.ravel(), train_pred_np.ravel())
    val_metrics = _metrics(val_y_np.ravel(), val_pred_np.ravel())
    test_metrics = _metrics(test_y_np.ravel(), test_pred_np.ravel())

    pred_path: Optional[Path] = None
    if pred_out_dir is not None:
        pred_out_dir.mkdir(parents=True, exist_ok=True)
        pred_frames = [
            _prediction_rows(
                split_name="train",
                years=train_years,
                district_df=district_df,
                y_true=train_y_np,
                y_pred=train_pred_np,
                operational_label=operational_label,
            ),
            _prediction_rows(
                split_name="val",
                years=val_years,
                district_df=district_df,
                y_true=val_y_np,
                y_pred=val_pred_np,
                operational_label=operational_label,
            ),
            _prediction_rows(
                split_name="test",
                years=test_years,
                district_df=district_df,
                y_true=test_y_np,
                y_pred=test_pred_np,
                operational_label=operational_label,
            ),
        ]
        pred_df = pd.concat(pred_frames, ignore_index=True)
        pred_key = operational_label.replace("/", "-")
        pred_path = pred_out_dir / f"predictions_opdate_{pred_key}.csv"
        pred_df.to_csv(pred_path, index=False)

    # Save embeddings for all seasons with best model.
    with torch.no_grad():
        all_w = _to_tensor(weather_arr, device)
        all_wm = _to_tensor(weather_mask_arr, device)
        all_s = _to_tensor(sat_arr, device)
        all_sm = _to_tensor(sat_mask_arr, device)
        _, emb = model(all_w, all_wm, all_s, all_sm, adj_device)
    emb_np = emb.detach().cpu().numpy()  # [S, N, D]
    embed_out_dir.mkdir(parents=True, exist_ok=True)
    emb_key = operational_label.replace("/", "-")
    emb_path = embed_out_dir / f"informer_gat_embeddings_opdate_{emb_key}.npz"
    np.savez_compressed(
        emb_path,
        embeddings=emb_np,
        seasons=np.array(seasons, dtype=np.int32),
        district_ids=np.array(district_ids),
    )
    run_seconds = time.perf_counter() - run_t0

    return {
        "operational_date": operational_label,
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years,
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_mape": train_metrics["mape"],
        "train_r2": train_metrics["r2"],
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_mape": val_metrics["mape"],
        "val_r2": val_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_mape": test_metrics["mape"],
        "test_r2": test_metrics["r2"],
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "train_seconds": float(run_seconds),
        "model_total_params": model_total_params,
        "model_trainable_params": model_trainable_params,
        "embedding_file": str(emb_path),
        "prediction_file": str(pred_path) if pred_path is not None else "",
        "season_issue_dates": [
            {
                "season_year": s.season_year,
                "target_date": str(s.target_date.date()),
                "issue_date": str(s.issue_date.date()),
                "harvest_date": str(s.harvest_date.date()),
                "horizon_days": s.horizon_days,
            }
            for s in season_meta
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Dual-Channel Informer + GAT yield model over operational calendar dates "
            "(e.g., Dec 05, Dec 15, ... Feb 05) and export node embeddings."
        )
    )
    parser.add_argument("--districts", type=str, default="data/processed/s2s_district/districts.parquet")
    parser.add_argument("--yield-file", type=str, default="data/yields/apy_query_report_model_ready_119.csv")
    parser.add_argument("--weather-dir", type=str, default="data/processed/s2s_district")
    parser.add_argument(
        "--sat-merged-dir",
        type=str,
        default="Remote sensing data/sentinel2_wheat_pipeline/output/merged",
    )
    parser.add_argument(
        "--sangrur-audit",
        type=str,
        default="data/yields/apy_query_report_sangrur_malerkotla_audit.csv",
    )
    parser.add_argument("--data-config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--split-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--years", type=int, nargs="*", default=[2017, 2018, 2019, 2020, 2021, 2022])
    parser.add_argument(
        "--operational-dates",
        type=str,
        nargs="*",
        default=["12-05", "12-15", "12-25", "01-04", "01-14", "01-24", "02-05"],
        help="Operational prediction dates as MM-DD within each season year.",
    )
    parser.add_argument("--forecast-horizon", type=int, default=46)
    parser.add_argument("--sat-seq-len", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--weather-d-model", type=int, default=64)
    parser.add_argument("--sat-d-model", type=int, default=64)
    parser.add_argument("--weather-heads", type=int, default=4)
    parser.add_argument("--sat-heads", type=int, default=4)
    parser.add_argument("--weather-layers", type=int, default=2)
    parser.add_argument("--sat-layers", type=int, default=2)
    parser.add_argument("--weather-d-ff", type=int, default=128)
    parser.add_argument("--sat-d-ff", type=int, default=128)
    parser.add_argument("--gat-hidden", type=int, default=64)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-weather-distil", action="store_true")
    parser.add_argument("--no-sat-distil", action="store_true")
    parser.add_argument("--no-distil", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out-dir", type=str, default="experiments/informer_gat")
    args = parser.parse_args()

    _set_global_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embed_dir = out_dir / "embeddings"
    pred_dir = out_dir / "predictions" if args.save_predictions else None

    district_df = _load_district_table(Path(args.districts))
    yield_df = _load_yield_panel(Path(args.yield_file))
    sat_map = _load_satellite_merged(Path(args.sat_merged_dir))
    sangrur_weights = _load_sangrur_malerkotla_weights(Path(args.sangrur_audit))
    adj = _build_adjacency_from_boundaries(
        district_df=district_df,
        config_path=Path(args.data_config),
    )

    seasons = sorted([int(y) for y in args.years])
    target_ids = set(district_df["district_id"].tolist())
    y_sub = yield_df[yield_df["season_start_year"].isin(seasons)]
    y_cov = set(y_sub["district_id"].unique().tolist())
    if len(target_ids - y_cov) > 0:
        _log(f"Warning: yield file missing some districts for selected years: {len(target_ids - y_cov)}")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        try:
            device = torch.device(args.device)
        except Exception:
            device = torch.device("cpu")

    _log(f"Using device: {device}")
    _log(f"Global seed: {int(args.seed)}")
    weather_distil = not (args.no_weather_distil or args.no_distil)
    sat_distil = not (args.no_sat_distil or args.no_distil)
    model_kwargs: Dict[str, object] = {
        "weather_d_model": int(args.weather_d_model),
        "sat_d_model": int(args.sat_d_model),
        "weather_heads": int(args.weather_heads),
        "sat_heads": int(args.sat_heads),
        "weather_layers": int(args.weather_layers),
        "sat_layers": int(args.sat_layers),
        "weather_d_ff": int(args.weather_d_ff),
        "sat_d_ff": int(args.sat_d_ff),
        "dropout": float(args.dropout),
        "gat_hidden": int(args.gat_hidden),
        "gat_heads": int(args.gat_heads),
        "gat_layers": int(args.gat_layers),
        "weather_distil": bool(weather_distil),
        "sat_distil": bool(sat_distil),
    }
    model_preview = DualChannelInformerGAT(
        weather_input_dim=len(WEATHER_FEATURES),
        sat_input_dim=len(SAT_FEATURES),
        **model_kwargs,
    )
    model_total_params = int(sum(p.numel() for p in model_preview.parameters()))
    model_trainable_params = int(sum(p.numel() for p in model_preview.parameters() if p.requires_grad))
    _log(
        "Model config: "
        f"w_d_model={args.weather_d_model}, s_d_model={args.sat_d_model}, "
        f"w_layers={args.weather_layers}, s_layers={args.sat_layers}, "
        f"w_heads={args.weather_heads}, s_heads={args.sat_heads}, "
        f"gat_hidden={args.gat_hidden}, gat_heads={args.gat_heads}, gat_layers={args.gat_layers}, "
        f"weather_distil={weather_distil}, sat_distil={sat_distil}"
    )
    _log(
        f"Model params: total={model_total_params:,}, trainable={model_trainable_params:,}"
    )
    results: List[dict] = []
    all_t0 = time.perf_counter()
    for op_label in args.operational_dates:
        _log(f"\n=== Operational date: {op_label} ===")
        op_t0 = time.perf_counter()
        res = _train_one_operational_day(
            operational_label=op_label,
            district_df=district_df,
            yield_df=yield_df,
            weather_root=Path(args.weather_dir),
            sat_map=sat_map,
            sangrur_weights=sangrur_weights,
            adj=adj,
            seasons=seasons,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            horizon_days=args.forecast_horizon,
            sat_seq_len=args.sat_seq_len,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            embed_out_dir=embed_dir,
            show_progress=not args.no_progress,
            log_every=args.log_every,
            pred_out_dir=pred_dir,
            model_kwargs=model_kwargs,
        )
        res["seed"] = int(args.seed)
        results.append(res)
        op_dt = time.perf_counter() - op_t0
        _log(
            f"op-{op_label}: val_rmse={res['val_rmse']:.3f}, val_mape={res['val_mape']:.2f}%, val_r2={res['val_r2']:.3f}, "
            f"test_rmse={res['test_rmse']:.3f}, test_mape={res['test_mape']:.2f}%, test_r2={res['test_r2']:.3f}, "
            f"epochs={res['epochs_ran']}, best_epoch={res['best_epoch']}, "
            f"time={_fmt_seconds(op_dt)}"
        )

    res_df = pd.DataFrame(results)
    if not res_df.empty and "operational_date" in res_df.columns:
        # Keep requested order for readability.
        order = {x: i for i, x in enumerate(args.operational_dates)}
        res_df["_ord"] = res_df["operational_date"].map(lambda x: order.get(x, 999))
        res_df = res_df.sort_values("_ord").drop(columns=["_ord"])
    csv_path = out_dir / "operational_date_metrics.csv"
    json_path = out_dir / "operational_date_metrics.json"
    res_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=2))
    _log(f"\nSaved metrics: {csv_path}")
    _log(f"Saved metrics json: {json_path}")
    _log(f"Total runtime: {_fmt_seconds(time.perf_counter() - all_t0)}")

    if not res_df.empty:
        best = res_df.sort_values("val_rmse").iloc[0]
        _log(
            "Best by val_rmse: "
            f"op-{best['operational_date']}, "
            f"val_rmse={best['val_rmse']:.3f}, test_rmse={best['test_rmse']:.3f}"
        )


if __name__ == "__main__":
    main()
