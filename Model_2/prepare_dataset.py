#!/usr/bin/env python3
"""
prepare_dataset.py — Combine Remote-Sensing & Weather Data into a Single Dataset
=================================================================================
Reads the three raw data sources and merges them into a single `dataset.npz`
file that `train.py` can load directly.

Sources:
  1. S2S weather forecasts    → data/processed/s2s_district/s2s_district_daily_YYYY.parquet
  2. Sentinel-2 satellite     → Remote sensing data/sentinel2_wheat_pipeline/output/merged/*.csv
  3. Yield labels             → data/yields/apy_query_report_model_ready_119.csv
  4. District adjacency       → built from GeoPackage boundaries

Output:
  Model_2/dataset.npz containing weather_x, weather_mask, sat_x, sat_mask,
  yields, adjacency, district_ids, season_years, district_names, state_names

Usage:
  python Model_2/prepare_dataset.py
  python Model_2/prepare_dataset.py --years 2017 2018 2019 2020 2021 2022
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _log(msg: str) -> None:
    print(f"[prepare_dataset] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Name normalisation (same logic as existing pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

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
    "muktsar": "sri muktsar sahib",
    "shaheed bhagat singh nagar": "nawanshahr",
    # UP
    "allahabad": "prayagraj",
    "sant ravidas nagar": "sant ravi das nagar",
    "sant kabeer nagar": "sant kabir nagar",
}

STATE_ALIASES = {
    "uttar pradesh": "Uttar Pradesh",
    "uttar_pradesh": "Uttar Pradesh",
    "haryana": "Haryana",
    "punjab": "Punjab",
}


def _norm(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _canon_state(text: str) -> str:
    return STATE_ALIASES.get(_norm(text), text)


def _canon_district(text: str) -> str:
    n = _norm(text)
    return DISTRICT_ALIASES.get(n, n)


# ═══════════════════════════════════════════════════════════════════════════════
# Load district table
# ═══════════════════════════════════════════════════════════════════════════════

def _load_district_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["state_norm"] = df["state_name"].map(_norm)
    df["district_norm"] = df["district_name"].map(_canon_district)
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Load yield panel
# ═══════════════════════════════════════════════════════════════════════════════

def _load_yield_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["season_start_year", "yield_kg_per_ha"]:
        if c not in df.columns:
            raise RuntimeError(f"Yield file missing column: {c}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Weather data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_weather_year(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "issue_date" in df.columns:
        df["issue_date"] = pd.to_datetime(df["issue_date"])
    return df


def _parse_operational_target_date(label: str, season_year: int) -> pd.Timestamp:
    month, day = [int(x) for x in label.split("-")]
    year = season_year if month >= 9 else season_year + 1
    return pd.Timestamp(dt.date(year, month, day))


def _pick_issue_date_for_operational_label(
    weather_year_df: pd.DataFrame,
    season_year: int,
    operational_label: str,
    horizon_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, int]:
    """Returns (target_date, issue_date, harvest_date, horizon_days)."""
    target = _parse_operational_target_date(operational_label, season_year)
    harvest = pd.Timestamp(dt.date(season_year + 1, 4, 15))

    available = sorted(weather_year_df["issue_date"].dropna().unique())
    candidates = [d for d in available if d <= target]
    if candidates:
        issue = pd.Timestamp(candidates[-1])
    else:
        issue = pd.Timestamp(available[0]) if available else target

    return target, issue, harvest, horizon_days


def _weather_tensor_for_season(
    weather_df: pd.DataFrame,
    district_ids: List[str],
    issue_date: pd.Timestamp,
    horizon_days: int,
    weather_features: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build [N, H, F] weather tensor and [N, H] mask for one season."""
    n = len(district_ids)
    f = len(weather_features)
    x = np.zeros((n, horizon_days, f), dtype=np.float32)
    m = np.zeros((n, horizon_days), dtype=np.float32)

    sub = weather_df[weather_df["issue_date"] == issue_date]
    id_to_idx = {did: i for i, did in enumerate(district_ids)}

    for row in sub.itertuples(index=False):
        did = str(row.district_id)
        i = id_to_idx.get(did)
        if i is None:
            continue
        lead = int(row.lead_day) if hasattr(row, "lead_day") else int(row.lead_time)
        if lead < 1 or lead > horizon_days:
            continue
        vals = [float(getattr(row, c)) for c in weather_features]
        arr = np.array(vals, dtype=np.float32)
        if np.isnan(arr).any():
            continue
        x[i, lead - 1, :] = arr
        m[i, lead - 1] = 1.0
    return x, m


# ═══════════════════════════════════════════════════════════════════════════════
# Satellite data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_sat_file_key(file_name: str) -> Tuple[str, str]:
    stem = file_name.replace("_remote_sensing_data.csv", "")
    if stem.startswith("Uttar_Pradesh_"):
        return "Uttar Pradesh", stem[len("Uttar_Pradesh_"):]
    if stem.startswith("Haryana_"):
        return "Haryana", stem[len("Haryana_"):]
    if stem.startswith("Punjab_"):
        return "Punjab", stem[len("Punjab_"):]
    raise ValueError(f"Unrecognized satellite filename: {file_name}")


def _load_satellite_merged(
    merged_dir: Path, sat_features: List[str]
) -> Dict[Tuple[str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for p in sorted(merged_dir.glob("*_remote_sensing_data.csv")):
        state_name, district_name = _parse_sat_file_key(p.name)
        key = (_norm(state_name), _canon_district(district_name))
        df = pd.read_csv(p)
        keep_cols = ["year", "time_step", "end_date"] + sat_features
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Satellite file {p} missing columns: {missing}")
        d = df[keep_cols].copy()
        d["end_date"] = pd.to_datetime(d["end_date"])
        for c in sat_features:
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
        match = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", str(row.season_label))
        if not match:
            continue
        year = int(match.group(1))
        ws = float(getattr(row, "sangrur_area_before", 0.0) or 0.0)
        wm = float(getattr(row, "malerkotla_area_added", 0.0) or 0.0)
        out[year] = (ws, wm)
    return out


def _satellite_tensor_for_season(
    sat_map: Dict[Tuple[str, str], pd.DataFrame],
    district_df: pd.DataFrame,
    season_year: int,
    op_date: pd.Timestamp,
    seq_len: int,
    sat_features: List[str],
    sangrur_weights: Dict[int, Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build [N, Ts, Fs] satellite tensor and [N, Ts] mask for one season."""
    n = len(district_df)
    f = len(sat_features)
    x = np.zeros((n, seq_len, f), dtype=np.float32)
    m = np.zeros((n, seq_len), dtype=np.float32)

    punjab_key = _norm("Punjab")
    sangrur_key = _canon_district("Sangrur")
    malerkotla_key = _canon_district("Malerkotla")

    for i, row in enumerate(district_df.itertuples(index=False)):
        key = (row.state_norm, row.district_norm)
        sat_df = sat_map.get(key)

        # Handle Sangrur/Malerkotla merging for Punjab.
        if row.state_norm == punjab_key and row.district_norm == sangrur_key:
            sang = sat_map.get((punjab_key, sangrur_key))
            mal = sat_map.get((punjab_key, malerkotla_key))
            if sang is not None:
                sang = sang[sang["year"] == season_year].copy()
            if mal is not None:
                mal = mal[mal["year"] == season_year].copy()
            if sang is not None and not sang.empty:
                if mal is not None and not mal.empty and season_year in sangrur_weights:
                    ws, wm = sangrur_weights.get(season_year, (1.0, 0.0))
                    ws, wm = float(ws), float(wm)
                    if wm > 0 and ws + wm > 0:
                        merged = sang.merge(
                            mal[["time_step"] + sat_features],
                            on="time_step",
                            how="left",
                            suffixes=("_s", "_m"),
                        )
                        for c in sat_features:
                            s_val = merged[f"{c}_s"].to_numpy(dtype=np.float32)
                            m_val = merged[f"{c}_m"].to_numpy(dtype=np.float32)
                            m_val = np.where(np.isnan(m_val), s_val, m_val)
                            merged[c] = (s_val * ws + m_val * wm) / (ws + wm)
                        sat_df = merged[["year", "time_step", "end_date"] + sat_features].copy()
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
            vals = [
                float(getattr(rec, c)) if pd.notna(getattr(rec, c)) else np.nan
                for c in sat_features
            ]
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
    season_year: int,
    seq_len: int,
    op_date: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill missing satellite steps with same-state mean values."""
    out = sat_x.copy()
    out_mask = sat_mask.copy()
    n, t, f = out.shape

    # Determine active time steps.
    season_start = pd.Timestamp(dt.date(season_year, 10, 1))
    active = np.zeros(t, dtype=np.float32)
    for ti in range(t):
        step_end = season_start + pd.Timedelta(days=(ti + 1) * 5)
        if step_end <= op_date:
            active[ti] = 1.0

    for state in sorted(district_df["state_name"].unique()):
        idx = district_df.index[district_df["state_name"] == state].tolist()
        if not idx:
            continue
        state_x = out[idx, :, :]
        state_m = out_mask[idx, :]
        for ti in range(t):
            if active[ti] == 0:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Adjacency matrix
# ═══════════════════════════════════════════════════════════════════════════════

def _build_adjacency_from_boundaries(
    district_df: pd.DataFrame,
    config_path: Path,
) -> np.ndarray:
    """Build [N, N] binary adjacency from GeoPackage boundaries."""
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
        lookup.get((s, d), None)
        for s, d in zip(gdf["state_norm"], gdf["district_norm"])
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

    # Fallback: connect isolated nodes to 3 nearest same-state neighbours.
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

    return adj


# ═══════════════════════════════════════════════════════════════════════════════
# Main: assemble everything into dataset.npz
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine remote sensing + weather data into a single trainable dataset."
    )
    parser.add_argument(
        "--config", type=str, default="Model_2/config.yaml",
        help="Path to Model_2 config file (relative to repo root).",
    )
    parser.add_argument(
        "--years", type=int, nargs="*", default=None,
        help="Override seasons to include (default: from config).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output path (default: from config).",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    cfg = _load_yaml(Path(args.config))
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    temp_cfg = cfg["temporal"]

    weather_features = feat_cfg["weather"]
    sat_features = feat_cfg["satellite"]
    seasons = args.years or temp_cfg["seasons"]
    operational_label = temp_cfg["operational_date"]
    horizon_days = temp_cfg["forecast_horizon"]
    sat_seq_len = temp_cfg["sat_seq_len"]
    output_path = Path(args.output or data_cfg["dataset_file"])

    _log(f"Seasons: {seasons}")
    _log(f"Operational date: {operational_label}")
    _log(f"Weather features ({len(weather_features)}): {weather_features}")
    _log(f"Satellite features ({len(sat_features)}): {sat_features}")

    # ── Load source data ─────────────────────────────────────────────────
    _log("Loading district table...")
    district_df = _load_district_table(Path(data_cfg["districts"]))
    district_ids = district_df["district_id"].tolist()
    n_districts = len(district_ids)
    _log(f"  → {n_districts} districts")

    _log("Loading yield panel...")
    yield_df = _load_yield_panel(Path(data_cfg["yield_file"]))

    _log("Loading satellite data...")
    sat_map = _load_satellite_merged(Path(data_cfg["sat_merged_dir"]), sat_features)
    _log(f"  → {len(sat_map)} district satellite files")

    sangrur_weights = _load_sangrur_malerkotla_weights(Path(data_cfg["sangrur_audit"]))

    # ── Build adjacency ──────────────────────────────────────────────────
    _log("Building adjacency matrix from boundaries...")
    adj = _build_adjacency_from_boundaries(district_df, Path(data_cfg["data_config"]))
    n_edges = int((adj.sum() - adj.trace()) / 2)
    _log(f"  → {n_districts} nodes, {n_edges} edges")

    # ── Assemble per-season tensors ──────────────────────────────────────
    weather_list, weather_mask_list = [], []
    sat_list, sat_mask_list = [], []
    yield_list = []
    issue_dates = []

    for year in seasons:
        _log(f"Processing season {year}...")

        # Weather
        w_path = Path(data_cfg["weather_dir"]) / f"s2s_district_daily_{year}.parquet"
        if not w_path.exists():
            raise FileNotFoundError(f"Missing weather file: {w_path}")
        weather_df = _load_weather_year(w_path)

        target, issue, harvest, h = _pick_issue_date_for_operational_label(
            weather_df, year, operational_label, horizon_days,
        )
        issue_dates.append(str(issue.date()))

        wx, wm = _weather_tensor_for_season(
            weather_df, district_ids, issue, horizon_days, weather_features,
        )

        # Satellite
        sx, sm = _satellite_tensor_for_season(
            sat_map, district_df, year, issue, sat_seq_len,
            sat_features, sangrur_weights,
        )
        sx, sm = _impute_satellite_by_state_mean(
            sx, sm, district_df, year, sat_seq_len, issue,
        )

        # Yields
        yy = (
            yield_df[yield_df["season_start_year"] == year]
            .set_index("district_id")
            .reindex(district_ids)["yield_kg_per_ha"]
            .to_numpy(dtype=np.float32)
        )
        if np.isnan(yy).any():
            miss = int(np.isnan(yy).sum())
            _log(f"  ⚠ {miss} districts missing yield for {year}, filling with 0")
            yy = np.nan_to_num(yy, nan=0.0)

        weather_list.append(wx)
        weather_mask_list.append(wm)
        sat_list.append(sx)
        sat_mask_list.append(sm)
        yield_list.append(yy)

        _log(
            f"  → weather: {wx.shape}, sat: {sx.shape}, "
            f"weather_coverage={wm.mean():.2%}, sat_coverage={sm.mean():.2%}"
        )

    # ── Stack into arrays ────────────────────────────────────────────────
    weather_x = np.stack(weather_list, axis=0)          # [S, N, Tw, Fw]
    weather_mask = np.stack(weather_mask_list, axis=0)   # [S, N, Tw]
    sat_x = np.stack(sat_list, axis=0)                   # [S, N, Ts, Fs]
    sat_mask = np.stack(sat_mask_list, axis=0)           # [S, N, Ts]
    yields = np.stack(yield_list, axis=0)                # [S, N]

    # ── Save ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        weather_x=weather_x,
        weather_mask=weather_mask,
        sat_x=sat_x,
        sat_mask=sat_mask,
        yields=yields,
        adjacency=adj,
        district_ids=np.array(district_ids),
        season_years=np.array(seasons, dtype=np.int32),
        district_names=district_df["district_name"].to_numpy(),
        state_names=district_df["state_name"].to_numpy(),
        issue_dates=np.array(issue_dates),
        weather_features=np.array(weather_features),
        sat_features=np.array(sat_features),
    )

    _log(f"\n{'='*60}")
    _log(f"Dataset saved to: {output_path}")
    _log(f"  weather_x     : {weather_x.shape}  (S×N×Tw×Fw)")
    _log(f"  weather_mask   : {weather_mask.shape}  (S×N×Tw)")
    _log(f"  sat_x          : {sat_x.shape}  (S×N×Ts×Fs)")
    _log(f"  sat_mask       : {sat_mask.shape}  (S×N×Ts)")
    _log(f"  yields         : {yields.shape}  (S×N)")
    _log(f"  adjacency      : {adj.shape}  (N×N)")
    _log(f"  districts      : {len(district_ids)}")
    _log(f"  seasons        : {seasons}")
    _log(f"  issue_dates    : {issue_dates}")
    _log(f"{'='*60}")


if __name__ == "__main__":
    main()
