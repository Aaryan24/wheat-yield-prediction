from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SAT_BASE_FEATURES = ["B7", "B8", "B8A", "B12"]

DISTRICT_ALIASES = {
    "gurgaon": "gurugram",
    "mewat": "nuh",
    "hisar": "hissar",
    "sonipat": "sonepat",
    "yamunanagar": "yamuna nagar",
    "ferozepur": "firozpur",
    "sas nagar": "mohali",
    "s a s nagar": "mohali",
    "s a s nagar sahibzada ajit singh nagar": "mohali",
    "shaheed bhagat singh nagar": "nawan shehar",
    "shahid bhagat singh nagar": "nawan shehar",
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


def norm(text: str) -> str:
    x = str(text).strip().lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def canon_district(text: str) -> str:
    n = norm(text)
    return DISTRICT_ALIASES.get(n, n)


def parse_operational_target_date(label: str, season_year: int) -> pd.Timestamp:
    m = re.match(r"^\s*(\d{2})[-/](\d{2})\s*$", str(label))
    if not m:
        raise ValueError(f"Operational date label must be MM-DD. Got: {label}")
    month = int(m.group(1))
    day = int(m.group(2))
    year = season_year if month >= 9 else season_year + 1
    return pd.Timestamp(dt.date(year, month, day))


def sat_step_active_mask(season_year: int, seq_len: int, op_date: pd.Timestamp) -> np.ndarray:
    season_start = pd.Timestamp(dt.date(season_year, 10, 1))
    active = np.zeros(seq_len, dtype=np.float32)
    for t in range(seq_len):
        end_date = season_start + pd.Timedelta(days=(t * 5) + 5)
        if end_date <= op_date:
            active[t] = 1.0
    return active


def _parse_sat_file_key(file_name: str) -> Tuple[str, str]:
    stem = file_name.replace("_remote_sensing_data.csv", "")
    if stem.startswith("Uttar_Pradesh_"):
        return "Uttar Pradesh", stem[len("Uttar_Pradesh_") :]
    if stem.startswith("Haryana_"):
        return "Haryana", stem[len("Haryana_") :]
    if stem.startswith("Punjab_"):
        return "Punjab", stem[len("Punjab_") :]
    raise ValueError(f"Unrecognized satellite merged filename: {file_name}")


def load_satellite_merged(merged_dir: Path) -> Dict[Tuple[str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for p in sorted(merged_dir.glob("*_remote_sensing_data.csv")):
        state_name, district_name = _parse_sat_file_key(p.name)
        key = (norm(state_name), canon_district(district_name))
        df = pd.read_csv(p)
        needed = ["year", "time_step", "end_date"] + SAT_BASE_FEATURES
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise RuntimeError(f"Satellite file {p} missing columns: {missing}")
        d = df[needed].copy()
        d["end_date"] = pd.to_datetime(d["end_date"])
        for c in SAT_BASE_FEATURES:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        out[key] = d
    return out


def load_sangrur_malerkotla_weights(path: Path) -> Dict[int, Tuple[float, float]]:
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


def load_sangrur_malerkotla_weights_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["season_year", "sangrur_area_before", "malerkotla_area_added", "weight_sangrur", "weight_malerkotla"]
        )
    df = pd.read_csv(path)
    if "season_label" not in df.columns:
        return pd.DataFrame(
            columns=["season_year", "sangrur_area_before", "malerkotla_area_added", "weight_sangrur", "weight_malerkotla"]
        )
    rows: List[dict] = []
    for row in df.itertuples(index=False):
        m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", str(row.season_label))
        if not m:
            continue
        year = int(m.group(1))
        ws = float(getattr(row, "sangrur_area_before", 0.0) or 0.0)
        wm = float(getattr(row, "malerkotla_area_added", 0.0) or 0.0)
        denom = ws + wm
        rows.append(
            {
                "season_year": year,
                "sangrur_area_before": ws,
                "malerkotla_area_added": wm,
                "weight_sangrur": float(ws / denom) if denom > 0 else 1.0,
                "weight_malerkotla": float(wm / denom) if denom > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("season_year").reset_index(drop=True)


def satellite_tensor_for_season(
    sat_map: Dict[Tuple[str, str], pd.DataFrame],
    district_df: pd.DataFrame,
    season_year: int,
    op_date: pd.Timestamp,
    seq_len: int,
    sangrur_weights: Dict[int, Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    n = len(district_df)
    f = len(SAT_BASE_FEATURES)
    x = np.zeros((n, seq_len, f), dtype=np.float32)
    m = np.zeros((n, seq_len), dtype=np.float32)

    punjab_key = norm("Punjab")
    sangrur_key = canon_district("Sangrur")
    malerkotla_key = canon_district("Malerkotla")

    missing_districts: List[str] = []

    for i, row in enumerate(district_df.itertuples(index=False)):
        key = (row.state_norm, row.district_norm)
        sat_df = sat_map.get(key)

        # Merge Malerkotla into Sangrur to keep label/feature entity aligned.
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
                    ws = float(ws)
                    wm = float(wm)
                    if wm > 0 and ws + wm > 0:
                        merged = sang.merge(
                            mal[["time_step"] + SAT_BASE_FEATURES],
                            on="time_step",
                            how="left",
                            suffixes=("_s", "_m"),
                        )
                        for c in SAT_BASE_FEATURES:
                            s_val = merged[f"{c}_s"].to_numpy(dtype=np.float32)
                            m_val = merged[f"{c}_m"].to_numpy(dtype=np.float32)
                            m_val = np.where(np.isnan(m_val), s_val, m_val)
                            merged[c] = (s_val * ws + m_val * wm) / (ws + wm)
                        sat_df = merged[["year", "time_step", "end_date"] + SAT_BASE_FEATURES].copy()
                    else:
                        sat_df = sang
                else:
                    sat_df = sang
            else:
                sat_df = None

        if sat_df is None:
            missing_districts.append(str(row.district_id))
            continue

        d = sat_df[sat_df["year"] == season_year].copy()
        if d.empty:
            missing_districts.append(str(row.district_id))
            continue
        d = d[d["end_date"] <= op_date].copy()
        if d.empty:
            missing_districts.append(str(row.district_id))
            continue

        d = d.sort_values("time_step")
        for rec in d.itertuples(index=False):
            t = int(rec.time_step)
            if t < 0 or t >= seq_len:
                continue
            vals = [float(getattr(rec, c)) if pd.notna(getattr(rec, c)) else np.nan for c in SAT_BASE_FEATURES]
            arr = np.array(vals, dtype=np.float32)
            if np.isnan(arr).any():
                continue
            x[i, t, :] = arr
            m[i, t] = 1.0

    return x, m, sorted(set(missing_districts))


def impute_satellite_by_state_mean(
    sat_x: np.ndarray,
    sat_mask: np.ndarray,
    district_df: pd.DataFrame,
    active_steps: np.ndarray,
    set_mask_valid: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    out = sat_x.copy()
    out_mask = sat_mask.copy()
    t = out.shape[1]
    assert len(active_steps) == t

    for state in sorted(district_df["state_name"].unique()):
        idx = district_df.index[district_df["state_name"] == state].tolist()
        if not idx:
            continue

        state_x = out[idx, :, :]
        state_m = out_mask[idx, :]

        for ti in range(t):
            if active_steps[ti] <= 0:
                continue
            valid_nodes = state_m[:, ti] > 0
            if not np.any(valid_nodes):
                continue
            mean_val = state_x[valid_nodes, ti, :].mean(axis=0)
            miss_nodes = ~valid_nodes
            if np.any(miss_nodes):
                state_x[miss_nodes, ti, :] = mean_val
                if set_mask_valid:
                    state_m[miss_nodes, ti] = 1.0

        out[idx, :, :] = state_x
        out_mask[idx, :] = state_m

    return out, out_mask


def coverage_rows_for_sample(
    district_df: pd.DataFrame,
    season_year: int,
    operational_date: str,
    active_steps: np.ndarray,
    sat_mask_before: np.ndarray,
    sat_mask_after: np.ndarray,
    missing_district_ids: Optional[List[str]] = None,
) -> List[dict]:
    rows: List[dict] = []
    active_idx = np.where(active_steps > 0.5)[0]
    if len(active_idx) == 0:
        return rows

    miss_set = set(missing_district_ids or [])

    for state in sorted(district_df["state_name"].unique()):
        idx = district_df.index[district_df["state_name"] == state].to_numpy(dtype=int)
        if len(idx) == 0:
            continue

        before = sat_mask_before[idx][:, active_idx]
        after = sat_mask_after[idx][:, active_idx]
        dids = district_df.iloc[idx]["district_id"].astype(str).tolist()
        missing_ids = sorted([d for d in dids if d in miss_set])

        rows.append(
            {
                "season_year": int(season_year),
                "operational_date": str(operational_date),
                "state_name": state,
                "active_steps": int(len(active_idx)),
                "districts_total": int(len(idx)),
                "valid_ratio_before": float(before.mean()),
                "valid_ratio_after": float(after.mean()),
                "districts_with_any_before": int((before.sum(axis=1) > 0).sum()),
                "districts_with_any_after": int((after.sum(axis=1) > 0).sum()),
                "districts_fully_missing_from_source": int(len(missing_ids)),
                "districts_fully_missing_ids": "|".join(missing_ids),
            }
        )

    return rows
