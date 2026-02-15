from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr


DATE_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})")


@dataclass(frozen=True)
class IssueFiles:
    date: datetime
    temp: Path
    wind: Path
    accum: Path


def _parse_date(path: Path) -> datetime:
    match = DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"Unable to parse date from {path.name}")
    return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _collect_issue_files(root: Path) -> List[IssueFiles]:
    temp = { _parse_date(p): p for p in root.glob("s2s_temp_*.grib") }
    wind = { _parse_date(p): p for p in root.glob("s2s_wind_*.grib") }
    accum = { _parse_date(p): p for p in root.glob("s2s_accum_*.grib") }

    common = sorted(set(temp) & set(wind) & set(accum))
    return [IssueFiles(date=d, temp=temp[d], wind=wind[d], accum=accum[d]) for d in common]


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute weighted mean/std with missing-value handling.

    values: (steps, n_cells)
    weights: (n_districts, n_cells)
    """
    valid = np.isfinite(values)
    safe_values = np.where(valid, values, 0.0)
    weights_t = weights.T

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        numerator = safe_values @ weights_t
        denom = valid.astype(np.float32) @ weights_t
        mean = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom != 0)

        numerator_sq = (safe_values ** 2) @ weights_t
        mean_sq = np.divide(numerator_sq, denom, out=np.zeros_like(numerator_sq), where=denom != 0)
    var = mean_sq - mean ** 2
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    return mean, std


def _flatten_grid(values: np.ndarray, missing_value: float | None = None) -> np.ndarray:
    # values shape: (steps, lat, lon)
    steps, n_lat, n_lon = values.shape
    flat = values.reshape(steps, n_lat * n_lon)
    if missing_value is not None:
        flat = np.where(flat == missing_value, np.nan, flat)
    # Guard against sentinel values that aren't an exact match
    flat = np.where(np.abs(flat) > 1e20, np.nan, flat)
    return flat


def _lead_hours_from_step(step: np.ndarray) -> np.ndarray:
    return step.astype("timedelta64[h]").astype(int)


def _lead_days_from_step(step: np.ndarray) -> np.ndarray:
    return step.astype("timedelta64[D]").astype(int)


def _daily_from_accum(values: np.ndarray) -> np.ndarray:
    daily = np.empty_like(values)
    daily[0] = values[0]
    daily[1:] = values[1:] - values[:-1]
    daily = np.maximum(daily, 0.0)
    return daily


def _daily_mean_from_6h(values: np.ndarray) -> np.ndarray:
    # values shape: (steps, n_cells), steps should be 184 (46 days * 4)
    days = values.shape[0] // 4
    return values.reshape(days, 4, values.shape[1]).mean(axis=1)


def _daily_max_from_6h(values: np.ndarray) -> np.ndarray:
    days = values.shape[0] // 4
    return values.reshape(days, 4, values.shape[1]).max(axis=1)


def _daily_min_from_6h(values: np.ndarray) -> np.ndarray:
    days = values.shape[0] // 4
    return values.reshape(days, 4, values.shape[1]).min(axis=1)


def aggregate_issue_date(
    issue: IssueFiles,
    weights: np.ndarray,
    district_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (temp_6h_df, daily_df) for a single issue date."""
    # Disable persistent cfgrib index files; avoids failures from stale/corrupted .idx sidecars.
    ds_temp = xr.open_dataset(issue.temp, engine="cfgrib", backend_kwargs={"indexpath": ""})
    ds_wind = xr.open_dataset(issue.wind, engine="cfgrib", backend_kwargs={"indexpath": ""})
    ds_accum = xr.open_dataset(issue.accum, engine="cfgrib", backend_kwargs={"indexpath": ""})

    # Expected layout from downloaded S2S files:
    # - temp/wind: 46 days at 6-hourly resolution -> 184 steps
    # - accum: 46 daily accumulations -> 46 steps
    if ds_temp.sizes.get("step") != 184:
        raise ValueError(f"Unexpected temp step count ({ds_temp.sizes.get('step')}) for {issue.temp}")
    if ds_wind.sizes.get("step") != 184:
        raise ValueError(f"Unexpected wind step count ({ds_wind.sizes.get('step')}) for {issue.wind}")
    if ds_accum.sizes.get("step") != 46:
        raise ValueError(f"Unexpected accum step count ({ds_accum.sizes.get('step')}) for {issue.accum}")

    # Temperature (6h)
    mx_missing = ds_temp["mx2t6"].attrs.get("GRIB_missingValue")
    mx_6h = _flatten_grid(ds_temp["mx2t6"].values, mx_missing)
    mx_mean, mx_std = _weighted_mean_std(mx_6h, weights)
    mn_missing = ds_temp["mn2t6"].attrs.get("GRIB_missingValue")
    mn_6h = _flatten_grid(ds_temp["mn2t6"].values, mn_missing)
    mn_mean, mn_std = _weighted_mean_std(mn_6h, weights)

    lead_hours = _lead_hours_from_step(ds_temp["step"].values)
    temp_rows = []
    for d_idx, district in district_table.iterrows():
        for t_idx, lead_h in enumerate(lead_hours):
            temp_rows.append(
                {
                    "district_id": district["district_id"],
                    "district_key": district["district_key"],
                    "state_name": district["state_name"],
                    "district_name": district["district_name"],
                    "issue_date": issue.date.date().isoformat(),
                    "lead_step_hours": int(lead_h),
                    "mx2t6_mean": float(mx_mean[t_idx, d_idx]),
                    "mx2t6_std": float(mx_std[t_idx, d_idx]),
                    "mn2t6_mean": float(mn_mean[t_idx, d_idx]),
                    "mn2t6_std": float(mn_std[t_idx, d_idx]),
                }
            )

    temp_df = pd.DataFrame(temp_rows)

    # Temperature (6h -> daily max/min)
    mx_daily = _daily_max_from_6h(mx_6h)
    mn_daily = _daily_min_from_6h(mn_6h)
    tmax_mean, tmax_std = _weighted_mean_std(mx_daily, weights)
    tmin_mean, tmin_std = _weighted_mean_std(mn_daily, weights)

    # Wind (6h -> daily mean)
    u10_missing = ds_wind["u10"].attrs.get("GRIB_missingValue")
    v10_missing = ds_wind["v10"].attrs.get("GRIB_missingValue")
    u10 = _flatten_grid(ds_wind["u10"].values, u10_missing)
    v10 = _flatten_grid(ds_wind["v10"].values, v10_missing)
    speed = np.sqrt(u10 ** 2 + v10 ** 2)
    speed_daily = _daily_mean_from_6h(speed)
    wind_mean, wind_std = _weighted_mean_std(speed_daily, weights)

    # Accumulation (daily)
    tp_missing = ds_accum["tp"].attrs.get("GRIB_missingValue")
    ssrd_missing = ds_accum["ssrd"].attrs.get("GRIB_missingValue")
    tp = _flatten_grid(ds_accum["tp"].values, tp_missing)
    ssrd = _flatten_grid(ds_accum["ssrd"].values, ssrd_missing)
    tp_daily = _daily_from_accum(tp)
    ssrd_daily = _daily_from_accum(ssrd)
    tp_mean, tp_std = _weighted_mean_std(tp_daily, weights)
    ssrd_mean, ssrd_std = _weighted_mean_std(ssrd_daily, weights)

    lead_days = _lead_days_from_step(ds_accum["step"].values)
    daily_rows = []
    for d_idx, district in district_table.iterrows():
        for t_idx, lead_d in enumerate(lead_days):
            daily_rows.append(
                {
                    "district_id": district["district_id"],
                    "district_key": district["district_key"],
                    "state_name": district["state_name"],
                    "district_name": district["district_name"],
                    "issue_date": issue.date.date().isoformat(),
                    "lead_day": int(lead_d),
                    "tmax_mean": float(tmax_mean[t_idx, d_idx]),
                    "tmax_std": float(tmax_std[t_idx, d_idx]),
                    "tmin_mean": float(tmin_mean[t_idx, d_idx]),
                    "tmin_std": float(tmin_std[t_idx, d_idx]),
                    "tp_mean": float(tp_mean[t_idx, d_idx]),
                    "tp_std": float(tp_std[t_idx, d_idx]),
                    "ssrd_mean": float(ssrd_mean[t_idx, d_idx]),
                    "ssrd_std": float(ssrd_std[t_idx, d_idx]),
                    "wind_speed_mean": float(wind_mean[t_idx, d_idx]),
                    "wind_speed_std": float(wind_std[t_idx, d_idx]),
                }
            )

    daily_df = pd.DataFrame(daily_rows)
    return temp_df, daily_df


def iterate_issue_dates(
    s2s_root: Path,
    start_year: int,
    end_year: int,
) -> Dict[int, List[IssueFiles]]:
    year_map: Dict[int, List[IssueFiles]] = {}
    for year in range(start_year, end_year + 1):
        year_dir = s2s_root / str(year)
        if not year_dir.exists():
            continue
        year_map[year] = _collect_issue_files(year_dir)
    return year_map
