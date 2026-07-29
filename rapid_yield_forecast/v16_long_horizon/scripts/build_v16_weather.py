#!/usr/bin/env python3
"""Pre-clock weather aggregates, at district and state level.

Every window ends on or before 5 March of the harvest year, so nothing here is
visible only after the forecast clock.

The state-level aggregate exists because of a measured fact: the shared
state-season yield shock has sd ~540 kg/ha and correlates -0.48 with
December-February rainfall, but a satellite-only shock model had no skill at
all.  Averaging weather over a state is the natural predictor for a shared
shock, and it was missing from the V15 hierarchy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
RAPID = V16.parent
DATA = V16 / "data"

CLOCK_MD = "03-05"


def season_start_year(dates: pd.Series) -> np.ndarray:
    """A wheat season starting in October of year Y is harvested in Y+1."""
    return np.where(dates.dt.month >= 10, dates.dt.year, dates.dt.year - 1)


def in_window(md: pd.Series, start: str, end: str) -> pd.Series:
    if start <= end:
        return (md >= start) & (md <= end)
    return (md >= start) | (md <= end)


WINDOWS = {
    "dec_feb": ("12-01", "02-28"),
    "jan_feb": ("01-01", "02-28"),
    "feb_mar05": ("02-01", CLOCK_MD),
    "full_preclock": ("11-01", CLOCK_MD),
}


def main() -> None:
    weather = pd.read_parquet(RAPID / "data" / "observed_weather_daily.parquet")
    weather["date"] = pd.to_datetime(weather["date"])
    weather["season_start_year"] = season_start_year(weather["date"])
    weather["md"] = weather["date"].dt.strftime("%m-%d")
    weather["hot30"] = (weather["T2M_MAX"] > 30).astype(float)
    weather["hot32"] = (weather["T2M_MAX"] > 32).astype(float)
    weather["gdd"] = np.clip(
        (weather["T2M_MAX"] + weather["T2M_MIN"]) / 2.0 - 5.0, 0, None)

    keys = ["district_id", "state_name", "season_start_year"]
    frames = []
    for label, (start, end) in WINDOWS.items():
        block = weather[in_window(weather["md"], start, end)]
        agg = block.groupby(keys).agg(**{
            f"wx_{label}_tmax_mean": ("T2M_MAX", "mean"),
            f"wx_{label}_tmax_max": ("T2M_MAX", "max"),
            f"wx_{label}_tmin_mean": ("T2M_MIN", "mean"),
            f"wx_{label}_precip_sum": ("PRECTOTCORR", "sum"),
            f"wx_{label}_solar_mean": ("ALLSKY_SFC_SW_DWN", "mean"),
            f"wx_{label}_rh_mean": ("RH2M", "mean"),
            f"wx_{label}_wind_mean": ("WS2M", "mean"),
            f"wx_{label}_hot30_days": ("hot30", "sum"),
            f"wx_{label}_hot32_days": ("hot32", "sum"),
            f"wx_{label}_gdd_sum": ("gdd", "sum"),
        }).reset_index()
        frames.append(agg.set_index(keys))
    district = pd.concat(frames, axis=1).reset_index()

    # anomalies against each district's own expanding prior mean (legal: shifted)
    district = district.sort_values(["district_id", "season_start_year"])
    value_columns = [c for c in district.columns if c.startswith("wx_")]
    for column in value_columns:
        prior = (district.groupby("district_id")[column]
                 .transform(lambda s: s.shift(1).expanding(min_periods=3).mean()))
        prior_sd = (district.groupby("district_id")[column]
                    .transform(lambda s: s.shift(1).expanding(min_periods=4).std()))
        district[f"wxz__{column}"] = (
            (district[column] - prior) / prior_sd.replace(0, np.nan))

    district.to_parquet(DATA / "v16_weather_district.parquet", index=False)

    state_columns = value_columns + [f"wxz__{c}" for c in value_columns]
    state = (district.groupby(["state_name", "season_start_year"])[state_columns]
             .mean().reset_index())
    state.columns = (["state_name", "season_start_year"]
                     + [f"st_{c}" for c in state_columns])
    state.to_parquet(DATA / "v16_weather_state.parquet", index=False)

    print(f"district rows {len(district)}, "
          f"years {district.season_start_year.min()}-{district.season_start_year.max()}")
    print(f"state rows {len(state)}, weather columns {len(value_columns)}")
    print(f"district anomaly columns {len(value_columns)}, "
          f"state columns {len(state_columns)}")


if __name__ == "__main__":
    main()
