#!/usr/bin/env python3
"""Can the shared season shock be predicted, and from what?

Established by measurement:
  * the season-mean shock (sd ~540 kg/ha) is LARGER than within-season district
    variation (sd ~290 kg/ha), so it is the dominant term in the problem;
  * a shock model built from state-aggregated MODIS anomalies alone has no
    skill at all (correlation -0.14 over 57 state-seasons);
  * yet the shock correlates -0.48 with December-February rainfall.

So the failure was the predictor set, not the idea.  This script compares shock
models built from satellite alone, pre-clock weather alone, and both, using
only seasons before each target season.

Every window used ends on or before 5 March, so no post-clock information leaks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

V16 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V16 / "scripts"))
from v16_common import BASELINE, TARGET  # noqa: E402

DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"
TEST_YEARS = list(range(2014, 2023))       # weather starts 2010; keep >=4 train seasons
MIN_TRAIN_SEASONS = 4


def log_anomaly(frame: pd.DataFrame) -> np.ndarray:
    return np.log(np.clip(frame[TARGET].to_numpy(float)
                          / frame[BASELINE].to_numpy(float), 0.3, 3.0))


def fit_ridge(train: pd.DataFrame, columns: list[str], target: str):
    usable = [c for c in columns if train[c].notna().mean() >= 0.6]
    if not usable:
        return None
    x = train[usable].to_numpy(float)
    centre = np.nanmean(x, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    x = np.where(np.isfinite(x), x, centre)
    spread = np.nanstd(x, axis=0)
    spread = np.where(spread > 1e-8, spread, 1.0)
    model = RidgeCV(alphas=np.logspace(0, 5, 30))
    model.fit((x - centre) / spread, train[target].to_numpy(float))
    return model, usable, centre, spread


def apply_ridge(fitted, frame: pd.DataFrame) -> np.ndarray:
    model, usable, centre, spread = fitted
    x = frame[usable].to_numpy(float)
    x = np.where(np.isfinite(x), x, centre)
    return model.predict((x - centre) / spread)


def main() -> None:
    panel = pd.read_parquet(DATA / "v16_panel.parquet")
    groups = json.loads((DATA / "v16_feature_groups.json").read_text())
    panel = panel[panel.tier_long & panel[TARGET].notna()
                  & panel[BASELINE].notna()].copy()
    panel["a"] = log_anomaly(panel)

    shocks = (panel.groupby(["state_name", "season_start_year"])["a"]
              .mean().rename("shock").reset_index())

    satellite = [c for c in groups["modis_anomaly"] if c.startswith("z__")]
    state_satellite = (panel.groupby(["state_name", "season_start_year"])[satellite]
                       .mean().reset_index())
    state_weather = pd.read_parquet(DATA / "v16_weather_state.parquet")

    design = shocks.merge(state_satellite, on=["state_name", "season_start_year"],
                          how="left").merge(
        state_weather, on=["state_name", "season_start_year"], how="left")

    weather_z = [c for c in state_weather.columns if c.startswith("st_wxz__")]
    weather_raw = [c for c in state_weather.columns
                   if c.startswith("st_wx_") and not c.startswith("st_wxz__")]

    # last season's shock for the same state: the cheapest possible predictor
    design = design.sort_values(["state_name", "season_start_year"])
    design["shock_lag_1"] = design.groupby("state_name")["shock"].shift(1)

    variants = {
        # weather columns only exist from 2010, so any variant using them must
        # restrict its TRAINING seasons to 2010+ or the availability filter
        # silently drops every weather column.
        "satellite_only": (satellite, 2000),
        "preclock_weather_only": (weather_z + weather_raw, 2010),
        "satellite_plus_weather": (satellite + weather_z + weather_raw, 2010),
        "weather_plus_shock_persistence": (
            weather_z + weather_raw + ["shock_lag_1"], 2010),
    }

    rows = []
    for name, (columns, first_train_year) in variants.items():
        for year in TEST_YEARS:
            train = design[design.season_start_year.between(first_train_year, year - 1)]
            train = train.dropna(subset=["shock"])
            test = design[design.season_start_year.eq(year)]
            if train.season_start_year.nunique() < MIN_TRAIN_SEASONS or test.empty:
                continue
            fitted = fit_ridge(train, columns, "shock")
            if fitted is None:
                continue
            rows.append(pd.DataFrame({
                "variant": name,
                "state_name": test.state_name.values,
                "season_start_year": test.season_start_year.values,
                "actual_shock": test.shock.values,
                "predicted_shock": apply_ridge(fitted, test),
                "train_seasons": train.season_start_year.nunique(),
            }))
    # persistence, evaluated on the same rows
    persistence = design[design.season_start_year.isin(TEST_YEARS)].copy()
    rows.append(pd.DataFrame({
        "variant": "shock_persistence_only",
        "state_name": persistence.state_name.values,
        "season_start_year": persistence.season_start_year.values,
        "actual_shock": persistence.shock.values,
        "predicted_shock": persistence.shock_lag_1.values,
        "train_seasons": np.nan,
    }))
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(ARTIFACTS / "shock_model_comparison.csv", index=False)

    print("=== Predicting the shared state-season shock, 2014-2022 ===")
    print("(persistence = last season's shock for the same state)\n")
    print(f"{'shock predictor':<26}{'n':>4}{'corr':>8}{'R2':>8}"
          f"{'sd resid (log)':>16}{'~kg/ha':>9}")
    for name in variants:
        block = result[result.variant.eq(name)].dropna()
        a, p = block.actual_shock.values, block.predicted_shock.values
        r2 = 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()
        print(f"{name:<26}{len(block):>4}{np.corrcoef(a, p)[0,1]:>8.3f}"
              f"{r2:>8.3f}{(a-p).std():>16.4f}{(a-p).std()*4500:>9.0f}")
    block = result[result.variant.eq("satellite_only")].dropna()
    a = block.actual_shock.values
    print(f"{'(no model: predict 0)':<26}{len(block):>4}{0.0:>8.3f}"
          f"{1 - (a**2).sum()/((a-a.mean())**2).sum():>8.3f}"
          f"{a.std():>16.4f}{a.std()*4500:>9.0f}")

    print("\n=== Which pre-clock signals carry the shock? (pooled correlations) ===")
    pooled = design[design.season_start_year.between(2010, 2022)].dropna(subset=["shock"])
    corrs = []
    for column in weather_z + weather_raw + satellite:
        sub = pooled[["shock", column]].dropna()
        if len(sub) >= 20 and sub[column].std() > 0:
            corrs.append((column, float(np.corrcoef(sub.shock, sub[column])[0, 1]),
                          len(sub)))
    corrs.sort(key=lambda t: -abs(t[1]))
    for column, value, n in corrs[:12]:
        print(f"  {value:+.3f}  (n={n:>3})  {column}")

    pd.DataFrame(corrs, columns=["feature", "corr_with_shock", "rows"]).to_csv(
        ARTIFACTS / "shock_feature_correlations.csv", index=False)


if __name__ == "__main__":
    main()
