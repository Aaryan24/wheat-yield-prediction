#!/usr/bin/env python3
"""Assemble the V16 long panel.

The core V15 limitation is that every modelling choice rested on four test
years, because Sentinel high-resolution crop state only starts in 2017.  The
MODIS sequence panel runs 2000-2022 and the yield table runs 1990-2022, so a
much longer evaluation is already possible with data on disk.

This script builds three nested tiers:

  tier_long    2000-2022  yield history + MODIS crop sequence          (~21 folds)
  tier_weather 2010-2022  + the 78 V14 physical/weather/economic inputs (~11 folds)
  tier_recent  2017-2022  + Sentinel high-resolution crop state         (  4 folds)

Architecture and hyper-parameter choices are made on tier_long, where the
number of independent season observations is large enough to resolve them.
tier_recent is used only to confirm a locked recipe, never to choose one.

No yield label from the target season or later is ever used to build a feature.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
RAPID = V16.parent
UGP = RAPID.parent
sys.path.insert(0, str(UGP))

V15_DATA = RAPID / "v15_complete_hierarchy" / "data"
DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"

TARGET = "yield_kg_per_ha"


# --------------------------------------------------------------------------
# yield history features, computed strictly from seasons before the target
# --------------------------------------------------------------------------
def history_features(long_yield: pd.DataFrame) -> pd.DataFrame:
    """Lagged district and state yield behaviour, 1990-2022.

    Every column here is a function of seasons strictly before the target
    season, so it is legal at any forecast clock.
    """
    frame = long_yield.sort_values(["district_id", "season_start_year"]).copy()
    grouped = frame.groupby("district_id")[TARGET]

    for lag in range(1, 11):
        frame[f"lag_{lag}_yield"] = grouped.shift(lag)

    # shifted series -> every rolling statistic excludes the target season
    shifted = grouped.shift(1)
    by_district = shifted.groupby(frame["district_id"])
    for window in (3, 5, 10, 20):
        frame[f"roll_{window}_mean"] = by_district.transform(
            lambda s, w=window: s.rolling(w, min_periods=2).mean())
        frame[f"roll_{window}_std"] = by_district.transform(
            lambda s, w=window: s.rolling(w, min_periods=3).std())
    for window in (5, 10, 20):
        frame[f"roll_{window}_slope"] = by_district.transform(
            lambda s, w=window: s.rolling(w, min_periods=4).apply(_slope, raw=True))

    # V5 / V14 three-season weighted baseline: the residual target anchor
    frame["baseline_weighted_recent"] = (
        0.60 * frame["lag_1_yield"]
        + 0.25 * frame["lag_2_yield"]
        + 0.15 * frame["lag_3_yield"]
    )

    # state-level lagged behaviour: the shared seasonal shock channel
    state_lag = (
        frame.groupby(["state_name", "season_start_year"])["lag_1_yield"]
        .mean().rename("state_lag_1_mean_yield").reset_index()
    )
    frame = frame.merge(state_lag, on=["state_name", "season_start_year"], how="left")
    frame["lag_1_minus_state"] = frame["lag_1_yield"] - frame["state_lag_1_mean_yield"]

    state_prev = (
        frame.groupby(["state_name", "season_start_year"])["lag_2_yield"]
        .mean().rename("state_lag_2_mean_yield").reset_index()
    )
    frame = frame.merge(state_prev, on=["state_name", "season_start_year"], how="left")
    frame["state_lag_1_change"] = (
        frame["state_lag_1_mean_yield"] - frame["state_lag_2_mean_yield"])

    # where the district sits relative to its own long-run level
    frame["lag_1_over_roll_10"] = frame["lag_1_yield"] / frame["roll_10_mean"]
    frame["roll_3_over_roll_10"] = frame["roll_3_mean"] / frame["roll_10_mean"]
    frame["detrended_lag_1"] = frame["lag_1_yield"] - frame["roll_10_mean"]
    frame["season_index"] = frame["season_start_year"] - 2000
    return frame


def _slope(values: np.ndarray) -> float:
    ok = np.isfinite(values)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(len(values), dtype=float)[ok]
    return float(np.polyfit(x, values[ok], 1)[0])


# --------------------------------------------------------------------------
# MODIS crop-sequence features
# --------------------------------------------------------------------------
def modis_features(sequence: np.ndarray, mask: np.ndarray,
                   meta: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Flatten the 3x35 MODIS clock sequence into tabular columns.

    Token 2 (5 March) is the primary forecast clock.  The Jan->Feb and
    Feb->Mar changes are kept explicitly because seasonal crop *movement*
    carries information the level alone does not.
    """
    seq = sequence.astype(np.float64).copy()
    seq[~mask] = np.nan
    out = {"district_id": meta["district_id"].values,
           "season_start_year": meta["season_start_year"].values.astype(int)}
    clocks = ("jan15", "feb15", "mar05")
    for t, clock in enumerate(clocks):
        for j, name in enumerate(names):
            out[f"modis_{clock}__{name}"] = seq[:, t, j]
    for j, name in enumerate(names):
        out[f"modis_d_janfeb__{name}"] = seq[:, 1, j] - seq[:, 0, j]
        out[f"modis_d_febmar__{name}"] = seq[:, 2, j] - seq[:, 1, j]
    return pd.DataFrame(out)


def modis_anomalies(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Expanding-window anomalies: value minus the district's own prior mean.

    The prior mean uses only seasons strictly before the target season, so this
    is a legal feature.  Anomalies matter more than levels because absolute
    NDVI differs between districts for reasons unrelated to yield.
    """
    frame = frame.sort_values(["district_id", "season_start_year"]).copy()
    cols = [f"modis_mar05__{n}" for n in names]
    for col in cols:
        prior = (frame.groupby("district_id")[col]
                 .transform(lambda s: s.shift(1).expanding(min_periods=3).mean()))
        prior_sd = (frame.groupby("district_id")[col]
                    .transform(lambda s: s.shift(1).expanding(min_periods=4).std()))
        frame[f"anom__{col}"] = frame[col] - prior
        frame[f"z__{col}"] = (frame[col] - prior) / prior_sd.replace(0, np.nan)
    return frame


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    long_yield = pd.read_parquet(V15_DATA / "long_yield_1990_2022.parquet")
    manifest = json.loads((V15_DATA / "data_manifest.json").read_text())
    names = manifest["modis_features"]

    npz = np.load(V15_DATA / "modis_sequences_2000_2022.npz", allow_pickle=True)
    modis_meta = pd.read_parquet(V15_DATA / "modis_metadata.parquet")

    hist = history_features(long_yield)
    modis = modis_features(npz["sequence"], npz["mask"], modis_meta, names)

    panel = hist.merge(modis, on=["district_id", "season_start_year"],
                       how="left", validate="one_to_one")
    panel = modis_anomalies(panel, names)

    # tier flags
    panel["has_modis"] = panel[f"modis_mar05__{names[0]}"].notna()
    panel["tier_long"] = panel.season_start_year.between(2000, 2022) & panel.has_modis
    panel["tier_weather"] = panel.tier_long & panel.season_start_year.ge(2010)
    panel["tier_recent"] = panel.tier_long & panel.season_start_year.ge(2017)

    # feature groups
    # season_index is deliberately EXCLUDED from the history group.  Measured on
    # 19 rolling-origin folds it costs ~57 kg/ha RMSE: extrapolating a fitted
    # yield trend into an unseen season is unsafe.  It is kept as its own group
    # so the harm remains reproducible.
    hist_cols = [c for c in panel.columns if c.startswith((
        "lag_", "roll_", "state_lag", "detrended_"))
        or c in ("lag_1_minus_state", "lag_1_over_roll_10", "roll_3_over_roll_10")]
    modis_level = [c for c in panel.columns if c.startswith("modis_")]
    modis_anom = [c for c in panel.columns if c.startswith(("anom__", "z__"))]

    groups = {
        "history": sorted(hist_cols),
        "modis_level": sorted(modis_level),
        "modis_anomaly": sorted(modis_anom),
        "history_modis": sorted(set(hist_cols) | set(modis_level) | set(modis_anom)),
        "trend_unsafe": ["season_index"],
    }

    panel.to_parquet(DATA / "v16_panel.parquet", index=False)
    (DATA / "v16_feature_groups.json").write_text(json.dumps(groups, indent=1))

    usable = panel[panel.tier_long & panel[TARGET].notna()
                   & panel.baseline_weighted_recent.notna()]
    summary = {
        "panel_rows": int(len(panel)),
        "districts": int(panel.district_id.nunique()),
        "years": [int(panel.season_start_year.min()), int(panel.season_start_year.max())],
        "tier_long_rows": int(panel.tier_long.sum()),
        "tier_weather_rows": int(panel.tier_weather.sum()),
        "tier_recent_rows": int(panel.tier_recent.sum()),
        "usable_rows_with_label_and_baseline": int(len(usable)),
        "usable_years": sorted(int(y) for y in usable.season_start_year.unique()),
        "independent_year_folds_long": int(usable.season_start_year.nunique()),
        "feature_counts": {k: len(v) for k, v in groups.items()},
        "post_2022_labels_read": False,
    }
    (ARTIFACTS / "panel_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
