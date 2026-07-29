#!/usr/bin/env python3
"""Is V15's training window, not its architecture, the thing holding it back?

V15 trains its XGBoost models on seasons 2017..train_end.  For the 2019 fold
that is two seasons, about 238 rows; for the last fold it is four seasons.
The window starts at 2017 because Sentinel crop state starts there -- but the
78 physical/weather/economic features that do the actual work run from 2010.

So V15 may be discarding two thirds of the labelled data it already has, purely
because one late-arriving modality is joined to the same table.

This tests it directly: the identical model, features, target, depth and seeds,
changed in exactly one respect -- the first training season.  It also re-tests
the sub-district tile features under the longer window, since a 132-column
feature block cannot possibly be fitted on 238 rows and their failure to
transfer may be a sample-size artifact rather than a statement about tiles.
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
sys.path.insert(0, str(V16 / "scripts"))
from rapid_yield_forecast.v14_anomaly_distribution.scripts import (  # noqa: E402
    run_v14_lab as lab)
from v16_common import year_block_bootstrap  # noqa: E402

DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"
V15A = RAPID / "v15_complete_hierarchy" / "artifacts"
TARGET = "yield_kg_per_ha"
YEARS = (2019, 2020, 2021, 2022)
DEV = (2019, 2020)
LATE = (2021, 2022)
FOLD_END = {2019: 2018, 2020: 2019, 2021: 2020, 2022: 2020}
# The 78 physical features are only fully populated from 2015 (2014: 77 of 78,
# 2013: 72, and before that fewer than 30).  The three-season baseline is also
# undefined before 2013.  So the honest extension is 2013-2016, not 2010.
STARTS = (2017, 2016, 2015, 2014, 2013)


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def main() -> None:
    base_panel, groups, _ = lab.load_panel()
    tiles = pd.read_parquet(DATA / "v16_tile_features.parquet")
    tile_groups = json.loads((DATA / "v16_tile_groups.json").read_text())
    panel = base_panel.merge(tiles, on=["district_id", "season_start_year"],
                             how="left")
    # rows without a defined residual target cannot train anything
    panel = panel[panel[TARGET].notna()
                  & panel["baseline_weighted_recent"].notna()].copy()

    v15 = pd.read_parquet(V15A / "final_predictions.parquet")[[
        "district_id", "season_start_year", TARGET, "lag_1_yield", "state_name",
        "shadow_point_prediction", "v15_point_prediction"]]

    physical = groups["physical"]
    variants = {"physical_only": physical,
                "physical_plus_tiles": physical + tile_groups["tile_all"]}

    blocks, audit = [], []
    for year in YEARS:
        train_end = FOLD_END[year]
        test = panel[panel.season_start_year.eq(year)].copy()
        block = test[["district_id", "season_start_year"]].copy()
        for start in STARTS:
            train = panel[panel.season_start_year.between(start, train_end)].copy()
            for name, features in variants.items():
                block[f"{name}__from{start}"] = lab.xgb_residual_predict(
                    train, test, features, 2)
            audit.append({"test_year": year, "train_start": start,
                          "train_seasons": int(train.season_start_year.nunique()),
                          "train_rows": int(len(train))})
        blocks.append(block)
        print(f"  fold {year}: trained from {STARTS}", flush=True)

    predictions = pd.concat(blocks, ignore_index=True)
    frame = v15.merge(predictions, on=["district_id", "season_start_year"],
                      validate="one_to_one")
    pd.DataFrame(audit).to_csv(ARTIFACTS / "v15_window_audit.csv", index=False)

    truth = frame[TARGET].to_numpy(float)
    dev = frame.season_start_year.isin(DEV).to_numpy()
    late = frame.season_start_year.isin(LATE).to_numpy()

    print("\n=== The XGBoost residual model alone, by first training season ===")
    print("(this is the V15 component, not the full V5+V14 stack)")
    rows = []
    for name in variants:
        for start in STARTS:
            column = f"{name}__from{start}"
            values = frame[column].to_numpy(float)
            seasons = [a["train_seasons"] for a in audit
                       if a["train_start"] == start and a["test_year"] == 2022][0]
            rows.append({"features": name, "train_start": start,
                         "train_seasons_last_fold": seasons,
                         "dev_rmse": rmse(values[dev], truth[dev]),
                         "late_rmse": rmse(values[late], truth[late]),
                         "four_year_rmse": rmse(values, truth)})
    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "v15_window_metrics.csv", index=False)
    print(report.to_string(index=False))

    anchor = frame.shadow_point_prediction.to_numpy(float)
    frame["anchor"] = anchor
    print("\n=== As a correction on the V14 anchor "
          "(weight chosen on 2019-20 only) ===")
    grid = [round(0.25 * i, 2) for i in range(0, 13)]
    rows = []
    for name in variants:
        for start in STARTS:
            if start == 2017 and name == "physical_only":
                continue
            correction = (frame[f"{name}__from{start}"].to_numpy(float)
                          - frame["physical_only__from2017"].to_numpy(float))
            best_gamma, best = 0.0, np.inf
            for gamma in grid:
                value = rmse((anchor + gamma * correction)[dev], truth[dev])
                if value < best:
                    best_gamma, best = gamma, value
            point = anchor + best_gamma * correction
            frame[f"port__{name}__{start}"] = point
            rows.append({"candidate": f"{name} from {start}",
                         "gamma": best_gamma,
                         "dev_rmse": rmse(point[dev], truth[dev]),
                         "late_rmse": rmse(point[late], truth[late]),
                         "four_year_rmse": rmse(point, truth),
                         "corr_with_error": float(np.corrcoef(
                             correction, truth - anchor)[0, 1])})
    port = pd.DataFrame([{
        "candidate": "V14 anchor", "gamma": 0.0,
        "dev_rmse": rmse(anchor[dev], truth[dev]),
        "late_rmse": rmse(anchor[late], truth[late]),
        "four_year_rmse": rmse(anchor, truth), "corr_with_error": np.nan},
        {"candidate": "V15 released", "gamma": 1.25,
         "dev_rmse": rmse(frame.v15_point_prediction.to_numpy(float)[dev],
                          truth[dev]),
         "late_rmse": rmse(frame.v15_point_prediction.to_numpy(float)[late],
                           truth[late]),
         "four_year_rmse": rmse(frame.v15_point_prediction.to_numpy(float), truth),
         "corr_with_error": np.nan}] + rows)
    port.to_csv(ARTIFACTS / "v15_window_port_metrics.csv", index=False)
    print(port.to_string(index=False))

    print("\n=== Season-resampled bootstrap against the V14 anchor ===")
    boot = []
    for column, label in [("v15_point_prediction", "V15 released")] + [
            (f"port__{n}__{s}", f"{n} from {s}")
            for n in variants for s in STARTS
            if not (s == 2017 and n == "physical_only")]:
        for period, subset in (("2019-2022", frame),
                               ("untouched 2021-2022", frame[late])):
            b = year_block_bootstrap(subset, column, "anchor")
            boot.append({"candidate": label, "period": period, **b})
            print(f"  {label:<28}{period:<22} gain {b['mean_gain']:+7.2f} "
                  f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
                  f"P(>0)={b['probability_positive']:.3f}")
    pd.DataFrame(boot).to_csv(ARTIFACTS / "v15_window_bootstrap.csv", index=False)
    frame.to_parquet(ARTIFACTS / "v15_window_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
