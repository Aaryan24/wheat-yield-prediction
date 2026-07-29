#!/usr/bin/env python3
"""Can any V16 component actually improve the released V15 model?

Everything V16 measured so far was on a deliberately thin feature set -- yield
history plus satellite -- because that is what reaches back to 2000 and gives
nineteen test seasons.  V15 is far stronger on 2019-2022 (269.5 vs ~331 kg/ha)
because it also carries 78 weather/soil/economic inputs, the V5 ensemble and a
transfer model.  So a V16 gain measured on the long panel does not
automatically transfer.

This script tests the transfer directly, using V15's own methodology so the
comparison is like for like:

  * the same 78-column physical panel, the same depth-2 two-seed XGBoost, the
    same residual target around the three-season baseline;
  * a MATCHED-MODEL DIFFERENCE -- identical rows, target, settings and seeds,
    differing only by the extra columns -- so the measured effect is the
    information the new features add and nothing else;
  * the correction weight chosen on 2019-2020 only, then applied unchanged to
    2021-2022, which V15 treated as the untouched confirmation block;
  * the column-permutation noise floor reported alongside, because V15's own
    released gain (0.93 kg/ha) sat inside it.

Candidates: the tabular tile features that worked on the long panel, the
unified encoder representation, and both together.
"""
from __future__ import annotations

import argparse
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
GAMMA_GRID = [round(0.25 * i, 2) for i in range(0, 13)]


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-unified", action="store_true",
                        help="run before the unified encoder has finished training")
    arguments = parser.parse_args()

    base_panel, groups, _ = lab.load_panel()
    tiles = pd.read_parquet(DATA / "v16_tile_features.parquet")
    tile_groups = json.loads((DATA / "v16_tile_groups.json").read_text())
    if arguments.skip_unified:
        unified = pd.DataFrame(columns=["district_id", "season_start_year",
                                        "representation_train_end"])
        unified_columns = []
    else:
        unified = pd.read_parquet(DATA / "v16_unified_features.parquet")
        unified_columns = [c for c in unified.columns if c.startswith("uni__")]

    v15 = pd.read_parquet(V15A / "final_predictions.parquet")[[
        "district_id", "season_start_year", "yield_kg_per_ha", "lag_1_yield",
        "state_name", "production_point_prediction", "shadow_point_prediction",
        "v15_point_prediction"]]

    candidates = {"tile_shape": tile_groups["tile_shape"],
                  "tile_anomaly": tile_groups["tile_anomaly"],
                  "tile_all": tile_groups["tile_all"]}
    if unified_columns:
        candidates["unified_encoder"] = unified_columns
        candidates["tiles_plus_encoder"] = tile_groups["tile_all"] + unified_columns

    blocks = []
    for year in YEARS:
        train_end = FOLD_END[year]
        fold_unified = unified[unified.representation_train_end.eq(train_end)]
        merged = base_panel.merge(tiles, on=["district_id", "season_start_year"],
                                  how="left")
        if unified_columns:
            merged = merged.merge(
                fold_unified.drop(columns=["representation_train_end"]),
                on=["district_id", "season_start_year"], how="left")
        if merged.duplicated(["district_id", "season_start_year"]).any():
            raise RuntimeError("duplicate rows after merge")

        train = merged[merged.season_start_year.between(2017, train_end)].copy()
        test = merged[merged.season_start_year.eq(year)].copy()
        block = test[["district_id", "season_start_year"]].copy()

        physical = groups["physical"]
        block["base"] = lab.xgb_residual_predict(train, test, physical, 2)
        for name, extra in candidates.items():
            usable = [c for c in extra if c in merged.columns]
            block[name] = lab.xgb_residual_predict(
                train, test, physical + usable, 2)
        blocks.append(block)
        print(f"  fold {year}: {len(test)} rows", flush=True)

    predictions = pd.concat(blocks, ignore_index=True)
    frame = v15.merge(predictions, on=["district_id", "season_start_year"],
                      validate="one_to_one")
    dev = frame.season_start_year.isin(DEV)
    late = frame.season_start_year.isin(LATE)
    truth = frame[TARGET].to_numpy(float)
    anchor = frame.shadow_point_prediction.to_numpy(float)

    print("\n=== Correction weight chosen on 2019-2020, applied to 2021-2022 ===")
    rows = []
    for name in candidates:
        correction = (frame[name] - frame["base"]).to_numpy(float)
        best_gamma, best_score = 0.0, np.inf
        for gamma in GAMMA_GRID:
            value = rmse((anchor + gamma * correction)[dev.to_numpy()],
                         truth[dev.to_numpy()])
            if value < best_score:
                best_gamma, best_score = gamma, value
        point = anchor + best_gamma * correction
        frame[f"port_{name}"] = point
        rows.append({
            "candidate": name, "selected_gamma": best_gamma,
            "dev_rmse": rmse(point[dev.to_numpy()], truth[dev.to_numpy()]),
            "late_rmse": rmse(point[late.to_numpy()], truth[late.to_numpy()]),
            "four_year_rmse": rmse(point, truth),
            "correction_sd": float(correction.std()),
            "corr_with_error": float(np.corrcoef(correction, truth - anchor)[0, 1]),
        })
    report = pd.DataFrame(rows)

    reference = pd.DataFrame([{
        "candidate": "V14 anchor (no correction)", "selected_gamma": 0.0,
        "dev_rmse": rmse(anchor[dev.to_numpy()], truth[dev.to_numpy()]),
        "late_rmse": rmse(anchor[late.to_numpy()], truth[late.to_numpy()]),
        "four_year_rmse": rmse(anchor, truth),
        "correction_sd": 0.0, "corr_with_error": np.nan},
        {"candidate": "V15 released crop correction", "selected_gamma": 1.25,
         "dev_rmse": rmse(frame.v15_point_prediction.to_numpy(float)[dev.to_numpy()],
                          truth[dev.to_numpy()]),
         "late_rmse": rmse(frame.v15_point_prediction.to_numpy(float)[late.to_numpy()],
                           truth[late.to_numpy()]),
         "four_year_rmse": rmse(frame.v15_point_prediction.to_numpy(float), truth),
         "correction_sd": float((frame.v15_point_prediction
                                 - frame.shadow_point_prediction).std()),
         "corr_with_error": float(np.corrcoef(
             frame.v15_point_prediction - frame.shadow_point_prediction,
             truth - anchor)[0, 1])}])
    report = pd.concat([reference, report], ignore_index=True)
    report.to_csv(ARTIFACTS / "v15_port_metrics.csv", index=False)
    print(report.to_string(index=False))

    print("\n=== Season-resampled bootstrap against the V14 anchor ===")
    boot = []
    frame["anchor"] = anchor
    for name in list(candidates) + ["v15_released"]:
        column = ("v15_point_prediction" if name == "v15_released"
                  else f"port_{name}")
        for label, subset in (("2019-2022", frame),
                              ("untouched 2021-2022", frame[late])):
            b = year_block_bootstrap(subset, column, "anchor")
            boot.append({"candidate": name, "period": label, **b})
            print(f"  {name:<20}{label:<22} gain {b['mean_gain']:+7.2f} "
                  f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
                  f"P(>0)={b['probability_positive']:.3f}")
    pd.DataFrame(boot).to_csv(ARTIFACTS / "v15_port_bootstrap.csv", index=False)

    frame.to_parquet(ARTIFACTS / "v15_port_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
