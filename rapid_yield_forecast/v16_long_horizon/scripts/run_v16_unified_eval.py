#!/usr/bin/env python3
"""Does the unified encoder beat the hand-built tile statistics it replaces?

The encoder compresses ~1,200 raw numbers per district-season (96 tiles x 12,
3 MODIS clocks x 35, 4 weather windows x 10, 9 climate terms) into 40, using
learned attention pooling instead of fixed percentiles.

The question is whether a learned squeeze beats a hand-picked one.  The honest
comparison is against the tabular tile features, NOT against history alone --
beating history alone would only re-prove that satellite data helps.

Nineteen rolling-origin folds, season-resampled bootstrap, and the
column-permutation noise floor, exactly as for every other candidate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V16 / "scripts"))
from v16_common import (  # noqa: E402
    BASELINE, TARGET, metrics, noise_floor, rolling_origin_predict,
    year_block_bootstrap,
)

DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"
TEST_YEARS = list(range(2004, 2023))
LATE = [2019, 2020, 2021, 2022]
BASE = "history_plus_tiles"


def main() -> None:
    panel = pd.read_parquet(DATA / "v16_panel.parquet")
    groups = json.loads((DATA / "v16_feature_groups.json").read_text())
    tiles = pd.read_parquet(DATA / "v16_tile_features.parquet")
    tile_groups = json.loads((DATA / "v16_tile_groups.json").read_text())
    unified = pd.read_parquet(DATA / "v16_unified_features.parquet")
    unified_columns = [c for c in unified.columns if c.startswith("uni__")]

    panel = panel[panel.tier_long & panel[TARGET].notna()
                  & panel[BASELINE].notna()].copy()
    panel = panel.merge(tiles, on=["district_id", "season_start_year"],
                        how="left", validate="one_to_one")

    district = groups["history"] + groups["modis_anomaly"]
    sets = {
        "history_plus_district_means": district,
        BASE: district + tile_groups["tile_all"],
        "history_plus_unified_encoder": district + unified_columns,
        "tiles_plus_unified_encoder": (district + tile_groups["tile_all"]
                                       + unified_columns),
        "unified_encoder_alone": groups["history"] + unified_columns,
    }

    predictions = {}
    for name, features in sets.items():
        blocks = []
        for year in TEST_YEARS:
            fold = unified[unified.representation_train_end.eq(year - 1)]
            merged = panel.merge(
                fold.drop(columns=["representation_train_end"]),
                on=["district_id", "season_start_year"], how="left")
            if merged.duplicated(["district_id", "season_start_year"]).any():
                raise RuntimeError("encoder rows not unique within fold")
            block = rolling_origin_predict(merged, features, [year])
            if len(block):
                blocks.append(block)
        predictions[name] = pd.concat(blocks, ignore_index=True)
        print(f"  fitted {name}: {len(features)} features", flush=True)

    rows = []
    for name, pred in predictions.items():
        for label, subset in (("all_19_folds", pred),
                              ("v15_window_2019_22",
                               pred[pred.season_start_year.isin(LATE)])):
            rows.append({"feature_set": name, "period": label, **metrics(subset)})
    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "unified_eval_metrics.csv", index=False)
    print("\n=== Unified encoder evaluation (kg/ha) ===")
    print(report[["feature_set", "period", "rows", "rmse", "mae", "bias",
                  "direction_accuracy"]].to_string(index=False))

    merged = predictions[BASE][["district_id", "state_name", "season_start_year",
                                TARGET, "lag_1_yield"]].copy()
    for name, pred in predictions.items():
        merged = merged.merge(
            pred[["district_id", "season_start_year", "prediction"]]
            .rename(columns={"prediction": name}),
            on=["district_id", "season_start_year"], validate="one_to_one")

    print("\n=== Against the hand-built tile statistics ===")
    boot = []
    for name in sets:
        if name == BASE:
            continue
        for label, subset in (("all_19_folds", merged),
                              ("v15_window_2019_22",
                               merged[merged.season_start_year.isin(LATE)])):
            b = year_block_bootstrap(subset, name, BASE)
            boot.append({"candidate": name, "baseline": BASE,
                         "period": label, **b})
            print(f"  {name:<32}{label:<20} gain {b['mean_gain']:+7.2f} "
                  f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
                  f"P(>0)={b['probability_positive']:.3f}")
    pd.DataFrame(boot).to_csv(ARTIFACTS / "unified_eval_bootstrap.csv", index=False)

    print("\n=== Noise floor ===")
    fold = unified[unified.representation_train_end.eq(2021)]
    merged_panel = panel.merge(fold.drop(columns=["representation_train_end"]),
                               on=["district_id", "season_start_year"], how="left")
    floors = []
    for name in (BASE, "tiles_plus_unified_encoder"):
        floor = noise_floor(merged_panel, sets[name], TEST_YEARS, draws=6)
        floors.append({"feature_set": name, **floor})
        print(f"  {name:<34} permuted {floor['permuted_mean']:7.2f} "
              f"+- {floor['permuted_sd']:.2f}")
    pd.DataFrame(floors).to_csv(ARTIFACTS / "unified_eval_noise_floor.csv",
                                index=False)
    merged.to_parquet(ARTIFACTS / "unified_eval_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
