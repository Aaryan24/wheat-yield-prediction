#!/usr/bin/env python3
"""Do sub-district tiles add yield skill that district means cannot?

This is the first time in the V16 sequence that the model is given genuinely
new information rather than a new way of arranging the same numbers.  Three
prior attempts -- V15's crop Transformer, V16's rewritten Transformer, and the
MODIS level features -- all failed to beat plain district-mean anomalies,
because at district resolution there was nothing further to extract.

Judged on nineteen rolling-origin folds against the strongest tabular baseline
established so far (history + district-mean MODIS anomalies), with the
season-resampled bootstrap and the column-permutation noise floor reported
alongside, exactly as for every other candidate.
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
BASE = "history_plus_district_anomaly"


def main() -> None:
    panel = pd.read_parquet(DATA / "v16_panel.parquet")
    groups = json.loads((DATA / "v16_feature_groups.json").read_text())
    tiles = pd.read_parquet(DATA / "v16_tile_features.parquet")
    tile_groups = json.loads((DATA / "v16_tile_groups.json").read_text())

    panel = panel[panel.tier_long & panel[TARGET].notna()
                  & panel[BASELINE].notna()].copy()
    panel = panel.merge(tiles, on=["district_id", "season_start_year"],
                        how="left", validate="one_to_one")

    district = groups["history"] + groups["modis_anomaly"]
    sets = {
        BASE: district,
        "plus_tile_shape": district + tile_groups["tile_shape"],
        "plus_tile_anomaly": district + tile_groups["tile_anomaly"],
        "plus_tile_all": district + tile_groups["tile_all"],
        "tiles_instead_of_district_means": (groups["history"]
                                            + tile_groups["tile_all"]),
    }

    predictions = {}
    for name, features in sets.items():
        predictions[name] = rolling_origin_predict(panel, features, TEST_YEARS)
        print(f"  fitted {name}: {len(features)} features", flush=True)

    rows = []
    for name, pred in predictions.items():
        for label, subset in (("all_19_folds", pred),
                              ("v15_window_2019_22",
                               pred[pred.season_start_year.isin(LATE)])):
            rows.append({"feature_set": name, "period": label, **metrics(subset)})
    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "tile_eval_metrics.csv", index=False)
    print("\n=== Tile evaluation (kg/ha) ===")
    print(report[["feature_set", "period", "rows", "years", "rmse", "mae",
                  "bias", "direction_accuracy"]].to_string(index=False))

    merged = predictions[BASE][["district_id", "state_name", "season_start_year",
                                TARGET, "lag_1_yield"]].copy()
    for name, pred in predictions.items():
        merged = merged.merge(
            pred[["district_id", "season_start_year", "prediction"]]
            .rename(columns={"prediction": name}),
            on=["district_id", "season_start_year"], validate="one_to_one")

    print("\n=== Do tiles beat district means? (season-resampled bootstrap) ===")
    boot = []
    for name in list(sets)[1:]:
        for label, subset in (("all_19_folds", merged),
                              ("v15_window_2019_22",
                               merged[merged.season_start_year.isin(LATE)])):
            b = year_block_bootstrap(subset, name, BASE)
            boot.append({"candidate": name, "baseline": BASE,
                         "period": label, **b})
            print(f"  {name:<32}{label:<20} gain {b['mean_gain']:+7.2f} "
                  f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
                  f"P(>0)={b['probability_positive']:.3f}")
    pd.DataFrame(boot).to_csv(ARTIFACTS / "tile_eval_bootstrap.csv", index=False)

    print("\n=== Noise floor ===")
    floors = []
    for name in (BASE, "plus_tile_anomaly"):
        floor = noise_floor(panel, sets[name], TEST_YEARS, draws=6)
        floors.append({"feature_set": name, **floor})
        print(f"  {name:<32} permuted {floor['permuted_mean']:7.2f} "
              f"+- {floor['permuted_sd']:.2f}")
    pd.DataFrame(floors).to_csv(ARTIFACTS / "tile_eval_noise_floor.csv", index=False)

    per_year = pd.DataFrame([{
        "season_start_year": int(year),
        "base_rmse": metrics(block, BASE)["rmse"],
        "tile_anomaly_rmse": metrics(block, "plus_tile_anomaly")["rmse"],
    } for year, block in merged.groupby("season_start_year")])
    per_year["gain"] = per_year.base_rmse - per_year.tile_anomaly_rmse
    per_year.to_csv(ARTIFACTS / "tile_eval_per_year.csv", index=False)
    print("\n=== Per season ===")
    print(per_year.to_string(index=False))

    merged.to_parquet(ARTIFACTS / "tile_eval_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
