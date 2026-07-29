#!/usr/bin/env python3
"""Does the V16 encoder add yield skill that plain MODIS anomalies do not?

This is the test V15 could not run.  V15 judged its encoder on four test
seasons, where the column-permutation noise floor (sd 0.57 kg/ha) was the same
size as the claimed gain (0.93 kg/ha).  Here the same question is asked on
nineteen rolling-origin folds, with:

  * a matched-model design -- identical rows, target, tree settings and seeds,
    differing only by the presence of the encoder columns, as V15 did;
  * a season-resampled bootstrap, because districts fail together in a season;
  * the column-permutation noise floor reported next to every gain.

The comparison that matters is against `history + MODIS anomalies`, NOT against
history alone.  Beating history alone would only re-prove that satellite data
helps, which was already established at +17.8 kg/ha.  The encoder has to earn
its place over the tabular anomalies it is built from.
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


def main() -> None:
    panel = pd.read_parquet(DATA / "v16_panel.parquet")
    groups = json.loads((DATA / "v16_feature_groups.json").read_text())
    encoder = pd.read_parquet(DATA / "v16_encoder_features.parquet")
    panel = panel[panel.tier_long & panel[TARGET].notna()
                  & panel[BASELINE].notna()].copy()

    encoder_columns = [c for c in encoder.columns if c.startswith("enc__")]
    sets = {
        "history_plus_modis_anomaly": groups["history"] + groups["modis_anomaly"],
        "plus_v16_encoder": (groups["history"] + groups["modis_anomaly"]
                             + encoder_columns),
        "encoder_without_tabular_anomaly": groups["history"] + encoder_columns,
    }

    predictions = {}
    for name, features in sets.items():
        blocks = []
        for year in TEST_YEARS:
            # each fold uses the representation built without that season, and
            # training-row features come from district-cross-fitted encoders
            fold_encoder = encoder[encoder.representation_train_end.eq(year - 1)]
            merged = panel.merge(
                fold_encoder.drop(columns=["representation_train_end",
                                           "feature_role"]),
                on=["district_id", "season_start_year"], how="left")
            if merged.duplicated(["district_id", "season_start_year"]).any():
                raise RuntimeError("encoder rows are not unique within a fold")
            block = rolling_origin_predict(merged, features, [year])
            if len(block):
                blocks.append(block)
        predictions[name] = pd.concat(blocks, ignore_index=True)
        print(f"  fitted {name}: {len(predictions[name])} rows", flush=True)

    rows = []
    for name, pred in predictions.items():
        for label, subset in (("all_19_folds", pred),
                              ("v15_window_2019_22",
                               pred[pred.season_start_year.isin(LATE)])):
            rows.append({"feature_set": name, "period": label, **metrics(subset)})
    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "encoder_eval_metrics.csv", index=False)
    print("\n=== Encoder evaluation (kg/ha) ===")
    print(report[["feature_set", "period", "rows", "years", "rmse", "mae",
                  "bias", "direction_accuracy"]].to_string(index=False))

    merged = predictions["history_plus_modis_anomaly"][[
        "district_id", "season_start_year", TARGET, "lag_1_yield"]].copy()
    for name, pred in predictions.items():
        merged = merged.merge(
            pred[["district_id", "season_start_year", "prediction"]]
            .rename(columns={"prediction": name}),
            on=["district_id", "season_start_year"], validate="one_to_one")

    print("\n=== Does the encoder beat the tabular anomalies it is built from? ===")
    boot = []
    for label, subset in (("all_19_folds", merged),
                          ("v15_window_2019_22",
                           merged[merged.season_start_year.isin(LATE)])):
        b = year_block_bootstrap(subset, "plus_v16_encoder",
                                 "history_plus_modis_anomaly")
        boot.append({"candidate": "plus_v16_encoder",
                     "baseline": "history_plus_modis_anomaly",
                     "period": label, **b})
        print(f"  {label:<20} gain {b['mean_gain']:+7.2f} "
              f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
              f"P(>0)={b['probability_positive']:.3f}")
    pd.DataFrame(boot).to_csv(ARTIFACTS / "encoder_eval_bootstrap.csv", index=False)

    print("\n=== Noise floor for the same comparison ===")
    fold_encoder = encoder[encoder.representation_train_end.eq(2021)]
    merged_panel = panel.merge(
        fold_encoder.drop(columns=["representation_train_end", "feature_role"]),
        on=["district_id", "season_start_year"], how="left")
    floors = []
    for name in ("history_plus_modis_anomaly", "plus_v16_encoder"):
        floor = noise_floor(merged_panel, sets[name], TEST_YEARS, draws=6)
        floors.append({"feature_set": name, **floor})
        print(f"  {name:<32} permuted {floor['permuted_mean']:7.2f} "
              f"+- {floor['permuted_sd']:.2f}")
    pd.DataFrame(floors).to_csv(ARTIFACTS / "encoder_eval_noise_floor.csv",
                                index=False)
    merged.to_parquet(ARTIFACTS / "encoder_eval_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
