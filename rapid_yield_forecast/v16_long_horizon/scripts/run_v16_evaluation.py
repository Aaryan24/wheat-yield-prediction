#!/usr/bin/env python3
"""Long-horizon evaluation: does satellite crop state actually add yield skill?

V15 answered this on four test years, where the honest confidence interval on
the gain included zero and the measurement noise from column reordering was the
same size as the gain itself.  Here the same question is asked on nineteen
rolling-origin folds, with the noise floor measured alongside every number and
significance judged by resampling whole seasons rather than district-seasons.
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
    TARGET, BASELINE, metrics, noise_floor, rolling_origin_predict,
    year_block_bootstrap,
)

DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"
TEST_YEARS = list(range(2004, 2023))          # 19 rolling-origin folds
LATE = [2019, 2020, 2021, 2022]                # the V15 comparison window


def main() -> None:
    panel = pd.read_parquet(DATA / "v16_panel.parquet")
    groups = json.loads((DATA / "v16_feature_groups.json").read_text())
    panel = panel[panel.tier_long & panel[TARGET].notna()
                  & panel[BASELINE].notna()].copy()

    sets = {
        "history_only": groups["history"],
        "history_plus_modis_level": groups["history"] + groups["modis_level"],
        "history_plus_modis_anomaly": groups["history"] + groups["modis_anomaly"],
        "history_plus_modis_all": groups["history_modis"],
    }

    predictions = {}
    rows = []
    for name, features in sets.items():
        pred = rolling_origin_predict(panel, features, TEST_YEARS)
        predictions[name] = pred
        for label, subset in (("all_19_folds", pred),
                              ("v15_window_2019_22",
                               pred[pred.season_start_year.isin(LATE)])):
            rows.append({"feature_set": name, "period": label,
                         **metrics(subset)})
        print(f"  fitted {name}: {len(pred)} rows, "
              f"{pred.season_start_year.nunique()} folds", flush=True)

    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "long_horizon_metrics.csv", index=False)

    # naive baseline: the weighted three-season history alone
    base = predictions["history_only"].copy()
    base["prediction"] = base[BASELINE]
    naive = {"feature_set": "weighted_history_baseline", "period": "all_19_folds",
             **metrics(base)}
    report = pd.concat([report, pd.DataFrame([naive])], ignore_index=True)

    print("\n=== Long-horizon rolling-origin results (kg/ha) ===")
    print(report[["feature_set", "period", "rows", "years", "rmse", "mae",
                  "bias", "equal_state_rmse", "direction_accuracy"]]
          .to_string(index=False))

    # merge candidates onto one frame for paired significance testing
    merged = predictions["history_only"][[
        "district_id", "season_start_year", TARGET, "lag_1_yield"]].copy()
    for name, pred in predictions.items():
        merged = merged.merge(
            pred[["district_id", "season_start_year", "prediction"]]
            .rename(columns={"prediction": name}),
            on=["district_id", "season_start_year"], validate="one_to_one")

    print("\n=== Does MODIS beat history alone? (season-resampled bootstrap) ===")
    boot_rows = []
    for name in list(sets)[1:]:
        for label, subset in (("all_19_folds", merged),
                              ("v15_window_2019_22",
                               merged[merged.season_start_year.isin(LATE)])):
            b = year_block_bootstrap(subset, name, "history_only")
            boot_rows.append({"candidate": name, "baseline": "history_only",
                              "period": label, **b})
            print(f"  {name:<30} {label:<20} gain {b['mean_gain']:+7.2f} "
                  f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
                  f"P(>0)={b['probability_positive']:.3f}")
    pd.DataFrame(boot_rows).to_csv(ARTIFACTS / "long_horizon_bootstrap.csv", index=False)

    print("\n=== Noise floor (pure column reordering, no information change) ===")
    floor_rows = []
    for name in ("history_only", "history_plus_modis_all"):
        floor = noise_floor(panel, sets[name], TEST_YEARS, draws=8)
        floor_rows.append({"feature_set": name, **floor})
        print(f"  {name:<26} rmse {floor['as_ordered_rmse']:7.2f}  "
              f"permuted {floor['permuted_mean']:7.2f} +- {floor['permuted_sd']:.2f} "
              f"[{floor['permuted_min']:.2f}, {floor['permuted_max']:.2f}]")
    pd.DataFrame(floor_rows).to_csv(ARTIFACTS / "long_horizon_noise_floor.csv",
                                    index=False)

    merged.to_parquet(ARTIFACTS / "long_horizon_predictions.parquet", index=False)

    per_year = (predictions["history_plus_modis_all"]
                .assign(err=lambda d: d.prediction - d[TARGET])
                .groupby("season_start_year")
                .agg(rows=("err", "size"),
                     rmse=("err", lambda s: float(np.sqrt(np.mean(s ** 2)))),
                     bias=("err", "mean")).reset_index())
    per_year.to_csv(ARTIFACTS / "long_horizon_per_year.csv", index=False)
    print("\n=== Per-season performance, history+MODIS ===")
    print(per_year.to_string(index=False))


if __name__ == "__main__":
    main()
