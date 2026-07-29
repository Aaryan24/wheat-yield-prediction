#!/usr/bin/env python3
"""How much of V15's machinery is actually carrying its accuracy?

V15 is a seven-stage stack: a five-member V5 ensemble with a disagreement gate
and a movement calibration, a four-model V14 future-outlook correction, a
crop Transformer, two matched XGBoost models, and a shifted empirical
distribution.  Each stage was added because it helped on some evaluation.

This script strips it back and measures what each layer is worth on the strict
2019-2022 artifacts, so the parts that cost complexity without buying accuracy
can be removed.  Nothing is refitted -- the released per-stage predictions are
recombined, so these are the honest strict numbers.

Every gain is also quoted against a season-resampled bootstrap, because with
four test seasons the usual district-level confidence is badly overstated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
RAPID = V16.parent
sys.path.insert(0, str(V16 / "scripts"))
from v16_common import year_block_bootstrap  # noqa: E402

V15A = RAPID / "v15_complete_hierarchy" / "artifacts"
V5A = RAPID / "v5" / "root_cybench_lab" / "artifacts" / "v5_integration"
ARTIFACTS = V16 / "artifacts"
TARGET = "yield_kg_per_ha"
MOVEMENT_SCALE = 1.5001443110


def score(frame: pd.DataFrame, column: str) -> dict[str, float]:
    error = frame[column].to_numpy(float) - frame[TARGET].to_numpy(float)
    rose = frame[TARGET].to_numpy(float) > frame.lag_1_yield.to_numpy(float)
    called = frame[column].to_numpy(float) > frame.lag_1_yield.to_numpy(float)
    per_state = (frame.assign(_e=error ** 2).groupby("state_name")["_e"]
                 .mean().pow(0.5))
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "equal_state_rmse": float(per_state.mean()),
            "direction_accuracy": float(np.mean(rose == called))}


def main() -> None:
    v15 = pd.read_parquet(V15A / "final_predictions.parquet")
    v5 = pd.read_csv(V5A / "predictions.csv")[[
        "district_id", "season_start_year", "weighted_history", "gate",
        "v5_prediction", "prediction_cybench_lock"]]
    frame = v15.merge(v5, on=["district_id", "season_start_year"],
                      validate="one_to_one")

    lag = frame.lag_1_yield.to_numpy(float)
    v5_point = frame.production_point_prediction.to_numpy(float)

    # back out the raw V5 ensemble from the locked movement calibration:
    #   v5 = L + s * (V_raw - L)
    raw_ensemble = lag + (v5_point - lag) / MOVEMENT_SCALE

    ladder = {
        "1. last season only": lag,
        "2. weighted 3-season history": frame.weighted_history.to_numpy(float),
        "3. V5 ensemble, no movement calibration": raw_ensemble,
        "4. V5 as released (calibration 1.5001443110)": v5_point,
        "4b. V5 with the scale rounded to 1.50": lag + 1.50 * (raw_ensemble - lag),
        "4c. V5 with the scale rounded to 1.25": lag + 1.25 * (raw_ensemble - lag),
        "5. V5 + V14 future correction": frame.shadow_point_prediction.to_numpy(float),
        "6. V15 as released (+ crop correction)":
            frame.v15_point_prediction.to_numpy(float),
    }
    rows = []
    for label, values in ladder.items():
        frame["_p"] = np.clip(values, 500, 7000)
        rows.append({"model": label, **score(frame, "_p")})
    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "v15_simplification_ladder.csv", index=False)

    print("=== What each layer of V15 is worth, strict 2019-2022 ===")
    print(report.to_string(index=False))

    print("\n=== Marginal value of each layer (kg/ha RMSE improvement) ===")
    values = report.set_index("model")["rmse"]
    order = list(ladder)
    for previous, current in zip(order, order[1:]):
        if current.startswith("4b") or current.startswith("4c"):
            continue
        print(f"  {current:<46} {values[previous] - values[current]:+7.2f}")

    print("\n=== The disagreement gate ===")
    gate = frame["gate"].astype(bool)
    print(f"  fires on {gate.sum()} of {len(frame)} rows ({gate.mean():.1%})")
    if gate.any():
        for label, subset in (("gate fired", frame[gate]),
                              ("gate did not fire", frame[~gate])):
            s = score(subset, "production_point_prediction")
            print(f"  {label:<20} n={len(subset):>4} "
                  f"V5 rmse {s['rmse']:7.2f}  bias {s['bias']:+7.2f}")

    print("\n=== Season-resampled significance of each added layer ===")
    boot = []
    pairs = [
        ("weighted_history", "V5 ensemble beats plain history?",
         "production_point_prediction"),
        ("production_point_prediction", "V14 beats V5?",
         "shadow_point_prediction"),
        ("shadow_point_prediction", "V15 crop correction beats V14?",
         "v15_point_prediction"),
    ]
    for baseline, question, candidate in pairs:
        b = year_block_bootstrap(frame, candidate, baseline)
        boot.append({"question": question, **b})
        print(f"  {question:<34} gain {b['mean_gain']:+7.2f} "
              f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
              f"P(>0)={b['probability_positive']:.3f}")
    pd.DataFrame(boot).to_csv(ARTIFACTS / "v15_simplification_bootstrap.csv",
                              index=False)

    print("\n=== Distribution: is the 0.95 width scale doing anything? ===")
    quantiles = [round(0.05 * i, 2) for i in range(1, 20)]
    columns = [f"q{int(round(a * 100)):02d}" for a in quantiles]
    y = frame[TARGET].to_numpy(float)
    centre = frame.v15_point_prediction.to_numpy(float)
    for scale in (0.90, 0.95, 1.00, 1.05):
        shifted = {c: centre + (scale / 0.95) * (frame[c].to_numpy(float) - centre)
                   for c in columns}
        losses = [float(np.mean(np.maximum(a * (y - shifted[c]),
                                           (a - 1) * (y - shifted[c]))))
                  for a, c in zip(quantiles, columns)]
        cover80 = float(np.mean((y >= shifted["q10"]) & (y <= shifted["q90"])))
        marker = "  <- released" if abs(scale - 0.95) < 1e-9 else ""
        print(f"  scale {scale:.2f}: pinball {np.mean(losses):7.2f}  "
              f"80% coverage {cover80:.3f}{marker}")


if __name__ == "__main__":
    main()
