#!/usr/bin/env python3
"""Build and honestly test district probability distributions for V15."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


V15 = Path(__file__).resolve().parents[1]
RAPID = V15.parent
ARTIFACTS = V15 / "artifacts"
V14_FINAL = RAPID / "v14_anomaly_distribution" / "artifacts" / "final_predictions.parquet"
SELECTED = ARTIFACTS / "encoder_correction_selected_predictions.parquet"
TARGET = "yield_kg_per_ha"
QUANTILES = np.round(np.arange(0.05, 1.0, 0.05), 2)
QCOLS = [f"q{int(q * 100):02d}" for q in QUANTILES]
DEVELOPMENT = (2019, 2020)
LATE = (2021, 2022)
YEARS = DEVELOPMENT + LATE


def selected_point() -> pd.DataFrame:
    selected = pd.read_parquet(SELECTED)
    selected = selected[
        selected["selection_rule"].eq("regularized_near_tie")
    ].copy()
    if selected.duplicated(["district_id", "season_start_year"]).any():
        raise RuntimeError("Selected V15 point rows are not unique")
    return selected[[
        "district_id", "season_start_year", "prediction",
        "raw_correction", "gamma", "correction_candidate",
    ]].rename(columns={"prediction": "v15_point_prediction"})


def interval_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    width = upper - lower
    penalty_low = (2 / alpha) * (lower - actual) * (actual < lower)
    penalty_high = (2 / alpha) * (actual - upper) * (actual > upper)
    return float(np.mean(width + penalty_low + penalty_high))


def quantile_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame[TARGET].to_numpy(float)
    pinball = []
    calibration = []
    for q, column in zip(QUANTILES, QCOLS):
        error = actual - frame[column].to_numpy(float)
        pinball.append(np.mean(np.maximum(q * error, (q - 1) * error)))
        calibration.append(abs(np.mean(actual <= frame[column]) - q))
    return {
        "coverage_80": float(np.mean(
            (actual >= frame["q10"]) & (actual <= frame["q90"])
        )),
        "coverage_90": float(np.mean(
            (actual >= frame["q05"]) & (actual <= frame["q95"])
        )),
        "width_80": float(np.mean(frame["q90"] - frame["q10"])),
        "width_90": float(np.mean(frame["q95"] - frame["q05"])),
        "interval_score_80": interval_score(
            actual, frame["q10"].to_numpy(float),
            frame["q90"].to_numpy(float), 0.20,
        ),
        "interval_score_90": interval_score(
            actual, frame["q05"].to_numpy(float),
            frame["q95"].to_numpy(float), 0.10,
        ),
        "mean_pinball": float(np.mean(pinball)),
        "quantile_calibration_mae": float(np.mean(calibration)),
    }


def shift_distribution(
    source: pd.DataFrame,
    center: str,
    scale: float,
) -> pd.DataFrame:
    block = source.copy()
    old_center = block["production_point_prediction"].to_numpy(float)
    new_center = block[center].to_numpy(float)
    values = np.column_stack([
        new_center + scale * (block[column].to_numpy(float) - old_center)
        for column in QCOLS
    ])
    values = np.maximum.accumulate(values, axis=1)
    for index, column in enumerate(QCOLS):
        block[column] = np.clip(values[:, index], 500, 7000)
    block["distribution_center"] = center
    block["distribution_scale"] = scale
    return block


def probability_below(
    values: np.ndarray,
    threshold: float,
) -> float:
    extended_values = np.r_[values[0] - 500, values, values[-1] + 500]
    extended_q = np.r_[0.0, QUANTILES, 1.0]
    return float(np.clip(
        np.interp(threshold, extended_values, extended_q), 0, 1
    ))


def add_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    matrix = result[QCOLS].to_numpy(float)
    rise = []
    severe = []
    for index, values in enumerate(matrix):
        previous = float(result.iloc[index]["lag_1_yield"])
        rise.append(1.0 - probability_below(values, previous))
        severe.append(probability_below(values, 0.90 * previous))
    result["probability_rise"] = rise
    result["probability_severe_drop"] = severe
    return result


def auc_metrics(frame: pd.DataFrame) -> dict[str, float]:
    rise_actual = (frame[TARGET] > frame["lag_1_yield"]).astype(int)
    severe_actual = (frame[TARGET] < 0.90 * frame["lag_1_yield"]).astype(int)
    return {
        "rise_auc": float(roc_auc_score(
            rise_actual, frame["probability_rise"]
        )),
        "severe_drop_auc": float(roc_auc_score(
            severe_actual, frame["probability_severe_drop"]
        )),
        "rise_brier": float(np.mean(
            (frame["probability_rise"] - rise_actual) ** 2
        )),
        "severe_drop_brier": float(np.mean(
            (frame["probability_severe_drop"] - severe_actual) ** 2
        )),
    }


def main() -> None:
    source = pd.read_parquet(V14_FINAL).merge(
        selected_point(),
        on=["district_id", "season_start_year"], validate="one_to_one",
    )
    centers = {
        "production_point_prediction": "V5",
        "shadow_point_prediction": "V14_shadow",
        "v15_point_prediction": "V15_frontier",
    }
    rows = []
    candidates = []
    for center, label in centers.items():
        for scale in np.round(np.arange(0.75, 1.51, 0.05), 2):
            block = shift_distribution(source, center, float(scale))
            block["distribution_candidate"] = f"{label}__scale_{scale:.2f}"
            candidates.append(block)
            dev = block[block["season_start_year"].isin(DEVELOPMENT)]
            values = quantile_metrics(dev)
            rows.append({
                "distribution_candidate": block["distribution_candidate"].iloc[0],
                "center": center, "center_label": label,
                "scale": float(scale), "period": "development", **values,
            })
    grid = pd.DataFrame(rows)
    # Pinball loss is a proper distribution score.  For the released range,
    # also require the development 80% interval to behave like an 80% range
    # and the 90% interval to cover at least 88%.  This prevents a sharp but
    # overconfident interval from winning merely by being narrow.
    grid["selection_score"] = (
        grid["mean_pinball"] + 0.05 * (grid["scale"] - 1.0).abs()
    )
    pinball_winner = grid.sort_values([
        "selection_score", "quantile_calibration_mae",
        "distribution_candidate",
    ]).iloc[0]
    calibrated = grid[
        grid["coverage_80"].between(0.78, 0.82)
        & grid["coverage_90"].ge(0.88)
    ]
    if calibrated.empty:
        raise RuntimeError("No development-calibrated distribution candidate")
    winner = calibrated.sort_values([
        "selection_score", "quantile_calibration_mae",
        "distribution_candidate",
    ]).iloc[0]
    selected = next(
        block for block in candidates
        if block["distribution_candidate"].iloc[0]
        == winner["distribution_candidate"]
    )
    selected = add_probabilities(selected)
    selected.to_parquet(
        ARTIFACTS / "v15_distribution_predictions.parquet", index=False
    )
    grid.to_csv(ARTIFACTS / "v15_distribution_grid.csv", index=False)

    metric_rows = []
    for period, years in (
        ("development", DEVELOPMENT),
        ("late", LATE),
        ("four_year", YEARS),
    ):
        part = selected[selected["season_start_year"].isin(years)]
        metric_rows.append({
            "period": period,
            "distribution_candidate": winner["distribution_candidate"],
            **quantile_metrics(part), **auc_metrics(part),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ARTIFACTS / "v15_distribution_metrics.csv", index=False)

    final = selected[[
        "district_id", "state_name", "district_name",
        "season_start_year", TARGET, "lag_1_yield",
        "production_point_prediction", "shadow_point_prediction",
        "v15_point_prediction", "raw_correction", "gamma",
        "correction_candidate", *QCOLS, "probability_rise",
        "probability_severe_drop", "distribution_candidate",
        "distribution_center", "distribution_scale",
    ]].copy()
    final["period"] = np.where(
        final["season_start_year"].isin(DEVELOPMENT),
        "development", "late",
    )
    final.to_parquet(ARTIFACTS / "final_predictions.parquet", index=False)
    summary = {
        "selected_distribution": winner.to_dict(),
        "pinball_optimum_not_released": pinball_winner.to_dict(),
        "metrics": metrics.to_dict("records"),
        "selection_years": list(DEVELOPMENT),
        "confirmation_years": list(LATE),
        "post_2022_yield_labels_read": False,
    }
    with (ARTIFACTS / "distribution_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
