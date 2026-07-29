#!/usr/bin/env python3
"""Add uncertainty audits and compile the final V13 policy artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score


V13 = Path(__file__).resolve().parents[1]
ROOT = V13.parents[1]
sys.path.insert(0, str(ROOT))
OUT = V13 / "artifacts"
V7_INTEGRATED = (
    V13.parent / "v7_forecast_research" / "track_integrated"
    / "artifacts" / "integrated_policy_predictions.parquet"
)
V12_MANIFEST = V13.parent / "v12_cross_attention_yield" / "data" / "manifest.json"

from rapid_yield_forecast.v13_crop_response_final.scripts.run_v13_final import trajectory_uncertainty  # noqa: E402


def safe_auc(target: pd.Series, probability: pd.Series) -> float:
    return (
        float(roc_auc_score(target, probability))
        if target.nunique() == 2
        else float("nan")
    )


def main() -> None:
    trajectory = pd.read_parquet(OUT / "trajectory_predictions.parquet")
    uncertainty = trajectory_uncertainty(trajectory)
    uncertainty.to_csv(OUT / "trajectory_uncertainty.csv", index=False)

    final = pd.read_parquet(OUT / "final_predictions.parquet")
    interval_columns = [
        "lo50_integrated", "hi50_integrated",
        "lo80_integrated", "hi80_integrated",
        "lo90_integrated", "hi90_integrated",
    ]
    final = final.drop(columns=[c for c in interval_columns if c in final], errors="ignore")
    intervals = pd.read_parquet(V7_INTEGRATED)
    intervals = intervals[intervals["clock"].isin(["jan15", "feb15", "mar05"])][
        ["district_id", "season_start_year", "clock"] + interval_columns
    ]
    final = final.merge(
        intervals,
        on=["district_id", "season_start_year", "clock"],
        how="left",
        validate="one_to_one",
    )
    final.to_parquet(OUT / "final_predictions.parquet", index=False)
    rows = []
    state_year_rows = []
    for (clock, period), part in final.groupby(["clock", "period"]):
        interval_valid = part["lo80_integrated"].notna() & part["hi80_integrated"].notna()
        rows.append({
            "clock": clock,
            "period": period,
            "point_rmse": float(mean_squared_error(part["actual"], part["prediction"]) ** 0.5),
            "point_mae": float(mean_absolute_error(part["actual"], part["prediction"])),
            "point_bias": float(np.mean(part["prediction"] - part["actual"])),
            "increase_auc": safe_auc(part["increase_target"], part["increase_probability"]),
            "increase_brier": float(brier_score_loss(part["increase_target"], part["increase_probability"])),
            "increase_accuracy_at_0_5": float(
                np.mean((part["increase_probability"] >= 0.5) == part["increase_target"].astype(bool))
            ),
            "increase_baseline_auc": safe_auc(
                part["increase_target"], part["increase_baseline_probability"]
            ),
            "increase_baseline_brier": float(
                brier_score_loss(part["increase_target"], part["increase_baseline_probability"])
            ),
            "severe_auc": safe_auc(part["severe_target"], part["severe_probability"]),
            "severe_brier": float(brier_score_loss(part["severe_target"], part["severe_probability"])),
            "interval_80_coverage": (
                float(np.mean(
                    (part.loc[interval_valid, "actual"] >= part.loc[interval_valid, "lo80_integrated"])
                    & (part.loc[interval_valid, "actual"] <= part.loc[interval_valid, "hi80_integrated"])
                ))
                if interval_valid.any() else float("nan")
            ),
            "interval_80_mean_width": (
                float(np.mean(
                    part.loc[interval_valid, "hi80_integrated"]
                    - part.loc[interval_valid, "lo80_integrated"]
                ))
                if interval_valid.any() else float("nan")
            ),
            "rows": len(part),
        })
        for (state, year), cell in part.groupby(["state_name", "season_start_year"]):
            state_year_rows.append({
                "clock": clock,
                "period": period,
                "state_name": state,
                "season_start_year": int(year),
                "point_rmse": float(mean_squared_error(cell["actual"], cell["prediction"]) ** 0.5),
                "increase_auc": safe_auc(cell["increase_target"], cell["increase_probability"]),
                "increase_brier": float(
                    brier_score_loss(cell["increase_target"], cell["increase_probability"])
                ),
                "severe_auc": safe_auc(cell["severe_target"], cell["severe_probability"]),
                "rows": len(cell),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "final_metrics.csv", index=False)
    pd.DataFrame(state_year_rows).to_csv(OUT / "state_year_final_metrics.csv", index=False)

    policy_path = OUT / "final_policy.json"
    policy = json.loads(policy_path.read_text())
    point_contract = bool(
        pd.read_csv(OUT / "trajectory_metrics.csv")
        .pivot(index="period", columns="variant", values="transition_rmse")
        .pipe(
            lambda table:
            table.loc["development", "full"] < table.loc["development", "no_future"]
            and table.loc["late", "full"] <= table.loc["late", "no_future"]
        )
    )
    late_future = uncertainty[
        uncertainty["period"].eq("late")
        & uncertainty["comparison"].eq("no_future_minus_full")
    ].iloc[0]
    strict_future = bool(point_contract and late_future["p025"] > 0)
    policy["future_weather_crop_trajectory_point_contract_pass"] = point_contract
    policy["future_weather_crop_trajectory_promoted"] = strict_future
    policy["future_weather_crop_trajectory_status"] = (
        "strictly promoted"
        if strict_future
        else "promising point estimate; future-weather increment is not uncertainty-stable"
    )
    policy["strict_final_metrics"] = metrics.to_dict("records")
    policy_path.write_text(json.dumps(policy, indent=2, default=str))
    v12_manifest = json.loads(V12_MANIFEST.read_text())
    direction_selected = pd.read_csv(OUT / "direction_selected_metrics.csv")
    point_selected = pd.read_csv(OUT / "point_selected_metrics.csv")
    deployment_manifest = {
        "final_outputs": [
            "district yield point prediction in kg/ha",
            "50%, 80%, and 90% yield ranges",
            "probability yield increases from last season",
            "probability of a severe yield decline",
            "research crop-trajectory outlook",
        ],
        "forecast_clocks": ["jan15", "feb15", "mar05"],
        "live_inputs": {
            "identity": ["district_id", "forecast clock", "season"],
            "yield_history": ["at least the previous three district yields"],
            "satellite": v12_manifest["crop_indices"],
            "satellite_views": v12_manifest["crop_views"],
            "experienced_weather": "six issue-safe crop-stage weather tokens",
            "future_weather": "ten dated forecast tokens, latest issue at least two days old",
            "economic": "lagged economic variables only",
            "static": ["soil available water", "bulk density", "drainage", "crop area", "phenology"],
        },
        "satellite_cleaning": {
            "rule": "abs(PSRI) > 2 is missing",
            "cells_removed_in_research_panel": 96,
        },
        "tabular_feature_order": v12_manifest["tabular_columns"],
        "march_direction_head": direction_selected[
            direction_selected["clock"].eq("mar05")
            & direction_selected["period"].eq("late")
        ].iloc[0].to_dict(),
        "point_selection": point_selected.to_dict("records"),
        "blend_rule": "March p(increase) = 0.85 * V12 probability + 0.15 * V13 head probability",
        "baseline_artifacts": {
            "point_and_intervals": str(V7_INTEGRATED.relative_to(V13.parent)),
            "direction": "v12_cross_attention_yield/artifacts/direction_increment/selected_predictions.parquet",
            "severe": "v11_global_wheat_transfer/artifacts/*/blended_predictions.parquet",
        },
        "deployment_models": policy["deployment"],
    }
    (OUT / "deployment_manifest.json").write_text(
        json.dumps(deployment_manifest, indent=2, default=str)
    )
    print(metrics.to_string(index=False))
    print("\nTRAJECTORY UNCERTAINTY\n", uncertainty.to_string(index=False))


if __name__ == "__main__":
    main()
