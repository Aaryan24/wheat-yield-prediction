#!/usr/bin/env python3
"""Create the complete V15 evidence audit without changing model selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


V15 = Path(__file__).resolve().parents[1]
ARTIFACTS = V15 / "artifacts"
TARGET = "yield_kg_per_ha"
DEVELOPMENT = (2019, 2020)
LATE = (2021, 2022)
YEARS = DEVELOPMENT + LATE


def point_metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    error = frame[column].to_numpy(float) - frame[TARGET].to_numpy(float)
    year_rmse = [
        float(np.sqrt(np.mean(
            (block[column] - block[TARGET]) ** 2
        )))
        for _, block in frame.groupby("season_start_year")
    ]
    state_rmse = [
        float(np.sqrt(np.mean(
            (block[column] - block[TARGET]) ** 2
        )))
        for _, block in frame.groupby("state_name")
    ]
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "mean_year_rmse": float(np.mean(year_rmse)),
        "max_year_rmse": float(np.max(year_rmse)),
        "equal_state_rmse": float(np.mean(state_rmse)),
        "direction_accuracy": float(np.mean(
            (frame[column] > frame["lag_1_yield"])
            == (frame[TARGET] > frame["lag_1_yield"])
        )),
    }


def grouped_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    grouping: list[str],
    draws: int = 10000,
) -> dict[str, float]:
    groups = [
        block.index.to_numpy()
        for _, block in frame.groupby(grouping)
    ]
    rng = np.random.default_rng(20260803 + len(grouping))
    gains = np.empty(draws)
    for draw in range(draws):
        indices = np.concatenate([
            groups[index]
            for index in rng.integers(0, len(groups), len(groups))
        ])
        actual = frame.loc[indices, TARGET].to_numpy(float)
        base_rmse = np.sqrt(np.mean(
            (frame.loc[indices, baseline].to_numpy(float) - actual) ** 2
        ))
        model_rmse = np.sqrt(np.mean(
            (frame.loc[indices, candidate].to_numpy(float) - actual) ** 2
        ))
        gains[draw] = base_rmse - model_rmse
    return {
        "draws": draws,
        "groups": len(groups),
        "mean_gain": float(np.mean(gains)),
        "median_gain": float(np.median(gains)),
        "p025": float(np.quantile(gains, 0.025)),
        "p975": float(np.quantile(gains, 0.975)),
        "probability_positive": float(np.mean(gains > 0)),
    }


def main() -> None:
    selected = pd.read_parquet(
        ARTIFACTS / "encoder_correction_selected_predictions.parquet"
    )
    regularized = selected[
        selected["selection_rule"].eq("regularized_near_tie")
    ].copy()
    exact = selected[
        selected["selection_rule"].eq("exact_development")
    ][["district_id", "season_start_year", "prediction"]].rename(
        columns={"prediction": "v15_exact"}
    )
    frame = regularized.merge(
        exact, on=["district_id", "season_start_year"], validate="one_to_one"
    )
    frame["v15_regularized"] = frame["prediction"]
    frame["v15_current_only"] = (
        frame["v5_prediction"] + 1.25 * frame["raw_correction"]
    )
    frame["v15_conservative_all_year_shadow"] = (
        frame["shadow_point_prediction"] + 0.125 * frame["raw_correction"]
    )
    model_columns = {
        "V5_production": "v5_prediction",
        "V14_future_weather_shadow": "shadow_point_prediction",
        "V15_current_crop_only": "v15_current_only",
        "V15_combined_regularized": "v15_regularized",
        "V15_combined_exact": "v15_exact",
        "V15_conservative_shadow": "v15_conservative_all_year_shadow",
    }
    metric_rows = []
    for name, column in model_columns.items():
        for period, years in (
            ("development", DEVELOPMENT),
            ("late", LATE),
            ("four_year", YEARS),
        ):
            block = frame[frame["season_start_year"].isin(years)]
            metric_rows.append({
                "model": name, "period": period,
                **point_metrics(block, column),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ARTIFACTS / "point_model_ablation_metrics.csv", index=False)

    audit_rows = []
    for level, keys in (
        ("year", ["season_start_year"]),
        ("state_year", ["season_start_year", "state_name"]),
    ):
        for key, block in frame.groupby(keys):
            if not isinstance(key, tuple):
                key = (key,)
            for name, column in model_columns.items():
                values = point_metrics(block, column)
                audit_rows.append({
                    "level": level,
                    "season_start_year": int(key[0]),
                    "state_name": key[1] if len(key) > 1 else "ALL",
                    "model": name, "rows": len(block), **values,
                })
    audit = pd.DataFrame(audit_rows)
    baseline = audit[audit["model"].eq("V5_production")][[
        "level", "season_start_year", "state_name", "rmse",
    ]].rename(columns={"rmse": "v5_rmse"})
    audit = audit.merge(
        baseline,
        on=["level", "season_start_year", "state_name"],
        validate="many_to_one",
    )
    audit["rmse_gain_vs_v5"] = audit["v5_rmse"] - audit["rmse"]
    audit.to_csv(ARTIFACTS / "point_year_state_audit.csv", index=False)

    bootstrap_rows = []
    for candidate in (
        "shadow_point_prediction", "v15_regularized",
        "v15_conservative_all_year_shadow",
    ):
        for period, years in (
            ("late", LATE), ("four_year", YEARS),
        ):
            block = frame[
                frame["season_start_year"].isin(years)
            ].reset_index(drop=True)
            for label, grouping in (
                ("state_year", ["state_name", "season_start_year"]),
                ("year", ["season_start_year"]),
            ):
                bootstrap_rows.append({
                    "candidate": candidate, "baseline": "v5_prediction",
                    "period": period, "resampling_unit": label,
                    **grouped_bootstrap(
                        block, candidate, "v5_prediction", grouping
                    ),
                })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(
        ARTIFACTS / "point_grouped_bootstrap.csv", index=False
    )

    trajectory = pd.read_csv(
        ARTIFACTS / "encoder_trajectory_metrics.csv"
    )
    normal = pd.read_csv(ARTIFACTS / "learned_normal_metrics.csv")
    distribution = pd.read_csv(
        ARTIFACTS / "v15_distribution_metrics.csv"
    )
    hierarchy = pd.read_csv(
        ARTIFACTS / "hierarchy_sensitivity_metrics.csv"
    )
    promoted = metrics[
        metrics["model"].eq("V15_combined_regularized")
    ]
    v5 = metrics[metrics["model"].eq("V5_production")]
    merged = promoted.merge(v5, on="period", suffixes=("_v15", "_v5"))
    summary = {
        "promoted_point_metrics": promoted.to_dict("records"),
        "point_gain_vs_v5": [
            {
                "period": row.period,
                "rmse_gain": float(row.rmse_v5 - row.rmse_v15),
                "direction_change": float(
                    row.direction_accuracy_v15 - row.direction_accuracy_v5
                ),
            }
            for row in merged.itertuples()
        ],
        "year_count_improved_vs_v5": int(
            (
                audit[
                    audit["level"].eq("year")
                    & audit["model"].eq("V15_combined_regularized")
                ]["rmse_gain_vs_v5"] > 0
            ).sum()
        ),
        "state_year_count_improved_vs_v5": int(
            (
                audit[
                    audit["level"].eq("state_year")
                    & audit["model"].eq("V15_combined_regularized")
                ]["rmse_gain_vs_v5"] > 0
            ).sum()
        ),
        "year_count": 4,
        "state_year_count": 12,
        "trajectory": trajectory.to_dict("records"),
        "best_learned_normal_development": (
            normal[normal["period"].eq("development")]
            .sort_values("rmse").iloc[0].to_dict()
        ),
        "best_learned_normal_late": (
            normal[normal["period"].eq("late")]
            .sort_values("rmse").iloc[0].to_dict()
        ),
        "best_hierarchy_development": (
            hierarchy[hierarchy["period"].eq("development")]
            .sort_values(["selection_score", "rmse"]).iloc[0].to_dict()
        ),
        "distribution": distribution.to_dict("records"),
        "bootstrap": bootstrap.to_dict("records"),
        "promotion_decision": (
            "V15 combined is a research frontier challenger. "
            "The state-year bootstrap is positive, but V5 remains the "
            "conservative production anchor because the bootstrap over only "
            "four independent years still crosses zero."
        ),
        "post_2022_yield_labels_read": False,
    }
    with (ARTIFACTS / "audit_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({
        "promoted": summary["promoted_point_metrics"],
        "point_gain_vs_v5": summary["point_gain_vs_v5"],
        "year_count_improved_vs_v5": summary["year_count_improved_vs_v5"],
        "state_year_count_improved_vs_v5": summary[
            "state_year_count_improved_vs_v5"
        ],
        "bootstrap": summary["bootstrap"],
    }, indent=2))


if __name__ == "__main__":
    main()
