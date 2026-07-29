#!/usr/bin/env python3
"""Test strict V15 crop-trajectory representations inside and beside V5.

Every representation used for a training district is cross-fitted by district.
The held-out year representation is produced by an encoder that did not see that
year.  Model and correction choices are selected on 2019-2020 only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


V15 = Path(__file__).resolve().parents[1]
ROOT = V15.parents[1]
sys.path.insert(0, str(ROOT))
RAPID = V15.parent
DATA = V15 / "data"
ARTIFACTS = V15 / "artifacts"
ENCODER = DATA / "strict_transfer_encoder_features.parquet"
V5_PRED = (
    RAPID / "v5" / "root_cybench_lab" / "artifacts"
    / "v5_integration" / "predictions.csv"
)
V14_FINAL = RAPID / "v14_anomaly_distribution" / "artifacts" / "final_predictions.parquet"

from rapid_yield_forecast.v14_anomaly_distribution.scripts import run_v14_lab as lab  # noqa: E402


TARGET = "yield_kg_per_ha"
YEARS = (2019, 2020, 2021, 2022)
DEVELOPMENT = (2019, 2020)
LATE = (2021, 2022)
FOLD_END = {2019: 2018, 2020: 2019, 2021: 2020, 2022: 2020}
VARIANTS = ("scratch", "modis_pretrained")
BASE_GROUPS = ("physical", "physical_modis")
DEPTHS = (1, 2)


def select_encoder(
    encoder: pd.DataFrame,
    train_end: int,
    test_year: int,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    selected = encoder[
        encoder["representation_train_end"].eq(train_end)
        & encoder["encoder_variant"].eq(variant)
        & (
            encoder["feature_role"].eq("train_crossfit")
            | (
                encoder["feature_role"].eq("test_full")
                & encoder["season_start_year"].eq(test_year)
            )
        )
    ].copy()
    columns = [column for column in selected if column.startswith("enc__")]
    selected = selected.drop(columns=[
        "state_name", "district_name", "clock", "representation_train_end",
        "feature_role", "held_group", "encoder_variant",
    ])
    keys = ["district_id", "season_start_year"]
    if selected.duplicated(keys).any():
        raise RuntimeError("Encoder cross-fit rows are not unique")

    current = [
        column for column in columns
        if "current_index_" in column
        or "no_future_delta_" in column
        or "no_future_fused_pool_" in column
    ]
    full = [
        column for column in columns
        if "current_index_" in column
        or "full_delta_" in column
        or "full_fused_pool_" in column
    ]
    effect = [
        column for column in columns
        if "current_index_" in column or "future_effect_" in column
    ]
    transition = [
        column for column in columns
        if "current_index_" in column
        or "no_future_delta_" in column
        or "full_delta_" in column
        or "future_effect_" in column
    ]
    pools = [
        column for column in columns
        if "current_index_" in column
        or "no_future_crop_pool_" in column
        or "full_crop_pool_" in column
        or "future_effect_" in column
    ]
    return selected, {
        "base": [],
        "current": current,
        "full": full,
        "effect": effect,
        "transition": transition,
        "pools": pools,
    }


def metric(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    return lab.metric_values(frame, prediction)


def score(values: dict[str, float] | pd.Series) -> float:
    return (
        0.50 * float(values["rmse"])
        + 0.25 * float(values["equal_state_rmse"])
        + 0.25 * float(values["mean_year_rmse"])
    )


def train_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    base, groups, _ = lab.load_panel()
    encoder = pd.read_parquet(ENCODER)
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for year in YEARS:
        train_end = FOLD_END[year]
        for variant in VARIANTS:
            encoded, feature_sets = select_encoder(
                encoder, train_end, year, variant
            )
            fold = base.merge(
                encoded,
                on=["district_id", "season_start_year"],
                how="left", validate="one_to_one",
            )
            train = fold[fold["season_start_year"].between(2017, train_end)].copy()
            test = fold[fold["season_start_year"].eq(year)].copy()
            for base_group in BASE_GROUPS:
                base_features = groups[base_group]
                for depth in DEPTHS:
                    for feature_set, encoded_features in feature_sets.items():
                        prediction = lab.xgb_residual_predict(
                            train, test, base_features + encoded_features, depth
                        )
                        block = test[[
                            "district_id", "state_name", "district_name",
                            "season_start_year", TARGET, "lag_1_yield",
                        ]].copy()
                        block["encoder_variant"] = variant
                        block["base_group"] = base_group
                        block["depth"] = depth
                        block["feature_set"] = feature_set
                        block["candidate"] = (
                            f"{variant}__{base_group}__d{depth}__{feature_set}"
                        )
                        block["prediction"] = prediction
                        rows.append(block)
                        audits.append({
                            "test_year": year,
                            "train_end": train_end,
                            "variant": variant,
                            "base_group": base_group,
                            "depth": depth,
                            "feature_set": feature_set,
                            "train_rows": len(train),
                            "test_rows": len(test),
                            "base_features": len(base_features),
                            "encoder_features": len(encoded_features),
                        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def wide_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "district_id", "state_name", "district_name",
        "season_start_year", TARGET, "lag_1_yield",
    ]
    wide = predictions.pivot_table(
        index=keys, columns="candidate", values="prediction"
    ).reset_index()
    wide.columns.name = None
    v5 = pd.read_csv(V5_PRED)[[
        "district_id", "season_start_year", "prediction",
    ]].rename(columns={"prediction": "v5_prediction"})
    shadow = pd.read_parquet(V14_FINAL)[[
        "district_id", "season_start_year", "shadow_point_prediction",
    ]]
    return (
        wide.merge(
            v5, on=["district_id", "season_start_year"],
            validate="one_to_one",
        )
        .merge(
            shadow, on=["district_id", "season_start_year"],
            validate="one_to_one",
        )
    )


def build_corrections(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        for base_group in BASE_GROUPS:
            for depth in DEPTHS:
                prefix = f"{variant}__{base_group}__d{depth}__"
                base = prefix + "base"
                definitions = {
                    "current_minus_base": (prefix + "current", base),
                    "full_minus_base": (prefix + "full", base),
                    "effect_features_minus_base": (prefix + "effect", base),
                    "transition_features_minus_base": (prefix + "transition", base),
                    "pools_minus_base": (prefix + "pools", base),
                    "full_minus_current": (prefix + "full", prefix + "current"),
                    "transition_minus_current": (
                        prefix + "transition", prefix + "current"
                    ),
                    "pools_minus_current": (prefix + "pools", prefix + "current"),
                }
                for name, (left, right) in definitions.items():
                    block = wide[[
                        "district_id", "state_name", "district_name",
                        "season_start_year", TARGET, "lag_1_yield",
                        "v5_prediction", "shadow_point_prediction",
                    ]].copy()
                    block["encoder_variant"] = variant
                    block["base_group"] = base_group
                    block["depth"] = depth
                    block["correction_name"] = name
                    block["correction_candidate"] = (
                        f"{variant}__{base_group}__d{depth}__{name}"
                    )
                    block["raw_correction"] = (
                        wide[left].to_numpy(float) - wide[right].to_numpy(float)
                    )
                    rows.append(block)
    return pd.concat(rows, ignore_index=True)


def select_corrections(
    corrections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows: list[dict[str, object]] = []
    gamma_grid = np.round(np.arange(-3.0, 3.01, 0.25), 2)
    for candidate, block in corrections.groupby("correction_candidate"):
        dev = block[block["season_start_year"].isin(DEVELOPMENT)].copy()
        for anchor in ("v5_prediction", "shadow_point_prediction"):
            for gamma in gamma_grid:
                dev["trial"] = (
                    dev[anchor].to_numpy(float)
                    + gamma * dev["raw_correction"].to_numpy(float)
                )
                values = metric(dev, "trial")
                grid_rows.append({
                    "correction_candidate": candidate,
                    "anchor": anchor,
                    "gamma": float(gamma),
                    "encoder_variant": block["encoder_variant"].iloc[0],
                    "base_group": block["base_group"].iloc[0],
                    "depth": int(block["depth"].iloc[0]),
                    "correction_name": block["correction_name"].iloc[0],
                    **values,
                    "selection_score": score(values),
                })
    grid = pd.DataFrame(grid_rows).sort_values([
        "selection_score", "rmse", "correction_candidate", "gamma",
    ])
    exact = grid.iloc[0]
    tolerance = max(0.35, 0.0015 * float(exact["selection_score"]))
    near = grid[
        grid["selection_score"].le(float(exact["selection_score"]) + tolerance)
    ].copy()
    near["gamma_abs"] = near["gamma"].abs()
    regularized = near.sort_values([
        "gamma_abs", "selection_score", "rmse", "correction_candidate",
    ]).iloc[0]

    selected_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for rule, winner in (
        ("exact_development", exact),
        ("regularized_near_tie", regularized),
    ):
        block = corrections[
            corrections["correction_candidate"].eq(
                winner["correction_candidate"]
            )
        ].copy()
        block["selected_anchor"] = winner["anchor"]
        block["gamma"] = float(winner["gamma"])
        block["selection_rule"] = rule
        block["prediction"] = (
            block[winner["anchor"]].to_numpy(float)
            + float(winner["gamma"]) * block["raw_correction"].to_numpy(float)
        )
        selected_rows.append(block)
        for period, years in (
            ("development", DEVELOPMENT),
            ("late", LATE),
            ("four_year", YEARS),
        ):
            part = block[block["season_start_year"].isin(years)]
            values = metric(part, "prediction")
            anchor_values = metric(part, winner["anchor"])
            v5_values = metric(part, "v5_prediction")
            metric_rows.append({
                "selection_rule": rule,
                "period": period,
                "correction_candidate": winner["correction_candidate"],
                "anchor": winner["anchor"],
                "gamma": float(winner["gamma"]),
                **{f"model_{key}": value for key, value in values.items()},
                **{f"anchor_{key}": value for key, value in anchor_values.items()},
                **{f"v5_{key}": value for key, value in v5_values.items()},
                "rmse_gain_vs_anchor": anchor_values["rmse"] - values["rmse"],
                "rmse_gain_vs_v5": v5_values["rmse"] - values["rmse"],
            })
    return (
        grid,
        pd.DataFrame(metric_rows),
        pd.concat(selected_rows, ignore_index=True),
    )


def independent_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, block in predictions.groupby("candidate"):
        for period, years in (
            ("development", DEVELOPMENT),
            ("late", LATE),
            ("four_year", YEARS),
        ):
            part = block[block["season_start_year"].isin(years)]
            values = metric(part, "prediction")
            rows.append({
                "candidate": candidate,
                "period": period,
                **values,
                "selection_score": score(values),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    predictions, audit = train_predictions()
    predictions.to_parquet(
        ARTIFACTS / "encoder_xgb_candidate_predictions.parquet", index=False
    )
    audit.to_csv(ARTIFACTS / "encoder_xgb_training_audit.csv", index=False)
    independent = independent_metrics(predictions)
    independent.to_csv(
        ARTIFACTS / "encoder_xgb_candidate_metrics.csv", index=False
    )
    wide = wide_predictions(predictions)
    corrections = build_corrections(wide)
    corrections.to_parquet(
        ARTIFACTS / "encoder_isolated_corrections.parquet", index=False
    )
    grid, selected_metrics, selected = select_corrections(corrections)
    grid.to_csv(
        ARTIFACTS / "encoder_correction_selection_grid.csv", index=False
    )
    selected_metrics.to_csv(
        ARTIFACTS / "encoder_correction_selected_metrics.csv", index=False
    )
    selected.to_parquet(
        ARTIFACTS / "encoder_correction_selected_predictions.parquet", index=False
    )
    summary = {
        "candidate_count": int(predictions["candidate"].nunique()),
        "correction_count": int(
            corrections["correction_candidate"].nunique()
        ),
        "independent_best_development": (
            independent[independent["period"].eq("development")]
            .sort_values(["selection_score", "rmse"]).iloc[0].to_dict()
        ),
        "selected_corrections": selected_metrics.to_dict("records"),
        "post_2022_yield_labels_read": False,
    }
    with (ARTIFACTS / "integration_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
