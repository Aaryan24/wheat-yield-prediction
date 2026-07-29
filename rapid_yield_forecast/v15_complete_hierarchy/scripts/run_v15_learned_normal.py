#!/usr/bin/env python3
"""Learn a district normal from strict 10-20 year history forecasts.

The target is the error left by the robust three-year normal.  Inputs contain
only competing history-only forecasts, their disagreements, state identity,
and calendar time.  No weather from the target season is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


V15 = Path(__file__).resolve().parents[1]
ROOT = V15.parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACTS = V15 / "artifacts"
SOURCE = ARTIFACTS / "normal_candidate_predictions.parquet"
TARGET = "yield_kg_per_ha"
YEARS = tuple(range(2016, 2023))
TEST_YEARS = (2019, 2020, 2021, 2022)
FOLD_END = {2016: 2015, 2017: 2016, 2018: 2017, 2019: 2018,
            2020: 2019, 2021: 2020, 2022: 2020}
SEEDS = (42, 73)


def make_wide(source: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "district_id", "state_name", "district_name",
        "season_start_year", TARGET, "lag_1_yield",
    ]
    wide = source.pivot_table(
        index=keys, columns="normal_candidate", values="normal_prediction"
    ).reset_index()
    wide.columns.name = None
    candidates = sorted(source["normal_candidate"].unique())
    for candidate in candidates:
        wide[f"delta__{candidate}"] = wide[candidate] - wide["weighted3"]
    wide["candidate_mean"] = wide[candidates].mean(axis=1)
    wide["candidate_sd"] = wide[candidates].std(axis=1)
    wide["candidate_min"] = wide[candidates].min(axis=1)
    wide["candidate_max"] = wide[candidates].max(axis=1)
    wide["candidate_range"] = wide["candidate_max"] - wide["candidate_min"]
    wide["year_scaled"] = (wide["season_start_year"] - 2000) / 20.0
    return wide


def design(
    frame: pd.DataFrame,
    features: list[str],
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    numeric = frame[features].replace([np.inf, -np.inf], np.nan).copy()
    states = pd.get_dummies(
        frame["state_name"], prefix="state", dtype=float
    ).reset_index(drop=True)
    numeric = numeric.reset_index(drop=True)
    matrix = pd.concat([numeric, states], axis=1)
    if columns is None:
        columns = matrix.columns.tolist()
    return matrix.reindex(columns=columns, fill_value=0), columns


def estimator(name: str, seed: int):
    if name.startswith("ridge"):
        alpha = float(name.replace("ridge", ""))
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(), Ridge(alpha=alpha),
        )
    if name == "extra":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesRegressor(
                n_estimators=600, max_depth=4, min_samples_leaf=20,
                max_features=0.75, random_state=seed, n_jobs=6,
            ),
        )
    if name == "xgb1":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBRegressor(
                n_estimators=400, max_depth=1, learning_rate=0.02,
                min_child_weight=25, subsample=0.85, colsample_bytree=0.75,
                reg_lambda=80, reg_alpha=8, tree_method="hist", n_jobs=6,
                random_state=seed, objective="reg:squarederror",
            ),
        )
    raise ValueError(name)


def metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    error = frame[column].to_numpy(float) - frame[TARGET].to_numpy(float)
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "direction_accuracy": float(np.mean(
            (frame[column] > frame["lag_1_yield"])
            == (frame[TARGET] > frame["lag_1_yield"])
        )),
    }


def main() -> None:
    source = pd.read_parquet(SOURCE)
    wide = make_wide(source)
    candidates = sorted(source["normal_candidate"].unique())
    features = (
        [f"delta__{candidate}" for candidate in candidates]
        + ["candidate_mean", "candidate_sd", "candidate_min",
           "candidate_max", "candidate_range", "year_scaled"]
    )
    rows: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    model_names = ("ridge10", "ridge100", "ridge1000", "extra", "xgb1")
    for year in YEARS:
        train_end = FOLD_END[year]
        train = wide[wide["season_start_year"].between(2010, train_end)].copy()
        test = wide[wide["season_start_year"].eq(year)].copy()
        x_train, columns = design(train, features)
        x_test, _ = design(test, features, columns)
        target = train[TARGET].to_numpy(float) - train["weighted3"].to_numpy(float)
        for name in model_names:
            seeds = SEEDS if name in {"extra", "xgb1"} else (42,)
            values = []
            for seed in seeds:
                model = estimator(name, seed)
                model.fit(x_train, target)
                values.append(model.predict(x_test))
            correction = np.mean(values, axis=0)
            block = test[[
                "district_id", "state_name", "district_name",
                "season_start_year", TARGET, "lag_1_yield", "weighted3",
            ]].copy()
            block["normal_candidate"] = f"learned_meta__{name}"
            block["normal_prediction"] = np.clip(
                block["weighted3"].to_numpy(float) + correction, 500, 7000
            )
            block["meta_correction"] = correction
            rows.append(block)
            audit.append({
                "test_year": year, "train_end": train_end,
                "model": name, "train_rows": len(train),
                "train_year_min": int(train["season_start_year"].min()),
                "train_year_max": int(train["season_start_year"].max()),
                "features": len(features),
            })
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_parquet(
        ARTIFACTS / "learned_normal_predictions.parquet", index=False
    )
    pd.DataFrame(audit).to_csv(
        ARTIFACTS / "learned_normal_training_audit.csv", index=False
    )
    metric_rows = []
    controls = wide[wide["season_start_year"].isin(TEST_YEARS)].copy()
    for period, years in (
        ("development", (2019, 2020)),
        ("late", (2021, 2022)),
        ("four_year", TEST_YEARS),
    ):
        for candidate, block in predictions[
            predictions["season_start_year"].isin(years)
        ].groupby("normal_candidate"):
            metric_rows.append({
                "period": period, "candidate": candidate,
                **metrics(block, "normal_prediction"),
            })
        control = controls[controls["season_start_year"].isin(years)]
        for candidate in ("weighted3", "xgb1", "extra"):
            metric_rows.append({
                "period": period, "candidate": f"control__{candidate}",
                **metrics(control, candidate),
            })
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(
        ARTIFACTS / "learned_normal_metrics.csv", index=False
    )
    summary = {
        "best_development": (
            metric_frame[metric_frame["period"].eq("development")]
            .sort_values("rmse").iloc[0].to_dict()
        ),
        "best_late": (
            metric_frame[metric_frame["period"].eq("late")]
            .sort_values("rmse").iloc[0].to_dict()
        ),
        "best_four_year": (
            metric_frame[metric_frame["period"].eq("four_year")]
            .sort_values("rmse").iloc[0].to_dict()
        ),
        "post_2022_yield_labels_read": False,
    }
    with (ARTIFACTS / "learned_normal_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
