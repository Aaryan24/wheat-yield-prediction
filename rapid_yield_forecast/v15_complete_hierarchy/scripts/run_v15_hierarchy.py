#!/usr/bin/env python3
"""Train and evaluate the complete V15 normal-shock-exposure hierarchy."""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
V15 = Path(__file__).resolve().parents[1]
ROOT = V15.parents[1]
sys.path.insert(0, str(ROOT))
RAPID = V15.parent
DATA = V15 / "data"
ARTIFACTS = V15 / "artifacts"
MODELS = V15 / "models"
LONG_YIELD = DATA / "long_yield_1990_2022.parquet"
BASE_PATH = RAPID / "v3" / "data" / "feature_table_v3_extended_history_03-05.parquet"
ENCODER_PATH = DATA / "strict_transfer_encoder_features.parquet"
V5_PATH = RAPID / "v5" / "root_cybench_lab" / "artifacts" / "v5_integration" / "predictions.csv"

from rapid_yield_forecast.v5.agent_model_lab.scripts import run_v5_model_lab as v5lab  # noqa: E402
from rapid_yield_forecast.v14_anomaly_distribution.scripts import run_v14_lab as v14lab  # noqa: E402


TARGET = "yield_kg_per_ha"
TEST_YEARS = [2019, 2020, 2021, 2022]
DEVELOPMENT = [2019, 2020]
LATE = [2021, 2022]
FOLD_END = {2019: 2018, 2020: 2019, 2021: 2020, 2022: 2020}
SEEDS = (42, 73)
NORMAL_CANDIDATES = (
    "weighted3", "mean10", "trend20", "ewma20",
    "ridge10", "ridge100", "ridge1000",
    "extra", "xgb1", "xgb2",
)
STATE_MODELS = ("ridge10", "ridge100", "ridge1000", "extra", "xgb1")
RESIDUAL_MODELS = ("zero", "ridge100", "ridge1000", "extra", "xgb1", "xgb2")
ENCODER_VARIANTS = ("none", "scratch", "modis_pretrained")

HISTORY_FEATURES = [
    "lag_1_yield", "lag_2_yield", "lag_3_yield", "lag_4_yield", "lag_5_yield",
    "yield_recent_mean", "yield_recent_std", "yield_recent_slope",
    "state_lag_1_mean_yield", "lag_1_minus_state",
    "ext_recent_5_mean", "ext_recent_5_std", "ext_recent_5_slope",
    "ext_recent_10_mean", "ext_recent_10_std", "ext_recent_10_slope",
    "ext_lag1_minus_state", "ext_prior_10_proxy_share",
]
STATIC_FEATURES = [
    "static_awc", "static_bulk_density", "static_drainage_class",
    "static_crop_area_percentage", "static_sos_doy", "static_eos_doy",
]


def finite_columns(
    frame: pd.DataFrame,
    columns: list[str],
    minimum: float = 0.20,
) -> list[str]:
    result = []
    for column in dict.fromkeys(columns):
        if column not in frame:
            continue
        values = frame[column].replace([np.inf, -np.inf], np.nan)
        if values.notna().mean() >= minimum and values.nunique(dropna=True) > 1:
            result.append(column)
    return result


def metric_values(
    frame: pd.DataFrame,
    prediction: str = "prediction",
) -> dict[str, float]:
    error = frame[prediction].to_numpy(float) - frame[TARGET].to_numpy(float)
    state_rmse = (
        frame.assign(_se=error ** 2)
        .groupby("state_name")["_se"].mean().mean() ** 0.5
    )
    year_rmse = [
        float(np.sqrt(np.mean(
            (part[prediction].to_numpy(float) - part[TARGET].to_numpy(float)) ** 2
        )))
        for _, part in frame.groupby("season_start_year")
    ]
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "equal_state_rmse": float(state_rmse),
        "mean_year_rmse": float(np.mean(year_rmse)),
        "max_year_rmse": float(np.max(year_rmse)),
        "direction_accuracy": float(np.mean(
            (frame[prediction] > frame["lag_1_yield"])
            == (frame[TARGET] > frame["lag_1_yield"])
        )),
    }


def selection_score(values: dict[str, float] | pd.Series) -> float:
    return (
        0.50 * float(values["rmse"])
        + 0.25 * float(values["equal_state_rmse"])
        + 0.25 * float(values["mean_year_rmse"])
    )


def add_long_history_features(long: pd.DataFrame) -> pd.DataFrame:
    frame = long.sort_values(["district_id", "season_start_year"]).copy()
    frame["log_target"] = np.log(frame[TARGET].clip(500, 7000))
    grouped = frame.groupby("district_id", sort=False)
    for lag in range(1, 21):
        frame[f"log_lag_{lag}"] = grouped["log_target"].shift(lag)
        frame[f"lag_valid_{lag}"] = frame[f"log_lag_{lag}"].notna().astype(float)
    for window in (3, 5, 10, 20):
        values = [f"log_lag_{lag}" for lag in range(1, window + 1)]
        frame[f"log_mean_{window}"] = frame[values].mean(axis=1)
        frame[f"log_std_{window}"] = frame[values].std(axis=1)
        frame[f"log_min_{window}"] = frame[values].min(axis=1)
        frame[f"log_max_{window}"] = frame[values].max(axis=1)
        frame[f"observed_{window}"] = frame[values].notna().sum(axis=1)
        slopes = []
        for row in frame[values].to_numpy(float):
            ok = np.isfinite(row)
            if ok.sum() < 3:
                slopes.append(np.nan)
            else:
                x = -np.arange(1, window + 1)[ok]
                slopes.append(float(np.polyfit(x, row[ok], 1)[0]))
        frame[f"log_slope_{window}"] = slopes

    state_year = (
        frame.groupby(["state_name", "season_start_year"], as_index=False)[TARGET]
        .mean().rename(columns={TARGET: "state_yield_mean"})
    )
    frame = frame.merge(
        state_year, on=["state_name", "season_start_year"],
        how="left", validate="many_to_one",
    )
    state_year["state_log"] = np.log(state_year["state_yield_mean"].clip(500, 7000))
    for lag in range(1, 6):
        state_year[f"state_log_lag_{lag}"] = (
            state_year.sort_values(["state_name", "season_start_year"])
            .groupby("state_name")["state_log"].shift(lag)
        )
    frame = frame.drop(columns=["state_yield_mean"]).merge(
        state_year.drop(columns=["state_yield_mean", "state_log"]),
        on=["state_name", "season_start_year"],
        how="left", validate="many_to_one",
    )
    frame["year_scaled"] = (frame["season_start_year"] - 2000) / 20.0
    return frame


def normal_feature_columns() -> list[str]:
    return (
        [f"log_lag_{lag}" for lag in range(1, 21)]
        + [f"lag_valid_{lag}" for lag in range(1, 21)]
        + [
            f"log_{stat}_{window}"
            for window in (3, 5, 10, 20)
            for stat in ("mean", "std", "min", "max", "slope")
        ]
        + [f"observed_{window}" for window in (3, 5, 10, 20)]
        + [f"state_log_lag_{lag}" for lag in range(1, 6)]
        + ["year_scaled"]
    )


def normal_design(
    frame: pd.DataFrame,
    features: list[str],
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    numeric = frame[features].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    district = pd.get_dummies(
        frame["district_id"], prefix="district", dtype=float
    ).reset_index(drop=True)
    state = pd.get_dummies(
        frame["state_name"], prefix="state", dtype=float
    ).reset_index(drop=True)
    design = pd.concat([numeric, district, state], axis=1)
    if columns is None:
        columns = design.columns.tolist()
    return design.reindex(columns=columns, fill_value=0.0), columns


def normal_estimator(name: str, seed: int):
    if name.startswith("ridge"):
        alpha = float(name.replace("ridge", ""))
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=alpha),
        )
    if name == "extra":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesRegressor(
                n_estimators=400, max_depth=7, min_samples_leaf=8,
                max_features=0.70, n_jobs=6, random_state=seed,
            ),
        )
    if name.startswith("xgb"):
        depth = int(name[-1])
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBRegressor(
                n_estimators=350, max_depth=depth, learning_rate=0.025,
                min_child_weight=12, subsample=0.85,
                colsample_bytree=0.70, reg_lambda=50,
                reg_alpha=4, tree_method="hist", n_jobs=6,
                random_state=seed, objective="reg:squarederror",
            ),
        )
    raise ValueError(name)


def manual_normal(row: pd.Series, method: str) -> float:
    values = np.asarray([
        math.exp(row[f"log_lag_{lag}"])
        if np.isfinite(row[f"log_lag_{lag}"]) else np.nan
        for lag in range(1, 21)
    ])
    if method == "weighted3":
        weights = np.asarray([0.60, 0.25, 0.15])
        valid = np.isfinite(values[:3])
        return float(np.sum(values[:3][valid] * weights[valid]) / weights[valid].sum())
    if method == "mean10":
        return float(np.nanmean(values[:10]))
    if method == "ewma20":
        ages = np.arange(1, 21)
        weights = np.exp(-(ages - 1) / 4.0)
        valid = np.isfinite(values)
        return float(np.sum(values[valid] * weights[valid]) / weights[valid].sum())
    if method == "trend20":
        valid = np.isfinite(values)
        if valid.sum() < 3:
            return float(np.nanmean(values[:10]))
        x = -np.arange(1, 21)[valid]
        prediction = float(np.polyval(np.polyfit(x, values[valid], 1), 0))
        recent = values[np.isfinite(values)][:10]
        lo = max(500.0, float(np.nanmin(recent) - 0.15 * np.ptp(recent)))
        hi = min(7000.0, float(np.nanmax(recent) + 0.15 * np.ptp(recent)))
        return float(np.clip(prediction, lo, hi))
    raise ValueError(method)


def build_normal_predictions(
    long: pd.DataFrame,
    official: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = add_long_history_features(long)
    features = normal_feature_columns()
    rows = []
    audits = []
    learned = [name for name in NORMAL_CANDIDATES if name.startswith(
        ("ridge", "extra", "xgb")
    )]
    for year in range(2010, 2023):
        train = history[
            history["season_start_year"].lt(year)
            & history["log_target"].notna()
            & history["observed_5"].ge(3)
        ].copy()
        test = history[history["season_start_year"].eq(year)].copy()
        test = official.loc[
            official["season_start_year"].eq(year),
            [
            "district_id", "state_name", "district_name",
            "season_start_year", TARGET, "lag_1_yield",
            ],
        ].merge(
            test.drop(columns=[
                "state_name", "district_name", TARGET,
            ], errors="ignore"),
            on=["district_id", "season_start_year"],
            how="left", validate="one_to_one",
        )
        for method in ("weighted3", "mean10", "trend20", "ewma20"):
            prediction = np.asarray([
                manual_normal(row, method) for _, row in test.iterrows()
            ])
            block = test[[
                "district_id", "state_name", "district_name",
                "season_start_year", TARGET, "lag_1_yield",
            ]].copy()
            block["normal_candidate"] = method
            block["normal_prediction"] = np.clip(prediction, 500, 7000)
            rows.append(block)
        x_train, design_columns = normal_design(train, features)
        x_test, _ = normal_design(test, features, design_columns)
        for method in learned:
            seed_predictions = []
            seeds = SEEDS if method in {"extra", "xgb1", "xgb2"} else (42,)
            for seed in seeds:
                model = normal_estimator(method, seed)
                model.fit(x_train, train["log_target"])
                seed_predictions.append(np.exp(model.predict(x_test)))
            prediction = np.mean(seed_predictions, axis=0)
            block = test[[
                "district_id", "state_name", "district_name",
                "season_start_year", TARGET, "lag_1_yield",
            ]].copy()
            block["normal_candidate"] = method
            block["normal_prediction"] = np.clip(prediction, 500, 7000)
            rows.append(block)
            audits.append({
                "test_year": year,
                "candidate": method,
                "train_rows": len(train),
                "train_year_min": int(train["season_start_year"].min()),
                "train_year_max": int(train["season_start_year"].max()),
                "features": len(features),
            })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def select_operational_normal(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    selection_rows = []
    for year in range(2010, 2023):
        prior = predictions[
            predictions["season_start_year"].between(max(2010, year - 5), year - 1)
        ]
        candidates = []
        for candidate, block in prior.groupby("normal_candidate"):
            if block["season_start_year"].nunique() == 0:
                continue
            temp = block.rename(columns={"normal_prediction": "prediction"})
            values = metric_values(temp)
            candidates.append({
                "test_year": year, "normal_candidate": candidate,
                "selection_year_min": int(block["season_start_year"].min()),
                "selection_year_max": int(block["season_start_year"].max()),
                "selection_years": int(block["season_start_year"].nunique()),
                **values, "selection_score": selection_score(values),
            })
        if candidates:
            grid = pd.DataFrame(candidates).sort_values(
                ["selection_score", "rmse", "normal_candidate"]
            )
            winner = grid.iloc[0]["normal_candidate"]
            selection_rows.extend(grid.to_dict("records"))
        else:
            winner = "ridge100"
            selection_rows.append({
                "test_year": year, "normal_candidate": winner,
                "selection_year_min": -1, "selection_year_max": -1,
                "selection_years": 0, "selection_score": math.nan,
            })
        chosen = predictions[
            predictions["season_start_year"].eq(year)
            & predictions["normal_candidate"].eq(winner)
        ].copy()
        chosen["selected_normal_candidate"] = winner
        selected_rows.append(chosen)
    selected = pd.concat(selected_rows, ignore_index=True)
    selected["actual_log_anomaly"] = np.log(
        selected[TARGET] / selected["normal_prediction"]
    )
    return selected, pd.DataFrame(selection_rows)


def encoder_fold(
    base: pd.DataFrame,
    encoder: pd.DataFrame,
    test_year: int,
    variant: str,
) -> tuple[pd.DataFrame, list[str]]:
    if variant == "none":
        frame = base.copy()
        return frame, []
    train_end = FOLD_END[test_year]
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
    feature_columns = [column for column in selected if column.startswith("enc__")]
    selected = selected.drop(columns=[
        "state_name", "district_name", "clock",
        "representation_train_end", "feature_role",
        "held_group", "encoder_variant",
    ])
    frame = base.merge(
        selected, on=["district_id", "season_start_year"],
        how="left", validate="one_to_one",
    )
    return frame, feature_columns


def base_feature_groups(base: pd.DataFrame) -> dict[str, list[str]]:
    history = finite_columns(base, HISTORY_FEATURES, 0.50)
    physical = finite_columns(
        base, list(v5lab.PHYSICAL_COMPACT) + STATIC_FEATURES, 0.45
    )
    economic = finite_columns(base, list(v5lab.ECON_COMPACT), 0.45)
    return {
        "history": history,
        "physical": list(dict.fromkeys(history + physical + economic)),
    }


def add_state_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["state_shock_actual"] = result.groupby(
        ["state_name", "season_start_year"]
    )["actual_log_anomaly"].transform("mean")
    state = (
        result[["state_name", "season_start_year", "state_shock_actual"]]
        .drop_duplicates()
        .sort_values(["state_name", "season_start_year"])
    )
    for lag in range(1, 4):
        state[f"shock_lag_{lag}"] = (
            state.groupby("state_name")["state_shock_actual"].shift(lag)
        )
    national = (
        state.groupby("season_start_year", as_index=False)["state_shock_actual"]
        .mean().rename(columns={"state_shock_actual": "national_shock"})
        .sort_values("season_start_year")
    )
    for lag in range(1, 4):
        national[f"national_shock_lag_{lag}"] = national["national_shock"].shift(lag)
    state = state.merge(
        national.drop(columns=["national_shock"]),
        on="season_start_year", how="left",
    )
    return result.merge(
        state.drop(columns=["state_shock_actual"]),
        on=["state_name", "season_start_year"],
        how="left", validate="many_to_one",
    )


def district_exposure(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for district, block in train.groupby("district_id"):
        x = block["state_shock_actual"].to_numpy(float)
        y = block["actual_log_anomaly"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 or np.var(x) < 1e-8:
            raw_beta, raw_intercept = 1.0, 0.0
        else:
            raw_beta = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
            raw_intercept = float(np.mean(y - raw_beta * x))
        n = len(x)
        rows.append({
            "district_id": district,
            "exposure_beta": float(np.clip(
                (n * raw_beta + 8.0) / (n + 8.0), 0.25, 1.75
            )),
            "exposure_intercept": float(np.clip(
                n * raw_intercept / (n + 12.0), -0.15, 0.15
            )),
            "exposure_years": n,
        })
    return pd.DataFrame(rows)


def state_feature_panel(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    columns = finite_columns(frame, feature_columns, 0.20)
    state = frame.groupby(
        ["state_name", "season_start_year"], as_index=False
    )[[TARGET, "state_shock_actual", *columns]].mean(numeric_only=True)
    shock_columns = [
        f"shock_lag_{lag}" for lag in range(1, 4)
    ] + [
        f"national_shock_lag_{lag}" for lag in range(1, 4)
    ]
    lags = frame[[
        "state_name", "season_start_year", *shock_columns
    ]].drop_duplicates(["state_name", "season_start_year"])
    return state.merge(
        lags, on=["state_name", "season_start_year"],
        how="left", validate="one_to_one",
    )


def tabular_design(
    frame: pd.DataFrame,
    features: list[str],
    columns: list[str] | None = None,
    district_dummy: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    usable = finite_columns(frame, features, 0.15)
    numeric = frame[usable].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    state = pd.get_dummies(
        frame["state_name"], prefix="state", dtype=float
    ).reset_index(drop=True)
    pieces = [numeric, state]
    if district_dummy and "district_id" in frame:
        pieces.append(pd.get_dummies(
            frame["district_id"], prefix="district", dtype=float
        ).reset_index(drop=True))
    design = pd.concat(pieces, axis=1)
    if columns is None:
        columns = design.columns.tolist()
    return design.reindex(columns=columns, fill_value=0.0), columns


def small_estimator(name: str, task: str, seed: int):
    if name.startswith("ridge"):
        alpha = float(name.replace("ridge", ""))
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(), Ridge(alpha=alpha),
        )
    if name == "extra":
        leaf = 3 if task == "state" else 18
        depth = 3 if task == "state" else 5
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesRegressor(
                n_estimators=400, max_depth=depth,
                min_samples_leaf=leaf, max_features=0.65,
                random_state=seed, n_jobs=6,
            ),
        )
    if name.startswith("xgb"):
        depth = int(name[-1])
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBRegressor(
                n_estimators=300, max_depth=depth, learning_rate=0.025,
                min_child_weight=4 if task == "state" else 20,
                subsample=0.85, colsample_bytree=0.65,
                reg_lambda=60, reg_alpha=5, tree_method="hist",
                n_jobs=6, random_state=seed,
                objective="reg:squarederror",
            ),
        )
    raise ValueError(name)


def predict_state_shock(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
) -> np.ndarray:
    x_train, columns = tabular_design(train, features)
    x_test, _ = tabular_design(test, features, columns)
    seeds = SEEDS if model_name in {"extra", "xgb1"} else (42,)
    predictions = []
    for seed in seeds:
        model = small_estimator(model_name, "state", seed)
        model.fit(x_train, train["state_shock_actual"])
        predictions.append(model.predict(x_test))
    return np.mean(predictions, axis=0)


def predict_district_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    model_name: str,
) -> np.ndarray:
    if model_name == "zero":
        return np.zeros(len(test))
    x_train, columns = tabular_design(
        train, features, district_dummy=False
    )
    x_test, _ = tabular_design(
        test, features, columns, district_dummy=False
    )
    seeds = SEEDS if model_name in {"extra", "xgb1", "xgb2"} else (42,)
    predictions = []
    for seed in seeds:
        model = small_estimator(model_name, "district", seed)
        model.fit(x_train, train[target])
        predictions.append(model.predict(x_test))
    return np.mean(predictions, axis=0)


def hierarchy_predictions(
    base: pd.DataFrame,
    encoder: pd.DataFrame,
    groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    audit_rows = []
    shock_lags = [
        f"shock_lag_{lag}" for lag in range(1, 4)
    ] + [
        f"national_shock_lag_{lag}" for lag in range(1, 4)
    ]
    for test_year in TEST_YEARS:
        train_end = FOLD_END[test_year]
        for encoder_variant in ENCODER_VARIANTS:
            fold, encoder_columns = encoder_fold(
                base, encoder, test_year, encoder_variant
            )
            train = fold[fold["season_start_year"].between(2010, train_end)].copy()
            test = fold[fold["season_start_year"].eq(test_year)].copy()
            exposures = district_exposure(train)
            train = train.merge(exposures, on="district_id", how="left")
            test = test.merge(exposures, on="district_id", how="left")
            train["district_residual"] = (
                train["actual_log_anomaly"]
                - train["exposure_intercept"]
                - train["exposure_beta"] * train["state_shock_actual"]
            )

            compact_encoder = [
                column for column in encoder_columns
                if any(token in column for token in [
                    "fused_pool_0", "future_effect_",
                    "delta_index_", "delta_abs_mean",
                    "delta_positive_fraction", "current_index_",
                ])
            ]
            state_sets = {
                "history": shock_lags,
                "physical": shock_lags + groups["physical"],
                "encoder": shock_lags + groups["physical"] + compact_encoder,
            }
            district_features = (
                groups["physical"] + compact_encoder
            )
            state_train_full = state_feature_panel(
                train, groups["physical"] + compact_encoder
            )
            state_test_full = state_feature_panel(
                test, groups["physical"] + compact_encoder
            )

            state_cache: dict[tuple[str, str], pd.DataFrame] = {}
            for state_set, state_features in state_sets.items():
                if encoder_variant == "none" and state_set == "encoder":
                    continue
                for state_model in STATE_MODELS:
                    state_train = state_train_full.copy()
                    state_test = state_test_full.copy()
                    state_test["predicted_state_shock"] = predict_state_shock(
                        state_train, state_test, state_features, state_model
                    )
                    state_cache[(state_set, state_model)] = state_test[[
                        "state_name", "season_start_year",
                        "predicted_state_shock",
                    ]]
            residual_cache = {}
            direct_cache = {}
            for residual_model in RESIDUAL_MODELS:
                residual_cache[residual_model] = predict_district_target(
                    train, test, district_features,
                    "district_residual", residual_model,
                )
                direct_cache[residual_model] = predict_district_target(
                    train, test, district_features,
                    "actual_log_anomaly", residual_model,
                )

            # Controls: direct anomaly prediction without the hierarchy.
            for residual_model, direct_prediction in direct_cache.items():
                point = np.clip(
                    test["normal_prediction"].to_numpy(float)
                    * np.exp(np.clip(direct_prediction, -0.60, 0.60)),
                    500, 7000,
                )
                block = test[[
                    "district_id", "state_name", "district_name",
                    "season_start_year", TARGET, "lag_1_yield",
                    "normal_prediction", "selected_normal_candidate",
                ]].copy()
                block["encoder_variant"] = encoder_variant
                block["state_feature_set"] = "none"
                block["state_model"] = "none"
                block["residual_model"] = residual_model
                block["architecture"] = "direct_anomaly"
                block["candidate"] = (
                    f"direct__{encoder_variant}__{residual_model}"
                )
                block["predicted_state_shock"] = 0.0
                block["exposure_beta"] = test["exposure_beta"].to_numpy(float)
                block["predicted_residual"] = direct_prediction
                block["predicted_anomaly"] = direct_prediction
                block["prediction"] = point
                prediction_rows.append(block)

            for (state_set, state_model), state_prediction in state_cache.items():
                mapped = test.merge(
                    state_prediction,
                    on=["state_name", "season_start_year"],
                    validate="many_to_one",
                )
                for residual_model, residual_prediction in residual_cache.items():
                    anomaly = (
                        mapped["exposure_intercept"].fillna(0).to_numpy(float)
                        + mapped["exposure_beta"].fillna(1).to_numpy(float)
                        * mapped["predicted_state_shock"].to_numpy(float)
                        + residual_prediction
                    )
                    point = np.clip(
                        mapped["normal_prediction"].to_numpy(float)
                        * np.exp(np.clip(anomaly, -0.60, 0.60)),
                        500, 7000,
                    )
                    block = mapped[[
                        "district_id", "state_name", "district_name",
                        "season_start_year", TARGET, "lag_1_yield",
                        "normal_prediction", "selected_normal_candidate",
                        "predicted_state_shock", "exposure_beta",
                    ]].copy()
                    block["encoder_variant"] = encoder_variant
                    block["state_feature_set"] = state_set
                    block["state_model"] = state_model
                    block["residual_model"] = residual_model
                    block["architecture"] = "hierarchy"
                    block["candidate"] = (
                        f"hier__{encoder_variant}__{state_set}"
                        f"__{state_model}__{residual_model}"
                    )
                    block["predicted_residual"] = residual_prediction
                    block["predicted_anomaly"] = anomaly
                    block["prediction"] = point
                    prediction_rows.append(block)
            audit_rows.append({
                "test_year": test_year,
                "train_end": train_end,
                "encoder_variant": encoder_variant,
                "train_rows": len(train),
                "test_rows": len(test),
                "encoder_features": len(compact_encoder),
                "district_features": len(finite_columns(
                    train, district_features, 0.15
                )),
                "state_rows": len(state_train_full),
            })
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(audit_rows)


def candidate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, block in predictions.groupby("candidate"):
        for period, years in [
            ("development", DEVELOPMENT),
            ("late", LATE),
            ("four_year", TEST_YEARS),
        ]:
            part = block[block["season_start_year"].isin(years)]
            if part["season_start_year"].nunique() != len(years):
                continue
            values = metric_values(part)
            rows.append({
                "candidate": candidate,
                "architecture": block["architecture"].iloc[0],
                "encoder_variant": block["encoder_variant"].iloc[0],
                "state_feature_set": block["state_feature_set"].iloc[0],
                "state_model": block["state_model"].iloc[0],
                "residual_model": block["residual_model"].iloc[0],
                "period": period, "rows": len(part),
                **values, "selection_score": selection_score(values),
            })
    return pd.DataFrame(rows)


def blend_with_v5(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v5 = pd.read_csv(V5_PATH)[[
        "district_id", "season_start_year", "prediction"
    ]].rename(columns={"prediction": "v5_prediction"})
    merged = predictions.merge(
        v5, on=["district_id", "season_start_year"],
        validate="many_to_one",
    )
    grid_rows = []
    weights = np.round(np.arange(0, 0.51, 0.025), 3)
    for candidate, block in merged.groupby("candidate"):
        dev = block[block["season_start_year"].isin(DEVELOPMENT)]
        for weight in weights:
            temp = dev.copy()
            temp["blend_prediction"] = (
                (1 - weight) * temp["v5_prediction"]
                + weight * temp["prediction"]
            )
            values = metric_values(temp, "blend_prediction")
            grid_rows.append({
                "candidate": candidate,
                "hierarchy_weight": float(weight),
                **{f"development_{key}": value for key, value in values.items()},
                "selection_score": selection_score(values),
            })
    grid = pd.DataFrame(grid_rows).sort_values(
        ["selection_score", "hierarchy_weight", "candidate"]
    )
    exact = grid.iloc[0]
    tolerance = 0.001 * float(exact["selection_score"])
    tied = grid[
        grid["selection_score"].le(
            float(exact["selection_score"]) + tolerance
        )
    ].sort_values(["hierarchy_weight", "selection_score"])
    regularized = tied.iloc[0]
    selected_rows = []
    metric_rows = []
    for rule, winner in [
        ("exact_development", exact),
        ("regularized_near_tie", regularized),
    ]:
        block = merged[
            merged["candidate"].eq(winner["candidate"])
        ].copy()
        weight = float(winner["hierarchy_weight"])
        block["blend_prediction"] = (
            (1 - weight) * block["v5_prediction"]
            + weight * block["prediction"]
        )
        block["blend_rule"] = rule
        block["hierarchy_weight"] = weight
        selected_rows.append(block)
        for period, years in [
            ("development", DEVELOPMENT),
            ("late", LATE),
            ("four_year", TEST_YEARS),
        ]:
            part = block[block["season_start_year"].isin(years)]
            model_values = metric_values(part, "blend_prediction")
            base_values = metric_values(part, "v5_prediction")
            metric_rows.append({
                "blend_rule": rule,
                "candidate": winner["candidate"],
                "hierarchy_weight": weight,
                "period": period,
                **{f"model_{key}": value for key, value in model_values.items()},
                **{f"v5_{key}": value for key, value in base_values.items()},
                "rmse_gain_vs_v5": base_values["rmse"] - model_values["rmse"],
            })
    return grid, pd.DataFrame(metric_rows), pd.concat(selected_rows, ignore_index=True)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    long = pd.read_parquet(LONG_YIELD)
    base = pd.read_parquet(BASE_PATH)
    if int(base["season_start_year"].max()) > 2022:
        raise RuntimeError("Post-2022 label seal broken")
    normal_predictions, normal_audit = build_normal_predictions(long, base)
    normal_predictions.to_parquet(
        ARTIFACTS / "normal_candidate_predictions.parquet", index=False
    )
    normal_audit.to_csv(ARTIFACTS / "normal_training_audit.csv", index=False)
    selected_normal, normal_grid = select_operational_normal(normal_predictions)
    selected_normal.to_parquet(
        ARTIFACTS / "normal_selected_predictions.parquet", index=False
    )
    normal_grid.to_csv(ARTIFACTS / "normal_selection_grid.csv", index=False)

    panel = base.merge(
        selected_normal[[
            "district_id", "season_start_year", "normal_prediction",
            "selected_normal_candidate", "actual_log_anomaly",
        ]],
        on=["district_id", "season_start_year"],
        validate="one_to_one",
    )
    panel = add_state_targets(panel)
    panel.to_parquet(DATA / "hierarchy_panel.parquet", index=False)
    encoder = pd.read_parquet(ENCODER_PATH)
    groups = base_feature_groups(panel)
    predictions, audit = hierarchy_predictions(panel, encoder, groups)
    predictions.to_parquet(
        ARTIFACTS / "hierarchy_candidate_predictions.parquet", index=False
    )
    audit.to_csv(ARTIFACTS / "hierarchy_training_audit.csv", index=False)
    metrics = candidate_metrics(predictions)
    metrics.to_csv(ARTIFACTS / "hierarchy_candidate_metrics.csv", index=False)
    dev = metrics[metrics["period"].eq("development")].sort_values(
        ["selection_score", "rmse"]
    )
    winner = dev.iloc[0]
    selected = predictions[
        predictions["candidate"].eq(winner["candidate"])
    ].copy()
    selected.to_parquet(
        ARTIFACTS / "hierarchy_selected_predictions.parquet", index=False
    )

    blend_grid, blend_metrics, blend_predictions = blend_with_v5(predictions)
    blend_grid.to_csv(ARTIFACTS / "v5_hierarchy_blend_grid.csv", index=False)
    blend_metrics.to_csv(
        ARTIFACTS / "v5_hierarchy_selected_metrics.csv", index=False
    )
    blend_predictions.to_parquet(
        ARTIFACTS / "v5_hierarchy_selected_predictions.parquet", index=False
    )

    normal_late = selected_normal[
        selected_normal["season_start_year"].isin(LATE)
    ].rename(columns={"normal_prediction": "prediction"})
    summary = {
        "normal_late_metrics": metric_values(normal_late),
        "hierarchy_winner": winner.to_dict(),
        "hierarchy_metrics": metrics[
            metrics["candidate"].eq(winner["candidate"])
        ].to_dict("records"),
        "v5_blend_metrics": blend_metrics.to_dict("records"),
        "candidate_count": int(predictions["candidate"].nunique()),
        "normal_candidate_count": len(NORMAL_CANDIDATES),
        "post_2022_yield_labels_read": False,
    }
    (ARTIFACTS / "hierarchy_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
