#!/usr/bin/env python3
"""Run V14 standalone anomaly, V5+outlook XGBoost, and distribution experiments."""

from __future__ import annotations

import json
import math
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
V14 = Path(__file__).resolve().parents[1]
ROOT = V14.parents[1]
sys.path.insert(0, str(ROOT))
RAPID = V14.parent
OUT = V14 / "artifacts"
MODELS = V14 / "models"
LONG_TABLE = RAPID / "v3" / "data" / "feature_table_v3_extended_history_03-05.parquet"
MODIS = RAPID / "v5" / "agent_modis_history" / "data" / "modis_strict_history.csv"
OUTLOOK = V14 / "data" / "strict_outlook_features.parquet"
V5_PRED = RAPID / "v5" / "root_cybench_lab" / "artifacts" / "v5_integration" / "predictions.csv"

from rapid_yield_forecast.v5.agent_model_lab.scripts import run_v5_model_lab as v5lab  # noqa: E402


TARGET = "yield_kg_per_ha"
TEST_YEARS = list(range(2016, 2023))
DEVELOPMENT = [2019, 2020]
LATE = [2021, 2022]
FOLD_END = {2019: 2018, 2020: 2019, 2021: 2020, 2022: 2020}
QUANTILES = np.round(np.arange(0.05, 1.0, 0.05), 2)
SEEDS = (42, 73)
NORMALS = ("weighted3", "mean5", "median5", "trend5", "ewma5")

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


@dataclass(frozen=True)
class Spec:
    family: str
    strength: float

    @property
    def name(self) -> str:
        return f"{self.family}_{str(self.strength).replace('.', 'p')}"


SPECS = (
    Spec("ridge", 100.0),
    Spec("ridge", 1000.0),
    Spec("huber", 10.0),
    Spec("extra", 3.0),
    Spec("xgb", 1.0),
    Spec("xgb", 2.0),
    Spec("state_ridge", 100.0),
)


def finite_columns(frame: pd.DataFrame, columns: list[str], minimum: float = 0.20) -> list[str]:
    result = []
    for column in dict.fromkeys(columns):
        if column not in frame:
            continue
        values = frame[column].replace([np.inf, -np.inf], np.nan)
        if values.notna().mean() >= minimum and values.nunique(dropna=True) > 1:
            result.append(column)
    return result


def lag_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"lag_{i}_yield" for i in range(1, 6)]].to_numpy(float)


def add_normals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    lags = lag_matrix(result)
    valid = np.isfinite(lags)
    weights = np.asarray([0.60, 0.25, 0.15, 0.0, 0.0])
    active_weight = valid * weights
    result["normal__weighted3"] = (
        np.nansum(lags * active_weight, axis=1)
        / np.sum(active_weight, axis=1).clip(min=1e-8)
    )
    result["normal__mean5"] = np.nanmean(lags, axis=1)
    result["normal__median5"] = np.nanmedian(lags, axis=1)
    ew = np.asarray([0.40, 0.25, 0.15, 0.12, 0.08])
    active_ew = valid * ew
    result["normal__ewma5"] = (
        np.nansum(lags * active_ew, axis=1)
        / np.sum(active_ew, axis=1).clip(min=1e-8)
    )
    trend = []
    for row in lags:
        ok = np.isfinite(row)
        if ok.sum() < 3:
            trend.append(np.nan)
            continue
        x = -np.arange(1, 6)[ok]
        y = row[ok]
        prediction = float(np.polyval(np.polyfit(x, y, 1), 0))
        lo = max(500.0, float(np.min(y) - 0.15 * np.ptp(y)))
        hi = min(7000.0, float(np.max(y) + 0.15 * np.ptp(y)))
        trend.append(float(np.clip(prediction, lo, hi)))
    result["normal__trend5"] = trend
    for name in NORMALS:
        result[f"anomaly__{name}"] = np.log(
            result[TARGET] / result[f"normal__{name}"]
        )
    return result


def modis_panel() -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(MODIS)
    raw = raw[raw["cutoff"].eq("mar05")].copy()
    numeric = [
        column for column in raw
        if column.startswith(("mod09q1_", "mod13q1_"))
        and raw[column].notna().any()
    ]
    combined = raw.groupby(
        ["district_id", "state_name", "district_name", "season_start_year"],
        as_index=False,
    )[numeric].first()
    wanted = [
        column for column in numeric
        if any(token in column for token in [
            "ndvi_last_valid_mean", "ndvi_recent48_mean_mean",
            "ndvi_season_max_mean", "ndvi_season_mean_mean",
            "ndvi_slope_per_day_mean", "evi_last_valid_mean",
            "evi_recent48_mean_mean", "evi_season_max_mean",
            "evi_season_mean_mean", "evi_slope_per_day_mean",
        ])
        and not column.endswith("stdDev")
    ]
    combined = combined.sort_values(["district_id", "season_start_year"]).reset_index(drop=True)
    transformed = []
    for column in wanted:
        past_mean = combined.groupby("district_id")[column].transform(
            lambda x: x.shift(1).expanding(min_periods=5).mean()
        )
        past_sd = combined.groupby("district_id")[column].transform(
            lambda x: x.shift(1).expanding(min_periods=5).std()
        )
        name = f"modisz__{column}"
        combined[name] = (combined[column] - past_mean) / past_sd.where(past_sd > 1e-6)
        state = f"modisstate__{column}"
        region = f"modisregion__{column}"
        dev = f"modisdev__{column}"
        combined[state] = combined.groupby(
            ["state_name", "season_start_year"]
        )[name].transform("mean")
        combined[region] = combined.groupby("season_start_year")[name].transform("mean")
        combined[dev] = combined[name] - combined[state]
        transformed.extend([name, state, region, dev])
    return combined[[
        "district_id", "season_start_year", *transformed
    ]], transformed


def load_panel() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    frame = pd.read_parquet(LONG_TABLE)
    if int(frame["season_start_year"].max()) > 2022:
        raise RuntimeError("Post-2022 label seal broken")
    modis, modis_features = modis_panel()
    frame = frame.merge(
        modis, on=["district_id", "season_start_year"],
        how="left", validate="one_to_one",
    )
    frame = add_normals(frame).copy()
    outlook = pd.read_parquet(OUTLOOK)
    outlook = outlook[outlook["clock"].eq("mar05")].drop(
        columns=["state_name", "district_name", "clock"]
    )
    physical = finite_columns(
        frame,
        list(v5lab.PHYSICAL_COMPACT) + STATIC_FEATURES,
        minimum=0.50,
    )
    economic = finite_columns(frame, list(v5lab.ECON_COMPACT), minimum=0.50)
    history = finite_columns(frame, HISTORY_FEATURES, minimum=0.50)
    groups = {
        "history": history,
        "physical": history + physical + economic,
        "modis": history + modis_features,
        "physical_modis": history + physical + economic + modis_features,
    }
    return frame.sort_values(["season_start_year", "district_id"]).reset_index(drop=True), groups, outlook


def with_outlook(
    base: pd.DataFrame,
    outlook: pd.DataFrame,
    train_end: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    selected = outlook[outlook["representation_train_end"].eq(train_end)].copy()
    feature_columns = [column for column in selected if column.startswith("outlook_")]
    frame = base.merge(
        selected.drop(columns=["feature_role", "representation_train_end"]),
        on=["district_id", "season_start_year"],
        how="left",
        validate="one_to_one",
    )
    core = [
        column for column in feature_columns
        if any(token in column for token in [
            "delta_mean", "delta_abs_mean", "positive_fraction",
            "_index_", "future_effect_mean", "future_effect_abs_mean",
        ])
        and "_summary_" not in column
    ]
    no_future = [column for column in core if "no_future" in column]
    full = [column for column in core if "full_" in column]
    effect = [column for column in core if "future_effect" in column]
    return frame, {
        "outlook_core": core,
        "outlook_no_future": no_future,
        "outlook_full": full,
        "outlook_effect": effect,
        "outlook_broad": feature_columns,
    }


def design(
    frame: pd.DataFrame,
    features: list[str],
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    x = frame[features].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    state = pd.get_dummies(frame["state_name"], prefix="state", dtype=float).reset_index(drop=True)
    x = pd.concat([x, state], axis=1)
    if columns is None:
        columns = x.columns.tolist()
    return x.reindex(columns=columns, fill_value=0.0), columns


def estimator(spec: Spec, seed: int):
    if spec.family == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(), Ridge(alpha=spec.strength),
        )
    if spec.family == "huber":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HuberRegressor(alpha=spec.strength, epsilon=1.5, max_iter=600),
        )
    if spec.family == "extra":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesRegressor(
                n_estimators=400, max_depth=int(spec.strength),
                min_samples_leaf=12, max_features=0.65,
                random_state=seed, n_jobs=6,
            ),
        )
    if spec.family == "xgb":
        depth = int(spec.strength)
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBRegressor(
                n_estimators=350, max_depth=depth, learning_rate=0.025,
                min_child_weight=20 if depth == 1 else 25,
                subsample=0.85, colsample_bytree=0.65,
                reg_lambda=50.0, reg_alpha=5.0,
                objective="reg:squarederror", tree_method="hist",
                n_jobs=6, random_state=seed,
            ),
        )
    raise ValueError(spec.family)


def district_exposure(train: pd.DataFrame, anomaly_column: str) -> pd.DataFrame:
    work = train[["district_id", "state_name", "season_start_year", anomaly_column]].copy()
    work["state_anomaly"] = work.groupby(
        ["state_name", "season_start_year"]
    )[anomaly_column].transform("mean")
    rows = []
    for district, group in work.groupby("district_id"):
        x = group["state_anomaly"].to_numpy(float)
        y = group[anomaly_column].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 or np.var(x) < 1e-8:
            beta_raw, intercept_raw = 1.0, 0.0
        else:
            beta_raw = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
            intercept_raw = float(np.mean(y - beta_raw * x))
        n = len(x)
        beta = (n * beta_raw + 5.0) / (n + 5.0)
        intercept = n * intercept_raw / (n + 10.0)
        rows.append({
            "district_id": district,
            "exposure_beta": float(np.clip(beta, 0.25, 1.75)),
            "exposure_intercept": float(np.clip(intercept, -0.20, 0.20)),
        })
    return pd.DataFrame(rows)


def state_ridge_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    anomaly_column: str,
    alpha: float,
) -> np.ndarray:
    columns = finite_columns(train, features, minimum=0.40)
    train_state = train.groupby(
        ["state_name", "season_start_year"], as_index=False
    )[[anomaly_column, *columns]].mean(numeric_only=True)
    test_state = test.groupby(
        ["state_name", "season_start_year"], as_index=False
    )[columns].mean(numeric_only=True)
    x_train, design_columns = design(train_state, columns)
    x_test, _ = design(test_state, columns, design_columns)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
    )
    model.fit(x_train, train_state[anomaly_column])
    test_state["state_prediction"] = model.predict(x_test)
    mapped = test[["district_id", "state_name", "season_start_year"]].merge(
        test_state[["state_name", "season_start_year", "state_prediction"]],
        on=["state_name", "season_start_year"], validate="many_to_one",
    )
    exposure = district_exposure(train, anomaly_column)
    mapped = mapped.merge(exposure, on="district_id", how="left")
    return (
        mapped["exposure_intercept"].fillna(0).to_numpy()
        + mapped["exposure_beta"].fillna(1).to_numpy()
        * mapped["state_prediction"].to_numpy()
    )


def fit_anomaly(
    spec: Spec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    anomaly_column: str,
) -> np.ndarray:
    usable = finite_columns(train, features, minimum=0.25)
    if spec.family == "state_ridge":
        return state_ridge_predict(
            train, test, usable, anomaly_column, spec.strength
        )
    x_train, columns = design(train, usable)
    x_test, _ = design(test, usable, columns)
    seeds = SEEDS if spec.family in {"extra", "xgb"} else (SEEDS[0],)
    predictions = []
    for seed in seeds:
        model = estimator(spec, seed)
        model.fit(x_train, train[anomaly_column])
        predictions.append(model.predict(x_test))
    return np.mean(predictions, axis=0)


def standalone_predictions(
    base: pd.DataFrame,
    base_groups: dict[str, list[str]],
    outlook: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audits = []
    no_outlook_sets = {
        "history": base_groups["history"],
        "physical": base_groups["physical"],
        "modis": base_groups["modis"],
        "physical_modis": base_groups["physical_modis"],
    }
    for year in TEST_YEARS:
        test_base = base[base["season_start_year"].eq(year)]
        train_base = base[base["season_start_year"].lt(year)]
        for normal in NORMALS:
            normal_column = f"normal__{normal}"
            anomaly_column = f"anomaly__{normal}"
            valid_train = (
                train_base[normal_column].notna()
                & train_base[anomaly_column].replace([np.inf, -np.inf], np.nan).notna()
            )
            train = train_base[valid_train].copy()
            test = test_base[test_base[normal_column].notna()].copy()
            point = np.clip(test[normal_column].to_numpy(float), 500, 7000)
            part = test[[
                "district_id", "state_name", "district_name",
                "season_start_year", TARGET, "lag_1_yield",
            ]].copy()
            part["normal"] = normal
            part["feature_set"] = "none"
            part["model"] = "zero_anomaly"
            part["candidate"] = f"{normal}__none__zero_anomaly"
            part["normal_prediction"] = test[normal_column].to_numpy(float)
            part["predicted_anomaly"] = 0.0
            part["prediction"] = point
            rows.append(part)
            for feature_name, features in no_outlook_sets.items():
                for spec in SPECS:
                    prediction_anomaly = fit_anomaly(
                        spec, train, test, features, anomaly_column
                    )
                    prediction = np.clip(
                        test[normal_column].to_numpy(float)
                        * np.exp(np.clip(prediction_anomaly, -0.65, 0.65)),
                        500, 7000,
                    )
                    part = test[[
                        "district_id", "state_name", "district_name",
                        "season_start_year", TARGET, "lag_1_yield",
                    ]].copy()
                    part["normal"] = normal
                    part["feature_set"] = feature_name
                    part["model"] = spec.name
                    part["candidate"] = f"{normal}__{feature_name}__{spec.name}"
                    part["normal_prediction"] = test[normal_column].to_numpy(float)
                    part["predicted_anomaly"] = prediction_anomaly
                    part["prediction"] = prediction
                    rows.append(part)
                    audits.append({
                        "test_year": year, "normal": normal,
                        "feature_set": feature_name, "model": spec.name,
                        "train_rows": len(train), "test_rows": len(test),
                        "train_year_max": int(train["season_start_year"].max()),
                        "features": len(finite_columns(train, features, 0.25)),
                    })
        if year < 2019:
            continue
        train_end = FOLD_END[year]
        fold, outlook_groups = with_outlook(base, outlook, train_end)
        train_base = fold[
            fold["season_start_year"].between(2017, train_end)
        ].copy()
        test_base = fold[fold["season_start_year"].eq(year)].copy()
        outlook_sets = {
            "outlook_only": base_groups["history"] + outlook_groups["outlook_core"],
            "physical_modis_outlook": (
                base_groups["physical_modis"] + outlook_groups["outlook_core"]
            ),
            "physical_modis_outlook_broad": (
                base_groups["physical_modis"] + outlook_groups["outlook_broad"]
            ),
        }
        for normal in NORMALS:
            normal_column = f"normal__{normal}"
            anomaly_column = f"anomaly__{normal}"
            train = train_base[
                train_base[normal_column].notna()
                & train_base[anomaly_column].replace([np.inf, -np.inf], np.nan).notna()
            ].copy()
            test = test_base[test_base[normal_column].notna()].copy()
            for feature_name, features in outlook_sets.items():
                for spec in SPECS:
                    prediction_anomaly = fit_anomaly(
                        spec, train, test, features, anomaly_column
                    )
                    prediction = np.clip(
                        test[normal_column].to_numpy(float)
                        * np.exp(np.clip(prediction_anomaly, -0.65, 0.65)),
                        500, 7000,
                    )
                    part = test[[
                        "district_id", "state_name", "district_name",
                        "season_start_year", TARGET, "lag_1_yield",
                    ]].copy()
                    part["normal"] = normal
                    part["feature_set"] = feature_name
                    part["model"] = spec.name
                    part["candidate"] = f"{normal}__{feature_name}__{spec.name}"
                    part["normal_prediction"] = test[normal_column].to_numpy(float)
                    part["predicted_anomaly"] = prediction_anomaly
                    part["prediction"] = prediction
                    rows.append(part)
                    audits.append({
                        "test_year": year, "normal": normal,
                        "feature_set": feature_name, "model": spec.name,
                        "train_rows": len(train), "test_rows": len(test),
                        "train_year_max": int(train["season_start_year"].max()),
                        "features": len(finite_columns(train, features, 0.25)),
                    })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def metric_values(frame: pd.DataFrame, prediction: str = "prediction") -> dict[str, float]:
    error = frame[prediction].to_numpy(float) - frame[TARGET].to_numpy(float)
    state_rmse = (
        frame.assign(se=error ** 2).groupby("state_name")["se"].mean().mean() ** 0.5
    )
    year_rmse = [
        mean_squared_error(part[TARGET], part[prediction]) ** 0.5
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


def candidate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, block in predictions.groupby("candidate"):
        for period, years in [
            ("rolling_2016_2022", TEST_YEARS),
            ("development", DEVELOPMENT),
            ("late", LATE),
        ]:
            part = block[block["season_start_year"].isin(years)]
            if part["season_start_year"].nunique() != len(years):
                continue
            rows.append({
                "candidate": candidate, "period": period,
                "normal": block["normal"].iloc[0],
                "feature_set": block["feature_set"].iloc[0],
                "model": block["model"].iloc[0],
                "rows": len(part), **metric_values(part),
            })
    return pd.DataFrame(rows)


def selection_score(row: pd.Series) -> float:
    return (
        0.50 * row["rmse"]
        + 0.25 * row["equal_state_rmse"]
        + 0.25 * row["mean_year_rmse"]
    )


def select_standalone(metrics: pd.DataFrame) -> pd.DataFrame:
    dev = metrics[metrics["period"].eq("development")].copy()
    dev["selection_score"] = dev.apply(selection_score, axis=1)
    long_only = dev[~dev["feature_set"].str.contains("outlook")]
    winners = []
    for category, frame in [("long_history", long_only), ("all_candidates", dev)]:
        row = frame.sort_values(["selection_score", "rmse"]).iloc[0].to_dict()
        row["selection_category"] = category
        winners.append(row)
    return pd.DataFrame(winners)


def xgb_residual_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    depth: int,
) -> np.ndarray:
    usable = finite_columns(train, features, minimum=0.25)
    x_train, columns = design(train, usable)
    x_test, _ = design(test, usable, columns)
    target = train[TARGET].to_numpy(float) - train["baseline_weighted_recent"].to_numpy(float)
    predictions = []
    for seed in SEEDS:
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBRegressor(
                n_estimators=350, max_depth=depth, learning_rate=0.025,
                min_child_weight=20 if depth == 1 else 25,
                subsample=0.85, colsample_bytree=0.65,
                reg_lambda=50.0, reg_alpha=5.0,
                objective="reg:squarederror", tree_method="hist",
                n_jobs=6, random_state=seed,
            ),
        )
        model.fit(x_train, target)
        predictions.append(model.predict(x_test))
    residual = np.mean(predictions, axis=0)
    return np.clip(
        test["baseline_weighted_recent"].to_numpy(float) + residual,
        500, 7000,
    )


def matched_xgb_predictions(
    base: pd.DataFrame,
    groups: dict[str, list[str]],
    outlook: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    base_features = groups["physical"]
    for year in DEVELOPMENT + LATE:
        train_end = FOLD_END[year]
        fold, outlook_groups = with_outlook(base, outlook, train_end)
        train = fold[fold["season_start_year"].between(2017, train_end)].copy()
        test = fold[fold["season_start_year"].eq(year)].copy()
        feature_sets = {
            "base": base_features,
            "base_no_future": base_features + outlook_groups["outlook_no_future"],
            "base_full": base_features + outlook_groups["outlook_full"],
            "base_future_effect": base_features + outlook_groups["outlook_effect"],
            "base_full_broad": base_features + outlook_groups["outlook_broad"],
        }
        for depth in (1, 2):
            for feature_set, features in feature_sets.items():
                prediction = xgb_residual_predict(train, test, features, depth)
                part = test[[
                    "district_id", "state_name", "district_name",
                    "season_start_year", TARGET, "lag_1_yield",
                ]].copy()
                part["feature_set"] = feature_set
                part["depth"] = depth
                part["candidate"] = f"xgb_d{depth}__{feature_set}"
                part["prediction"] = prediction
                rows.append(part)
    return pd.concat(rows, ignore_index=True)


def grouped_rmse_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    draws: int = 5000,
) -> dict[str, float]:
    groups = [
        part.index.to_numpy()
        for _, part in frame.groupby(["state_name", "season_start_year"])
    ]
    rng = np.random.default_rng(20260729)
    gains = []
    for _ in range(draws):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        actual = frame.loc[idx, TARGET].to_numpy(float)
        base_rmse = np.sqrt(np.mean((frame.loc[idx, baseline].to_numpy(float) - actual) ** 2))
        candidate_rmse = np.sqrt(np.mean((frame.loc[idx, candidate].to_numpy(float) - actual) ** 2))
        gains.append(base_rmse - candidate_rmse)
    values = np.asarray(gains)
    return {
        "bootstrap_mean_gain": float(values.mean()),
        "bootstrap_p025": float(np.quantile(values, 0.025)),
        "bootstrap_p975": float(np.quantile(values, 0.975)),
        "bootstrap_probability_positive": float(np.mean(values > 0)),
        "bootstrap_draws": draws,
    }


def select_xgb_and_blend(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v5 = pd.read_csv(V5_PRED)[[
        "district_id", "season_start_year", "prediction"
    ]].rename(columns={"prediction": "v5_prediction"})
    merged = predictions.merge(
        v5, on=["district_id", "season_start_year"],
        how="inner", validate="many_to_one",
    )
    grid = []
    for candidate, block in merged.groupby("candidate"):
        dev = block[block["season_start_year"].isin(DEVELOPMENT)]
        for weight in np.linspace(0, 0.5, 11):
            temp = dev.copy()
            temp["blend_prediction"] = (
                (1 - weight) * temp["v5_prediction"]
                + weight * temp["prediction"]
            )
            values = metric_values(temp, "blend_prediction")
            grid.append({
                "candidate": candidate,
                "feature_set": block["feature_set"].iloc[0],
                "depth": int(block["depth"].iloc[0]),
                "outlook_weight": float(weight),
                **{f"development_{key}": value for key, value in values.items()},
                "selection_score": (
                    0.50 * values["rmse"]
                    + 0.25 * values["equal_state_rmse"]
                    + 0.25 * values["mean_year_rmse"]
                ),
            })
    grid_frame = pd.DataFrame(grid).sort_values(["selection_score", "outlook_weight"])
    winner = grid_frame.iloc[0]
    selected = merged[merged["candidate"].eq(winner["candidate"])].copy()
    selected["blend_prediction"] = (
        (1 - winner["outlook_weight"]) * selected["v5_prediction"]
        + winner["outlook_weight"] * selected["prediction"]
    )
    metrics = []
    for period, years in [("development", DEVELOPMENT), ("late", LATE)]:
        part = selected[selected["season_start_year"].isin(years)]
        candidate_values = metric_values(part, "blend_prediction")
        base_values = metric_values(part, "v5_prediction")
        metrics.append({
            **winner.to_dict(), "period": period,
            **{f"model_{key}": value for key, value in candidate_values.items()},
            **{f"v5_{key}": value for key, value in base_values.items()},
            **(
                grouped_rmse_bootstrap(
                    part.reset_index(drop=True), "blend_prediction", "v5_prediction"
                )
                if period == "late" else {}
            ),
        })
    return grid_frame, pd.DataFrame(metrics), selected


def weighted_quantile(values: np.ndarray, quantiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return np.interp(quantiles, cumulative, values, left=values[0], right=values[-1])


def residual_pool(
    calibration: pd.DataFrame,
    row: pd.Series,
    method: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    residual = calibration[TARGET].to_numpy(float) - calibration["cal_prediction"].to_numpy(float)
    years = calibration["season_start_year"].to_numpy(int)
    global_weights = np.ones(len(calibration), float)
    if "equal_year" in method:
        counts = pd.Series(years).value_counts().to_dict()
        global_weights = np.asarray([1.0 / counts[year] for year in years])
    target_scale = max(float(row.get("yield_recent_std", np.nan)), 0.07 * float(row["normal_prediction"]), 150.0)
    if "scaled" in method:
        cal_scale = np.maximum.reduce([
            calibration["yield_recent_std"].fillna(0).to_numpy(float),
            0.07 * calibration["normal_prediction"].to_numpy(float),
            np.full(len(calibration), 150.0),
        ])
        residual = residual / cal_scale
    state_mask = calibration["state_name"].eq(row["state_name"]).to_numpy()
    if "state" not in method or state_mask.sum() < 5:
        return residual, global_weights, target_scale
    state_residual = residual[state_mask]
    state_weights = global_weights[state_mask]
    shrink = state_mask.sum() / (state_mask.sum() + 50.0)
    global_weights = global_weights / global_weights.sum() * (1 - shrink)
    state_weights = state_weights / state_weights.sum() * shrink
    return (
        np.concatenate([residual, state_residual]),
        np.concatenate([global_weights, state_weights]),
        target_scale,
    )


def distribution_predictions(
    selected_predictions: pd.DataFrame,
    base: pd.DataFrame,
    candidate: str,
) -> pd.DataFrame:
    chosen = selected_predictions[selected_predictions["candidate"].eq(candidate)].copy()
    normal = chosen["normal"].iloc[0]
    base_rows = base[base["season_start_year"].isin(TEST_YEARS)].copy()
    base_rows = base_rows[[
        "district_id", "state_name", "district_name", "season_start_year",
        TARGET, "lag_1_yield", "yield_recent_std", f"normal__{normal}",
    ]].rename(columns={f"normal__{normal}": "normal_prediction"})
    prediction_map = chosen[["district_id", "season_start_year", "prediction"]]
    base_rows = base_rows.merge(
        prediction_map,
        on=["district_id", "season_start_year"],
        how="left", validate="one_to_one",
    )
    base_rows["cal_prediction"] = base_rows["prediction"].fillna(
        base_rows["normal_prediction"]
    )
    rows = []
    methods = ("global", "equal_year", "state_equal_year", "scaled_state_equal_year")
    for year in DEVELOPMENT + LATE:
        calibration = base_rows[base_rows["season_start_year"].between(2016, year - 1)].copy()
        target = base_rows[base_rows["season_start_year"].eq(year)].copy()
        for method in methods:
            for _, row in target.iterrows():
                residual, weights, scale = residual_pool(calibration, row, method)
                if "scaled" in method:
                    residual = residual * scale
                quantile_residual = weighted_quantile(residual, QUANTILES, weights)
                point = float(row["cal_prediction"])
                quantile_yield = np.clip(point + quantile_residual, 500, 7000)
                probability_rise = float(
                    np.sum(weights[(point + residual) > row["lag_1_yield"]]) / np.sum(weights)
                )
                probability_severe = float(
                    np.sum(weights[(point + residual) <= 0.90 * row["lag_1_yield"]]) / np.sum(weights)
                )
                output = {
                    "district_id": row["district_id"],
                    "state_name": row["state_name"],
                    "district_name": row["district_name"],
                    "season_start_year": year,
                    TARGET: row[TARGET],
                    "lag_1_yield": row["lag_1_yield"],
                    "method": method,
                    "candidate": candidate,
                    "point_prediction": point,
                    "distribution_mean": float(np.average(point + residual, weights=weights)),
                    "distribution_sd": float(np.sqrt(np.average(
                        (point + residual - np.average(point + residual, weights=weights)) ** 2,
                        weights=weights,
                    ))),
                    "probability_rise": probability_rise,
                    "probability_severe_drop": probability_severe,
                    "calibration_rows": len(calibration),
                    "calibration_years": calibration["season_start_year"].nunique(),
                }
                for q, value in zip(QUANTILES, quantile_yield):
                    output[f"q{int(q * 100):02d}"] = float(value)
                rows.append(output)
    return pd.DataFrame(rows)


def distribution_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, period), part in predictions.assign(
        period=np.where(
            predictions["season_start_year"].isin(DEVELOPMENT), "development", "late"
        )
    ).groupby(["method", "period"]):
        losses = []
        actual = part[TARGET].to_numpy(float)
        for q in QUANTILES:
            predicted = part[f"q{int(q * 100):02d}"].to_numpy(float)
            losses.append((q - (actual < predicted).astype(float)) * (actual - predicted))
        pinball = float(np.mean(np.stack(losses, axis=1)))
        rows.append({
            "method": method, "period": period, "rows": len(part),
            "mean_pinball_loss": pinball,
            "coverage_50": float(np.mean(
                (actual >= part["q25"]) & (actual <= part["q75"])
            )),
            "width_50": float(np.mean(part["q75"] - part["q25"])),
            "coverage_80": float(np.mean(
                (actual >= part["q10"]) & (actual <= part["q90"])
            )),
            "width_80": float(np.mean(part["q90"] - part["q10"])),
            "coverage_90": float(np.mean(
                (actual >= part["q05"]) & (actual <= part["q95"])
            )),
            "width_90": float(np.mean(part["q95"] - part["q05"])),
        })
    return pd.DataFrame(rows)


def year_and_state_metrics(
    frame: pd.DataFrame,
    candidate: str,
    comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    selected = frame[frame["candidate"].eq(candidate)].copy()
    if comparison is not None:
        selected = selected.merge(
            comparison[["district_id", "season_start_year", "prediction"]].rename(
                columns={"prediction": "comparison_prediction"}
            ),
            on=["district_id", "season_start_year"], validate="one_to_one",
        )
    rows = []
    for keys, part in selected.groupby(["season_start_year", "state_name"]):
        year, state = keys
        row = {
            "candidate": candidate, "season_start_year": year,
            "state_name": state, **metric_values(part),
        }
        if comparison is not None:
            row.update({
                f"comparison_{key}": value
                for key, value in metric_values(part, "comparison_prediction").items()
            })
        rows.append(row)
    return pd.DataFrame(rows)


def fit_deployment_model(
    selected: pd.Series,
    base: pd.DataFrame,
    groups: dict[str, list[str]],
    outlook: pd.DataFrame,
) -> dict[str, object]:
    candidate = str(selected["candidate"])
    normal, feature_set, model_name = candidate.split("__", 2)
    if "outlook" in feature_set:
        frame, outlook_groups = with_outlook(base, outlook, 2020)
        feature_sets = {
            "outlook_only": groups["history"] + outlook_groups["outlook_core"],
            "physical_modis_outlook": groups["physical_modis"] + outlook_groups["outlook_core"],
            "physical_modis_outlook_broad": groups["physical_modis"] + outlook_groups["outlook_broad"],
        }
        frame = frame[frame["season_start_year"].between(2017, 2022)].copy()
    else:
        frame = base.copy()
        feature_sets = groups
    features = feature_sets[feature_set]
    anomaly = f"anomaly__{normal}"
    frame = frame[
        frame[f"normal__{normal}"].notna()
        & frame[anomaly].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    if model_name == "zero_anomaly":
        return {"candidate": candidate, "model_path": None}
    family, strength_text = model_name.rsplit("_", 1)
    strength = float(strength_text.replace("p", "."))
    spec = Spec(family, strength)
    if family == "state_ridge":
        return {
            "candidate": candidate,
            "model_path": None,
            "note": "state_ridge is reproducibly refit from the training panel",
        }
    usable = finite_columns(frame, features, 0.25)
    x, columns = design(frame, usable)
    model = estimator(spec, 42)
    model.fit(x, frame[anomaly])
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / "standalone_anomaly_model.joblib"
    joblib.dump({
        "model": model,
        "candidate": candidate,
        "normal": normal,
        "feature_set": feature_set,
        "features": usable,
        "design_columns": columns,
        "fit_through": 2022,
        "score_claimed": False,
    }, path)
    return {"candidate": candidate, "model_path": str(path.relative_to(V14))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    base, groups, outlook = load_panel()
    predictions, fit_audit = standalone_predictions(base, groups, outlook)
    predictions.to_parquet(OUT / "standalone_candidate_predictions.parquet", index=False)
    fit_audit.to_csv(OUT / "standalone_fit_audit.csv", index=False)
    metrics = candidate_metrics(predictions)
    metrics.to_csv(OUT / "standalone_candidate_metrics.csv", index=False)
    selection = select_standalone(metrics)
    selection.to_csv(OUT / "standalone_selection.csv", index=False)

    selected_rows = []
    state_year = []
    for _, winner in selection.iterrows():
        candidate = winner["candidate"]
        chosen = predictions[predictions["candidate"].eq(candidate)].copy()
        chosen["selection_category"] = winner["selection_category"]
        selected_rows.append(chosen)
        state_year.append(year_and_state_metrics(predictions, candidate))
    selected_predictions = pd.concat(selected_rows, ignore_index=True)
    selected_predictions.to_parquet(OUT / "standalone_selected_predictions.parquet", index=False)
    pd.concat(state_year, ignore_index=True).to_csv(
        OUT / "standalone_selected_state_year.csv", index=False
    )

    xgb_predictions = matched_xgb_predictions(base, groups, outlook)
    xgb_predictions.to_parquet(OUT / "xgb_outlook_predictions.parquet", index=False)
    xgb_grid, xgb_metrics, xgb_selected = select_xgb_and_blend(xgb_predictions)
    xgb_grid.to_csv(OUT / "xgb_v5_blend_grid.csv", index=False)
    xgb_metrics.to_csv(OUT / "xgb_v5_selected_metrics.csv", index=False)
    xgb_selected.to_parquet(OUT / "xgb_v5_selected_predictions.parquet", index=False)

    all_winner = selection[
        selection["selection_category"].eq("all_candidates")
    ].iloc[0]
    distribution = distribution_predictions(
        predictions, base, str(all_winner["candidate"])
    )
    distribution.to_parquet(OUT / "distribution_candidate_predictions.parquet", index=False)
    dist_metrics = distribution_metrics(distribution)
    dist_metrics.to_csv(OUT / "distribution_metrics.csv", index=False)
    selected_method = (
        dist_metrics[dist_metrics["period"].eq("development")]
        .sort_values("mean_pinball_loss").iloc[0]["method"]
    )
    distribution_selected = distribution[distribution["method"].eq(selected_method)].copy()
    distribution_selected.to_parquet(
        OUT / "distribution_selected_predictions.parquet", index=False
    )

    v5 = pd.read_csv(V5_PRED)
    late_v5 = v5[v5["season_start_year"].isin(LATE)][[
        "district_id", "state_name", "season_start_year", TARGET,
        "lag_1_yield", "prediction",
    ]].rename(columns={"prediction": "v5_prediction"})
    late_standalone = predictions[
        predictions["candidate"].eq(all_winner["candidate"])
        & predictions["season_start_year"].isin(LATE)
    ].merge(
        late_v5[["district_id", "season_start_year", "v5_prediction"]],
        on=["district_id", "season_start_year"], validate="one_to_one",
    )
    standalone_boot = grouped_rmse_bootstrap(
        late_standalone.reset_index(drop=True), "prediction", "v5_prediction"
    )
    standalone_late = metric_values(late_standalone)
    v5_late = metric_values(
        late_standalone.rename(columns={
            "prediction": "standalone_prediction",
            "v5_prediction": "prediction",
        })
    )
    years_improved = sum(
        mean_squared_error(part[TARGET], part["prediction"]) ** 0.5
        < mean_squared_error(part[TARGET], part["v5_prediction"]) ** 0.5
        for _, part in late_standalone.groupby("season_start_year")
    )
    cells_improved = sum(
        mean_squared_error(part[TARGET], part["prediction"]) ** 0.5
        < mean_squared_error(part[TARGET], part["v5_prediction"]) ** 0.5
        for _, part in late_standalone.groupby(["state_name", "season_start_year"])
    )
    strict = bool(
        standalone_late["rmse"] < v5_late["rmse"]
        and years_improved == 2 and cells_improved >= 4
        and standalone_boot["bootstrap_p025"] > 0
    )
    close = standalone_late["rmse"] <= v5_late["rmse"] + max(10.0, 0.03 * v5_late["rmse"])
    status = "production" if strict else "shadow" if close else "reject"

    xgb_late = xgb_metrics[xgb_metrics["period"].eq("late")].iloc[0]
    xgb_strict = bool(
        xgb_late["model_rmse"] < xgb_late["v5_rmse"]
        and xgb_late.get("bootstrap_p025", -np.inf) > 0
    )
    xgb_close = xgb_late["model_rmse"] <= xgb_late["v5_rmse"] + max(
        10.0, 0.03 * xgb_late["v5_rmse"]
    )
    xgb_status = "production" if xgb_strict else "shadow" if xgb_close else "reject"

    deployment = fit_deployment_model(all_winner, base, groups, outlook)
    final = {
        "standalone": {
            "candidate": all_winner["candidate"],
            "selection_category": "all_candidates",
            "development_selection_score": all_winner["selection_score"],
            "late_metrics": standalone_late,
            "v5_late_metrics": v5_late,
            "late_years_improved": years_improved,
            "late_state_year_cells_improved": cells_improved,
            "bootstrap": standalone_boot,
            "status": status,
        },
        "xgb_outlook_v5": {
            "candidate": xgb_late["candidate"],
            "outlook_weight": xgb_late["outlook_weight"],
            "late_model_rmse": xgb_late["model_rmse"],
            "late_v5_rmse": xgb_late["v5_rmse"],
            "bootstrap_p025": xgb_late.get("bootstrap_p025"),
            "status": xgb_status,
        },
        "distribution": {
            "point_candidate": all_winner["candidate"],
            "selected_method": selected_method,
            "metrics": dist_metrics[
                dist_metrics["method"].eq(selected_method)
            ].to_dict("records"),
        },
        "deployment": deployment,
        "selection_years": DEVELOPMENT,
        "late_confirmation_years": LATE,
        "rolling_years_reported": TEST_YEARS,
        "post_2022_yield_labels_read": False,
    }
    (OUT / "final_summary.json").write_text(json.dumps(final, indent=2, default=str))
    print(json.dumps(final, indent=2, default=str))


if __name__ == "__main__":
    main()
