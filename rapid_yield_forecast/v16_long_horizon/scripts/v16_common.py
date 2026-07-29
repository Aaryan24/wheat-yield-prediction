#!/usr/bin/env python3
"""Shared fitting and scoring utilities for V16.

Two things here are deliberate departures from V15:

1.  `rolling_origin_predict` trains on every season strictly before the target
    season and walks forward one season at a time.  V15 could only afford four
    such folds; on the long panel there are nineteen.

2.  `noise_floor` measures how much a score moves under changes that carry no
    information at all -- reordering the same feature columns, which perturbs
    XGBoost's `colsample_bytree` draw.  Any reported gain smaller than this is
    not resolved by the data, and V15 had gains in exactly that range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

TARGET = "yield_kg_per_ha"
BASELINE = "baseline_weighted_recent"
SEEDS = (42, 73)
CLIP = (500.0, 7000.0)


def design(frame: pd.DataFrame, features: list[str],
           columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Numeric features plus state one-hot, with a stable column order."""
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    state = pd.get_dummies(frame["state_name"], prefix="state", dtype=float)
    x = pd.concat([x.reset_index(drop=True), state.reset_index(drop=True)], axis=1)
    if columns is None:
        columns = list(x.columns)
    else:
        for column in columns:
            if column not in x:
                x[column] = np.nan
        x = x[columns]
    return x, columns


def finite_columns(frame: pd.DataFrame, features: list[str],
                   minimum: float = 0.25) -> list[str]:
    """Drop features that are almost entirely missing in the training fold."""
    keep = []
    for feature in features:
        if feature not in frame:
            continue
        share = pd.to_numeric(frame[feature], errors="coerce").notna().mean()
        if share >= minimum:
            keep.append(feature)
    return keep


def xgb_residual_predict(train: pd.DataFrame, test: pd.DataFrame,
                         features: list[str], depth: int = 2,
                         seeds: tuple[int, ...] = SEEDS) -> np.ndarray:
    """Predict yield as baseline + learned residual, averaged over seeds.

    Same hyper-parameters as the locked V14/V15 models so that any difference
    in score comes from the data and features, not from retuning.
    """
    usable = finite_columns(train, features)
    x_train, columns = design(train, usable)
    x_test, _ = design(test, usable, columns)
    target = (train[TARGET].to_numpy(float)
              - train[BASELINE].to_numpy(float))
    predictions = []
    for seed in seeds:
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBRegressor(
                n_estimators=350, max_depth=depth, learning_rate=0.025,
                min_child_weight=20 if depth == 1 else 25,
                subsample=0.85, colsample_bytree=0.65,
                reg_lambda=50.0, reg_alpha=5.0,
                objective="reg:squarederror", tree_method="hist",
                n_jobs=6, random_state=seed, verbosity=0,
            ),
        )
        model.fit(x_train, target)
        predictions.append(model.predict(x_test))
    residual = np.mean(predictions, axis=0)
    return np.clip(test[BASELINE].to_numpy(float) + residual, *CLIP)


def rolling_origin_predict(panel: pd.DataFrame, features: list[str],
                           test_years: list[int], depth: int = 2,
                           min_train_years: int = 4,
                           seeds: tuple[int, ...] = SEEDS,
                           feature_order: list[str] | None = None) -> pd.DataFrame:
    """Walk forward one season at a time; train only on strictly earlier seasons."""
    order = feature_order if feature_order is not None else features
    rows = []
    for year in test_years:
        train = panel[panel.season_start_year < year]
        test = panel[panel.season_start_year.eq(year)]
        if test.empty or train.season_start_year.nunique() < min_train_years:
            continue
        prediction = xgb_residual_predict(train, test, order, depth, seeds)
        block = test[["district_id", "state_name", "season_start_year",
                      TARGET, "lag_1_yield", BASELINE]].copy()
        block["prediction"] = prediction
        block["train_years"] = train.season_start_year.nunique()
        block["train_rows"] = len(train)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def metrics(frame: pd.DataFrame, prediction: str = "prediction") -> dict[str, float]:
    error = frame[prediction].to_numpy(float) - frame[TARGET].to_numpy(float)
    per_state = (frame.assign(_e=error ** 2)
                 .groupby("state_name")["_e"].mean().pow(0.5))
    per_year = (frame.assign(_e=error ** 2)
                .groupby("season_start_year")["_e"].mean().pow(0.5))
    rose = frame[TARGET].to_numpy(float) > frame["lag_1_yield"].to_numpy(float)
    called = frame[prediction].to_numpy(float) > frame["lag_1_yield"].to_numpy(float)
    return {
        "rows": int(len(frame)),
        "years": int(frame.season_start_year.nunique()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "equal_state_rmse": float(per_state.mean()),
        "mean_year_rmse": float(per_year.mean()),
        "max_year_rmse": float(per_year.max()),
        "direction_accuracy": float(np.mean(rose == called)),
    }


def noise_floor(panel: pd.DataFrame, features: list[str], test_years: list[int],
                depth: int = 2, draws: int = 12, seed: int = 0) -> dict[str, float]:
    """Spread of RMSE under pure column reordering -- the measurement noise.

    Reordering columns changes nothing about the information available, but it
    changes which columns `colsample_bytree` draws together.  The resulting
    spread is a floor below which no reported gain is meaningful.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for draw in range(draws):
        order = list(rng.permutation(features)) if draw else list(features)
        prediction = rolling_origin_predict(
            panel, features, test_years, depth, feature_order=order)
        scores.append(metrics(prediction)["rmse"])
    scores = np.array(scores)
    return {
        "as_ordered_rmse": float(scores[0]),
        "permuted_mean": float(scores[1:].mean()),
        "permuted_sd": float(scores[1:].std(ddof=1)),
        "permuted_min": float(scores[1:].min()),
        "permuted_max": float(scores[1:].max()),
        "draws": int(draws),
    }


def year_block_bootstrap(frame: pd.DataFrame, candidate: str, baseline: str,
                         draws: int = 10000, seed: int = 0) -> dict[str, float]:
    """Resample whole seasons, because districts fail together within a season.

    Treating 2,668 district-seasons as independent badly overstates precision;
    the honest unit is the season.
    """
    rng = np.random.default_rng(seed)
    years = frame.season_start_year.unique()
    per_year = {
        int(y): (
            float(np.mean((frame[frame.season_start_year.eq(y)][baseline]
                           - frame[frame.season_start_year.eq(y)][TARGET]) ** 2)),
            float(np.mean((frame[frame.season_start_year.eq(y)][candidate]
                           - frame[frame.season_start_year.eq(y)][TARGET]) ** 2)),
        ) for y in years
    }
    gains = []
    for _ in range(draws):
        pick = rng.choice(years, size=len(years), replace=True)
        b = np.mean([per_year[int(y)][0] for y in pick])
        c = np.mean([per_year[int(y)][1] for y in pick])
        gains.append(np.sqrt(b) - np.sqrt(c))
    gains = np.array(gains)
    return {
        "mean_gain": float(gains.mean()),
        "p025": float(np.percentile(gains, 2.5)),
        "p975": float(np.percentile(gains, 97.5)),
        "probability_positive": float((gains > 0).mean()),
        "resampling_unit": "season",
        "groups": int(len(years)),
    }
