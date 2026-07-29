#!/usr/bin/env python3
"""Strengthen V14 with multi-year anomaly, isolated-outlook, and distribution tests.

This script deliberately does not throw away close candidates.  It reports:

1. anomaly shrinkage selected on 2016-2018 and tested on 2019-2022;
2. anomaly shrinkage selected on 2019-2020 and confirmed on 2021-2022;
3. the isolated correction caused by future-crop outlook features;
4. probability distributions centred on the production V5 point forecast.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


V14 = Path(__file__).resolve().parents[1]
ROOT = V14.parents[1]
sys.path.insert(0, str(ROOT))

from rapid_yield_forecast.v14_anomaly_distribution.scripts import run_v14_lab as lab  # noqa: E402


ARTIFACTS = V14 / "artifacts"
MODELS = V14 / "models"
TARGET = lab.TARGET
QUANTILES = lab.QUANTILES
EARLY_SELECTION = [2016, 2017, 2018]
FOUR_YEAR_TEST = [2019, 2020, 2021, 2022]
DEVELOPMENT = lab.DEVELOPMENT
LATE = lab.LATE
SHRINK_GRID = np.round(np.arange(0.0, 1.21, 0.10), 2)


def weighted_average(values: np.ndarray, years: np.ndarray) -> float:
    counts = pd.Series(years).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[int(year)] for year in years], dtype=float)
    return float(np.average(values, weights=weights))


def add_prediction_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    prefix: str = "",
) -> dict[str, float]:
    values = lab.metric_values(frame, prediction_column)
    return {f"{prefix}{key}": value for key, value in values.items()}


def tune_shrinkage(
    predictions: pd.DataFrame,
    selection_years: list[int],
    eligible_candidates: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = predictions[
        predictions["candidate"].isin(eligible_candidates)
        & predictions["season_start_year"].isin(selection_years)
    ].copy()
    for candidate, block in subset.groupby("candidate"):
        if block["season_start_year"].nunique() != len(selection_years):
            continue
        actual_log_anomaly = np.log(
            block[TARGET].to_numpy(float) / block["normal_prediction"].to_numpy(float)
        )
        predicted_anomaly = block["predicted_anomaly"].to_numpy(float)
        years = block["season_start_year"].to_numpy(int)
        for shrink in SHRINK_GRID:
            offset = weighted_average(
                actual_log_anomaly - shrink * predicted_anomaly,
                years,
            )
            offset = float(np.clip(offset, -0.10, 0.10))
            temp = block.copy()
            temp["tuned_prediction"] = np.clip(
                temp["normal_prediction"].to_numpy(float)
                * np.exp(offset + shrink * temp["predicted_anomaly"].to_numpy(float)),
                500,
                7000,
            )
            metrics = lab.metric_values(temp, "tuned_prediction")
            rows.append({
                "candidate": candidate,
                "normal": block["normal"].iloc[0],
                "feature_set": block["feature_set"].iloc[0],
                "model": block["model"].iloc[0],
                "selection_years": "-".join(map(str, selection_years)),
                "shrink": float(shrink),
                "log_offset": offset,
                **metrics,
                "selection_score": (
                    0.50 * metrics["rmse"]
                    + 0.25 * metrics["equal_state_rmse"]
                    + 0.25 * metrics["mean_year_rmse"]
                ),
            })
    return pd.DataFrame(rows).sort_values(["selection_score", "rmse", "candidate"])


def apply_shrinkage(
    predictions: pd.DataFrame,
    row: pd.Series,
    protocol: str,
) -> pd.DataFrame:
    block = predictions[predictions["candidate"].eq(row["candidate"])].copy()
    block["prediction_unshrunk"] = block["prediction"]
    block["prediction"] = np.clip(
        block["normal_prediction"].to_numpy(float)
        * np.exp(
            float(row["log_offset"])
            + float(row["shrink"]) * block["predicted_anomaly"].to_numpy(float)
        ),
        500,
        7000,
    )
    block["selection_protocol"] = protocol
    block["selected_shrink"] = float(row["shrink"])
    block["selected_log_offset"] = float(row["log_offset"])
    return block


def period_rows(
    frame: pd.DataFrame,
    prediction_column: str,
    protocol: str,
    candidate: str,
) -> list[dict[str, object]]:
    rows = []
    periods = [
        ("early_selection", EARLY_SELECTION),
        ("development", DEVELOPMENT),
        ("late", LATE),
        ("four_year", FOUR_YEAR_TEST),
        ("all_rolling", list(range(2016, 2023))),
    ]
    for period, years in periods:
        part = frame[frame["season_start_year"].isin(years)]
        if part["season_start_year"].nunique() != len(years):
            continue
        rows.append({
            "protocol": protocol,
            "candidate": candidate,
            "period": period,
            "rows": len(part),
            **lab.metric_values(part, prediction_column),
        })
    return rows


def year_state_audit(
    frame: pd.DataFrame,
    prediction_column: str,
    protocol: str,
    baseline_column: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for grouping in [("year", ["season_start_year"]), ("state_year", ["season_start_year", "state_name"])]:
        level, keys = grouping
        for key, part in frame.groupby(keys):
            if not isinstance(key, tuple):
                key = (key,)
            row: dict[str, object] = {
                "protocol": protocol,
                "level": level,
                "season_start_year": int(key[0]),
                "state_name": key[1] if len(key) > 1 else "ALL",
                "rows": len(part),
                **lab.metric_values(part, prediction_column),
            }
            if baseline_column is not None:
                row.update(add_prediction_metrics(part, baseline_column, "baseline_"))
                row["rmse_gain_vs_baseline"] = row["baseline_rmse"] - row["rmse"]
            rows.append(row)
    return pd.DataFrame(rows)


def anomaly_extensions() -> dict[str, object]:
    predictions = pd.read_parquet(ARTIFACTS / "standalone_candidate_predictions.parquet")
    long_candidates = sorted(
        predictions.loc[
            ~predictions["feature_set"].str.contains("outlook"), "candidate"
        ].unique()
    )
    all_candidates = sorted(predictions["candidate"].unique())

    early_grid = tune_shrinkage(predictions, EARLY_SELECTION, long_candidates)
    development_grid = tune_shrinkage(predictions, DEVELOPMENT, all_candidates)
    recent_four_grid = tune_shrinkage(
        predictions, [2017, 2018, 2019, 2020], long_candidates
    )
    recent_three_grid = tune_shrinkage(
        predictions, [2018, 2019, 2020], long_candidates
    )
    early_grid["protocol"] = "select_2016_2018_test_2019_2022"
    development_grid["protocol"] = "select_2019_2020_test_2021_2022"
    recent_four_grid["protocol"] = "select_2017_2020_test_2021_2022"
    recent_three_grid["protocol"] = "select_2018_2020_test_2021_2022"
    grids = pd.concat(
        [early_grid, development_grid, recent_four_grid, recent_three_grid],
        ignore_index=True,
    )
    grids.to_csv(ARTIFACTS / "anomaly_shrink_grid.csv", index=False)

    selected_frames = []
    metric_rows: list[dict[str, object]] = []
    audit_frames = []
    winners = [
        ("select_2016_2018_test_2019_2022", early_grid.iloc[0]),
        ("select_2019_2020_test_2021_2022", development_grid.iloc[0]),
        ("select_2017_2020_test_2021_2022", recent_four_grid.iloc[0]),
        ("select_2018_2020_test_2021_2022", recent_three_grid.iloc[0]),
    ]
    for protocol, winner in winners:
        chosen = apply_shrinkage(predictions, winner, protocol)
        selected_frames.append(chosen)
        metric_rows.extend(period_rows(chosen, "prediction", protocol, str(winner["candidate"])))
        audit_frames.append(year_state_audit(chosen, "prediction", protocol))

    # A diversity ensemble is selected using only 2016-2018.  It prevents one
    # model/normal family from dominating merely because of one unusual year.
    diverse_rows = []
    seen: set[tuple[str, str]] = set()
    for _, row in early_grid.iterrows():
        key = (str(row["normal"]), str(row["model"]))
        if key in seen:
            continue
        diverse_rows.append(row)
        seen.add(key)
        if len(diverse_rows) == 5:
            break
    ensemble_parts = []
    for row in diverse_rows:
        part = apply_shrinkage(predictions, row, "early_diverse_top5")
        ensemble_parts.append(part[[
            "district_id", "season_start_year", "prediction"
        ]].rename(columns={"prediction": f"prediction_{len(ensemble_parts)}"}))
    ensemble = ensemble_parts[0]
    for part in ensemble_parts[1:]:
        ensemble = ensemble.merge(
            part, on=["district_id", "season_start_year"], validate="one_to_one"
        )
    reference = predictions[
        predictions["candidate"].eq(diverse_rows[0]["candidate"])
    ][[
        "district_id", "state_name", "district_name", "season_start_year",
        TARGET, "lag_1_yield",
    ]]
    ensemble = reference.merge(
        ensemble, on=["district_id", "season_start_year"], validate="one_to_one"
    )
    ensemble["prediction"] = ensemble.filter(regex=r"^prediction_\d+$").mean(axis=1)
    ensemble["candidate"] = "early_diverse_top5"
    ensemble["selection_protocol"] = "select_2016_2018_test_2019_2022"
    selected_frames.append(ensemble)
    metric_rows.extend(period_rows(
        ensemble,
        "prediction",
        "select_2016_2018_test_2019_2022",
        "early_diverse_top5",
    ))
    audit_frames.append(year_state_audit(
        ensemble, "prediction", "early_diverse_top5"
    ))

    selected = pd.concat(selected_frames, ignore_index=True, sort=False)
    selected.to_parquet(ARTIFACTS / "anomaly_protocol_predictions.parquet", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ARTIFACTS / "anomaly_protocol_metrics.csv", index=False)
    pd.concat(audit_frames, ignore_index=True).to_csv(
        ARTIFACTS / "anomaly_year_state_audit.csv", index=False
    )

    summary: dict[str, object] = {
        "early_winner": early_grid.iloc[0].to_dict(),
        "development_winner": development_grid.iloc[0].to_dict(),
        "recent_four_winner": recent_four_grid.iloc[0].to_dict(),
        "recent_three_winner": recent_three_grid.iloc[0].to_dict(),
        "early_diverse_members": [
            {
                "candidate": row["candidate"],
                "shrink": row["shrink"],
                "log_offset": row["log_offset"],
            }
            for row in diverse_rows
        ],
        "metrics": metrics.to_dict("records"),
    }
    return summary


def pivot_xgb(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "district_id", "state_name", "district_name",
        "season_start_year", TARGET, "lag_1_yield",
    ]
    wide = predictions.pivot_table(
        index=keys,
        columns="candidate",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def correction_candidates(wide: pd.DataFrame) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    for depth in (1, 2):
        prefix = f"xgb_d{depth}__"
        base = wide[f"{prefix}base"].to_numpy(float)
        no_future = wide[f"{prefix}base_no_future"].to_numpy(float)
        full = wide[f"{prefix}base_full"].to_numpy(float)
        effect = wide[f"{prefix}base_future_effect"].to_numpy(float)
        broad = wide[f"{prefix}base_full_broad"].to_numpy(float)
        candidates[f"d{depth}_no_future_minus_base"] = no_future - base
        candidates[f"d{depth}_full_minus_base"] = full - base
        candidates[f"d{depth}_effect_minus_base"] = effect - base
        candidates[f"d{depth}_broad_minus_base"] = broad - base
        candidates[f"d{depth}_full_minus_no_future"] = full - no_future
        candidates[f"d{depth}_effect_minus_no_future"] = effect - no_future
        candidates[f"d{depth}_broad_minus_no_future"] = broad - no_future
        candidates[f"d{depth}_future_ensemble_minus_base"] = (
            (full + effect + broad) / 3.0 - base
        )
        candidates[f"d{depth}_future_ensemble_minus_no_future"] = (
            (full + effect + broad) / 3.0 - no_future
        )
    return candidates


def outlook_extensions() -> dict[str, object]:
    predictions = pd.read_parquet(ARTIFACTS / "xgb_outlook_predictions.parquet")
    v5 = pd.read_csv(lab.V5_PRED)[[
        "district_id", "season_start_year", "prediction"
    ]].rename(columns={"prediction": "v5_prediction"})
    wide = pivot_xgb(predictions).merge(
        v5, on=["district_id", "season_start_year"], validate="one_to_one"
    )

    # Raw model audit across all four genuinely forecast years.
    raw_rows = []
    for candidate in sorted(predictions["candidate"].unique()):
        block = predictions[predictions["candidate"].eq(candidate)]
        depth = int(block["depth"].iloc[0])
        baseline = predictions[
            predictions["candidate"].eq(f"xgb_d{depth}__base")
        ][["district_id", "season_start_year", "prediction"]].rename(
            columns={"prediction": "baseline_prediction"}
        )
        block = block.merge(
            baseline, on=["district_id", "season_start_year"], validate="one_to_one"
        )
        for period, years in [
            ("development", DEVELOPMENT),
            ("late", LATE),
            ("four_year", FOUR_YEAR_TEST),
        ]:
            part = block[block["season_start_year"].isin(years)]
            row = {
                "candidate": candidate,
                "feature_set": block["feature_set"].iloc[0],
                "depth": depth,
                "period": period,
                **lab.metric_values(part),
                **add_prediction_metrics(part, "baseline_prediction", "base_"),
            }
            row["rmse_gain_vs_same_depth_base"] = row["base_rmse"] - row["rmse"]
            raw_rows.append(row)
    raw_metrics = pd.DataFrame(raw_rows)
    raw_metrics.to_csv(ARTIFACTS / "outlook_raw_four_year_metrics.csv", index=False)

    corrections = correction_candidates(wide)
    correction_grid_rows = []
    correction_prediction_rows = []
    gamma_grid = np.round(np.arange(-2.0, 5.01, 0.25), 2)
    for name, correction in corrections.items():
        temp = wide.copy()
        temp["correction"] = correction
        for gamma in gamma_grid:
            temp["corrected_prediction"] = np.clip(
                temp["v5_prediction"] + gamma * temp["correction"],
                500,
                7000,
            )
            dev = temp[temp["season_start_year"].isin(DEVELOPMENT)]
            metrics = lab.metric_values(dev, "corrected_prediction")
            correction_grid_rows.append({
                "correction_candidate": name,
                "gamma": float(gamma),
                **{f"development_{key}": value for key, value in metrics.items()},
                "selection_score": (
                    0.50 * metrics["rmse"]
                    + 0.25 * metrics["equal_state_rmse"]
                    + 0.25 * metrics["mean_year_rmse"]
                ),
            })
    correction_grid = pd.DataFrame(correction_grid_rows).sort_values(
        ["selection_score", "correction_candidate", "gamma"]
    )
    correction_grid.to_csv(ARTIFACTS / "outlook_isolated_correction_grid.csv", index=False)

    exact_winner = correction_grid.iloc[0]
    # A one-standard-error-style rule: corrections whose development score is
    # within 0.1% of the best are treated as tied, and the smallest correction
    # is preferred.  This guards against an unstable large extrapolation.
    tolerance = 0.001 * float(exact_winner["selection_score"])
    tied = correction_grid[
        correction_grid["selection_score"].le(
            float(exact_winner["selection_score"]) + tolerance
        )
        & correction_grid["gamma"].ge(0)
    ].copy()
    regularized_winner = tied.sort_values(
        ["gamma", "selection_score"]
    ).iloc[0]
    future_pool = correction_grid[
        correction_grid["gamma"].ge(0)
        & correction_grid["correction_candidate"].str.contains(
            "minus_no_future"
        )
    ].copy()
    future_best = future_pool.iloc[0]
    future_tied = future_pool[
        future_pool["selection_score"].le(
            float(future_best["selection_score"])
            + 0.001 * float(future_best["selection_score"])
        )
    ]
    future_regularized = future_tied.sort_values(
        ["gamma", "selection_score"]
    ).iloc[0]
    winners = {
        "exact_development_optimum": exact_winner,
        "regularized_near_tie": regularized_winner,
        "future_only_regularized": future_regularized,
    }
    selected_metrics = []
    selected_audits = []
    for protocol, winner in winners.items():
        name = str(winner["correction_candidate"])
        temp = wide.copy()
        temp["correction"] = corrections[name]
        temp["prediction"] = np.clip(
            temp["v5_prediction"] + float(winner["gamma"]) * temp["correction"],
            500,
            7000,
        )
        temp["candidate"] = name
        temp["protocol"] = protocol
        temp["gamma"] = float(winner["gamma"])
        correction_prediction_rows.append(temp)
        for period, years in [
            ("development", DEVELOPMENT),
            ("late", LATE),
            ("four_year", FOUR_YEAR_TEST),
        ]:
            part = temp[temp["season_start_year"].isin(years)]
            row = {
                "protocol": protocol,
                "candidate": name,
                "gamma": float(winner["gamma"]),
                "period": period,
                **lab.metric_values(part),
                **add_prediction_metrics(part, "v5_prediction", "v5_"),
            }
            row["rmse_gain_vs_v5"] = row["v5_rmse"] - row["rmse"]
            if period in {"late", "four_year"}:
                row.update(lab.grouped_rmse_bootstrap(
                    part.reset_index(drop=True), "prediction", "v5_prediction"
                ))
            selected_metrics.append(row)
        selected_audits.append(year_state_audit(
            temp, "prediction", f"outlook_{protocol}", "v5_prediction"
        ))
    pd.concat(correction_prediction_rows, ignore_index=True).to_parquet(
        ARTIFACTS / "outlook_isolated_selected_predictions.parquet", index=False
    )
    selected_metrics_frame = pd.DataFrame(selected_metrics)
    selected_metrics_frame.to_csv(
        ARTIFACTS / "outlook_isolated_selected_metrics.csv", index=False
    )
    pd.concat(selected_audits, ignore_index=True).to_csv(
        ARTIFACTS / "outlook_year_state_audit.csv", index=False
    )

    # Pairwise bootstraps preserve all near-ties instead of ranking only by RMSE.
    pair_rows = []
    for depth in (1, 2):
        for feature in [
            "base_no_future", "base_full", "base_future_effect", "base_full_broad"
        ]:
            candidate = f"xgb_d{depth}__{feature}"
            baseline = f"xgb_d{depth}__base"
            pair = wide[[
                "district_id", "state_name", "season_start_year", TARGET,
                candidate, baseline,
            ]].rename(columns={candidate: "candidate_prediction", baseline: "base_prediction"})
            for period, years in [("late", LATE), ("four_year", FOUR_YEAR_TEST)]:
                part = pair[pair["season_start_year"].isin(years)].reset_index(drop=True)
                boot = lab.grouped_rmse_bootstrap(
                    part, "candidate_prediction", "base_prediction"
                )
                pair_rows.append({
                    "candidate": candidate,
                    "baseline": baseline,
                    "period": period,
                    **boot,
                })
    pd.DataFrame(pair_rows).to_csv(
        ARTIFACTS / "outlook_pairwise_bootstrap.csv", index=False
    )
    return {
        "raw_metrics": raw_metrics.to_dict("records"),
        "isolated_winners": {
            protocol: winner.to_dict() for protocol, winner in winners.items()
        },
        "isolated_metrics": selected_metrics,
    }


def quantile_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame[TARGET].to_numpy(float)
    losses = []
    for q in QUANTILES:
        prediction = frame[f"q{int(q * 100):02d}"].to_numpy(float)
        losses.append((q - (actual < prediction).astype(float)) * (actual - prediction))
    result = {
        "mean_pinball_loss": float(np.mean(np.stack(losses, axis=1))),
        "approx_crps": float(2.0 * np.mean(np.stack(losses, axis=1))),
    }
    for coverage, lower, upper in [(50, 25, 75), (80, 10, 90), (90, 5, 95)]:
        result[f"coverage_{coverage}"] = float(np.mean(
            (actual >= frame[f"q{lower:02d}"])
            & (actual <= frame[f"q{upper:02d}"])
        ))
        result[f"width_{coverage}"] = float(np.mean(
            frame[f"q{upper:02d}"] - frame[f"q{lower:02d}"]
        ))
    return result


def cdf_at(quantiles: np.ndarray, threshold: float) -> float:
    values = np.maximum.accumulate(np.asarray(quantiles, dtype=float))
    probabilities = QUANTILES.astype(float)
    if threshold < values[0]:
        return float(0.05 * max(0.0, (threshold - 500.0) / max(values[0] - 500.0, 1.0)))
    if threshold > values[-1]:
        return float(0.95 + 0.05 * min(
            1.0, (threshold - values[-1]) / max(7000.0 - values[-1], 1.0)
        ))
    return float(np.interp(threshold, values, probabilities))


def add_distribution_moments(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    q_columns = [f"q{int(q * 100):02d}" for q in QUANTILES]
    q_matrix = result[q_columns].to_numpy(float)
    q_matrix = np.maximum.accumulate(q_matrix, axis=1)
    result[q_columns] = q_matrix
    # Add simple 0 and 1 endpoints for integration.  This does not claim that
    # 500/7000 are likely; they merely close the numerical integral.
    probabilities = np.concatenate([[0.0], QUANTILES, [1.0]])
    values = np.concatenate([
        np.maximum(500.0, q_matrix[:, :1] - (q_matrix[:, 1:2] - q_matrix[:, :1])),
        q_matrix,
        np.minimum(7000.0, q_matrix[:, -1:] + (q_matrix[:, -1:] - q_matrix[:, -2:-1])),
    ], axis=1)
    mean = np.trapezoid(values, probabilities, axis=1)
    second = np.trapezoid(values ** 2, probabilities, axis=1)
    result["distribution_mean"] = mean
    result["distribution_sd"] = np.sqrt(np.maximum(second - mean ** 2, 0.0))
    result["probability_rise"] = [
        1.0 - cdf_at(row, threshold)
        for row, threshold in zip(q_matrix, result["lag_1_yield"].to_numpy(float))
    ]
    result["probability_severe_drop"] = [
        cdf_at(row, 0.90 * threshold)
        for row, threshold in zip(q_matrix, result["lag_1_yield"].to_numpy(float))
    ]
    return result


def probability_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual_rise = (frame[TARGET] > frame["lag_1_yield"]).astype(int).to_numpy()
    actual_severe = (frame[TARGET] <= 0.90 * frame["lag_1_yield"]).astype(int).to_numpy()
    rise_probability = frame["probability_rise"].clip(1e-6, 1 - 1e-6).to_numpy(float)
    severe_probability = frame["probability_severe_drop"].clip(1e-6, 1 - 1e-6).to_numpy(float)

    def auc(actual: np.ndarray, probability: np.ndarray) -> float:
        return float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else math.nan

    def log_loss(actual: np.ndarray, probability: np.ndarray) -> float:
        return float(-np.mean(
            actual * np.log(probability)
            + (1 - actual) * np.log(1 - probability)
        ))

    return {
        "rise_auc": auc(actual_rise, rise_probability),
        "rise_brier": float(np.mean((rise_probability - actual_rise) ** 2)),
        "rise_log_loss": log_loss(actual_rise, rise_probability),
        "severe_drop_auc": auc(actual_severe, severe_probability),
        "severe_drop_brier": float(np.mean((severe_probability - actual_severe) ** 2)),
        "severe_drop_log_loss": log_loss(actual_severe, severe_probability),
    }


def build_history_shape_distributions() -> pd.DataFrame:
    source = pd.read_parquet(ARTIFACTS / "distribution_candidate_predictions.parquet")
    v5 = pd.read_csv(lab.V5_PRED)[[
        "district_id", "season_start_year", "prediction"
    ]].rename(columns={"prediction": "v5_prediction"})
    source = source.merge(v5, on=["district_id", "season_start_year"], validate="many_to_one")
    rows = []
    q_columns = [f"q{int(q * 100):02d}" for q in QUANTILES]
    for method, block in source.groupby("method"):
        source_median = block["q50"].to_numpy(float)
        for inflation in [0.80, 1.00, 1.20, 1.40, 1.60]:
            output = block[[
                "district_id", "state_name", "district_name", "season_start_year",
                TARGET, "lag_1_yield", "v5_prediction",
            ]].copy()
            for column in q_columns:
                output[column] = np.clip(
                    block["v5_prediction"].to_numpy(float)
                    + inflation * (block[column].to_numpy(float) - source_median),
                    500,
                    7000,
                )
            output["method"] = f"history_shape__{method}__w{inflation:.2f}"
            output["calibration_role"] = "strict_rolling_history_shape"
            rows.append(add_distribution_moments(output))
    return pd.concat(rows, ignore_index=True)


def distribution_selection_score(values: dict[str, float]) -> float:
    under_80 = max(0.0, 0.78 - values["coverage_80"])
    under_90 = max(0.0, 0.88 - values["coverage_90"])
    over_80 = max(0.0, values["coverage_80"] - 0.88)
    return (
        values["mean_pinball_loss"]
        + 1000.0 * under_80 ** 2
        + 700.0 * under_90 ** 2
        + 150.0 * over_80 ** 2
    )


def calibrate_quantile_shifts(
    frame: pd.DataFrame,
    calibration_years: list[int],
    method_name: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = frame.copy()
    calibration = frame[frame["season_start_year"].isin(calibration_years)].copy()
    years = calibration["season_start_year"].to_numpy(int)
    counts = pd.Series(years).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[int(year)] for year in years], dtype=float)
    shifts: dict[str, float] = {}
    for q in QUANTILES:
        column = f"q{int(q * 100):02d}"
        residual = (
            calibration[TARGET].to_numpy(float)
            - calibration[column].to_numpy(float)
        )
        shift = float(lab.weighted_quantile(
            residual, np.asarray([q]), weights
        )[0])
        shifts[column] = shift
        result[column] = np.clip(result[column] + shift, 500, 7000)
    result["method"] = method_name
    result["calibration_role"] = "quantile_shift_fit_2019_2020"
    return add_distribution_moments(result), shifts


def v5_residual_distribution(base: pd.DataFrame) -> pd.DataFrame:
    v5 = pd.read_csv(lab.V5_PRED)
    scale_columns = base[[
        "district_id", "season_start_year", "yield_recent_std", "normal__weighted3"
    ]]
    v5 = v5.merge(
        scale_columns,
        on=["district_id", "season_start_year"],
        validate="one_to_one",
    )
    calibration = v5[v5["season_start_year"].isin(DEVELOPMENT)].copy()
    target = v5[v5["season_start_year"].isin(FOUR_YEAR_TEST)].copy()
    methods = ["global_equal_year", "scaled_global_equal_year", "scaled_state_equal_year"]
    q_columns = [f"q{int(q * 100):02d}" for q in QUANTILES]
    rows = []
    for method in methods:
        for _, row in target.iterrows():
            residual = (
                calibration[TARGET].to_numpy(float)
                - calibration["prediction"].to_numpy(float)
            )
            cal_years = calibration["season_start_year"].to_numpy(int)
            counts = pd.Series(cal_years).value_counts().to_dict()
            weights = np.asarray([1.0 / counts[int(year)] for year in cal_years])
            target_scale = max(
                float(row["yield_recent_std"]) if np.isfinite(row["yield_recent_std"]) else 0.0,
                0.07 * float(row["normal__weighted3"]),
                150.0,
            )
            if "scaled" in method:
                cal_scale = np.maximum.reduce([
                    calibration["yield_recent_std"].fillna(0).to_numpy(float),
                    0.07 * calibration["normal__weighted3"].to_numpy(float),
                    np.full(len(calibration), 150.0),
                ])
                residual = residual / cal_scale
            if "state" in method:
                state_mask = calibration["state_name"].eq(row["state_name"]).to_numpy()
                if state_mask.sum() >= 5:
                    shrink = state_mask.sum() / (state_mask.sum() + 50.0)
                    state_residual = residual[state_mask]
                    state_weights = weights[state_mask]
                    weights = weights / weights.sum() * (1 - shrink)
                    state_weights = state_weights / state_weights.sum() * shrink
                    residual = np.concatenate([residual, state_residual])
                    weights = np.concatenate([weights, state_weights])
            if "scaled" in method:
                residual = residual * target_scale
            quantile_residual = lab.weighted_quantile(residual, QUANTILES, weights)
            output = {
                "district_id": row["district_id"],
                "state_name": row["state_name"],
                "district_name": row["district_name"],
                "season_start_year": int(row["season_start_year"]),
                TARGET: float(row[TARGET]),
                "lag_1_yield": float(row["lag_1_yield"]),
                "v5_prediction": float(row["prediction"]),
                "method": f"v5_dev_residual__{method}",
                "calibration_role": "fit_2019_2020_in_sample_for_dev_strict_for_late",
            }
            for column, value in zip(q_columns, row["prediction"] + quantile_residual):
                output[column] = float(np.clip(value, 500, 7000))
            rows.append(output)
    return add_distribution_moments(pd.DataFrame(rows))


def distribution_extensions() -> dict[str, object]:
    history = build_history_shape_distributions()
    rows = []
    for method, block in history.groupby("method"):
        dev = block[block["season_start_year"].isin(DEVELOPMENT)]
        values = quantile_metrics(dev)
        rows.append({
            "method": method,
            "family": "history_shape",
            "period": "development",
            **values,
            **probability_metrics(dev),
            "selection_score": distribution_selection_score(values),
        })
    history_metrics = pd.DataFrame(rows).sort_values("selection_score")
    clean_winner = str(history_metrics.iloc[0]["method"])
    clean = history[history["method"].eq(clean_winner)].copy()
    calibrated, shifts = calibrate_quantile_shifts(
        clean, DEVELOPMENT, f"calibrated__{clean_winner}"
    )

    base, _, _ = lab.load_panel()
    direct = v5_residual_distribution(base)
    shadow_point = pd.read_parquet(
        ARTIFACTS / "outlook_isolated_selected_predictions.parquet"
    )
    shadow_point = shadow_point[
        shadow_point["protocol"].eq("regularized_near_tie")
    ][[
        "district_id", "season_start_year", "prediction", "v5_prediction"
    ]].rename(columns={"prediction": "shadow_point_prediction"})
    shadow_distribution = clean.merge(
        shadow_point,
        on=["district_id", "season_start_year", "v5_prediction"],
        validate="one_to_one",
    )
    point_shift = (
        shadow_distribution["shadow_point_prediction"].to_numpy(float)
        - shadow_distribution["v5_prediction"].to_numpy(float)
    )
    for q in QUANTILES:
        column = f"q{int(q * 100):02d}"
        shadow_distribution[column] = np.clip(
            shadow_distribution[column].to_numpy(float) + point_shift,
            500,
            7000,
        )
    shadow_distribution["method"] = f"outlook_corrected__{clean_winner}"
    shadow_distribution["calibration_role"] = (
        "strict_history_shape_around_regularized_outlook_shadow_point"
    )
    shadow_distribution = add_distribution_moments(shadow_distribution)

    all_candidates = pd.concat(
        [history, calibrated, direct, shadow_distribution],
        ignore_index=True,
        sort=False,
    )
    all_candidates.to_parquet(
        ARTIFACTS / "v5_distribution_candidate_predictions.parquet", index=False
    )

    metric_rows = []
    for method, block in all_candidates.groupby("method"):
        for period, years in [
            ("development", DEVELOPMENT),
            ("late", LATE),
            ("four_year", FOUR_YEAR_TEST),
        ]:
            part = block[block["season_start_year"].isin(years)]
            if part["season_start_year"].nunique() != len(years):
                continue
            values = quantile_metrics(part)
            metric_rows.append({
                "method": method,
                "period": period,
                "rows": len(part),
                **values,
                **probability_metrics(part),
                "selection_score": (
                    distribution_selection_score(values)
                    if period == "development" else math.nan
                ),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ARTIFACTS / "v5_distribution_metrics.csv", index=False)

    # The clean winner is chosen using strict rolling distributions in 2019-20.
    # The calibrated version is retained as a separate candidate and judged only
    # on 2021-22, since the quantile shifts were fit on 2019-20.
    production_candidates = [
        clean_winner,
        f"calibrated__{clean_winner}",
        f"outlook_corrected__{clean_winner}",
    ]
    selected = all_candidates[
        all_candidates["method"].isin(production_candidates)
    ].copy()
    selected.to_parquet(
        ARTIFACTS / "v5_distribution_selected_predictions.parquet", index=False
    )

    # Per-year evidence is essential with only two late years.
    audits = []
    for method, block in selected.groupby("method"):
        for year, part in block.groupby("season_start_year"):
            audits.append({
                "method": method,
                "season_start_year": int(year),
                **quantile_metrics(part),
                **probability_metrics(part),
            })
    pd.DataFrame(audits).to_csv(
        ARTIFACTS / "v5_distribution_year_audit.csv", index=False
    )
    recipe = {
        "point_model": "V5 production forecast",
        "strict_history_shape_winner": clean_winner,
        "calibrated_candidate": f"calibrated__{clean_winner}",
        "outlook_shadow_candidate": f"outlook_corrected__{clean_winner}",
        "calibration_years": DEVELOPMENT,
        "quantile_shifts_kg_per_ha": shifts,
        "quantiles": QUANTILES.tolist(),
        "selection_rule": (
            "Choose history-shape width on 2019-2020; retain its calibrated "
            "version separately; confirm both on 2021-2022."
        ),
    }
    (MODELS / "v5_distribution_recipe.json").write_text(
        json.dumps(recipe, indent=2)
    )
    return {
        "strict_history_winner": clean_winner,
        "calibrated_candidate": f"calibrated__{clean_winner}",
        "outlook_shadow_candidate": f"outlook_corrected__{clean_winner}",
        "recipe": recipe,
        "metrics": metrics[
            metrics["method"].isin(production_candidates)
        ].to_dict("records"),
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    anomaly = anomaly_extensions()
    outlook = outlook_extensions()
    distribution = distribution_extensions()
    summary = {
        "anomaly": anomaly,
        "outlook": outlook,
        "distribution": distribution,
        "evaluation_policy": {
            "near_tie_rule": (
                "A candidate within 10 kg/ha or 3% of the incumbent is retained "
                "as shadow unless multi-year evidence is consistently adverse."
            ),
            "early_anomaly_selection_years": EARLY_SELECTION,
            "four_year_anomaly_test_years": FOUR_YEAR_TEST,
            "outlook_selection_years": DEVELOPMENT,
            "outlook_confirmation_years": LATE,
            "post_2022_yield_labels_read": False,
        },
    }
    (ARTIFACTS / "extension_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
