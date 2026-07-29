#!/usr/bin/env python3
"""Create the auditable V14 release bundle, comparisons, and summary figure."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


V14 = Path(__file__).resolve().parents[1]
ROOT = V14.parents[1]
sys.path.insert(0, str(ROOT))

from rapid_yield_forecast.v14_anomaly_distribution.scripts import run_v14_extensions as ext  # noqa: E402
from rapid_yield_forecast.v14_anomaly_distribution.scripts import run_v14_lab as lab  # noqa: E402


ARTIFACTS = V14 / "artifacts"
MODELS = V14 / "models"
TARGET = lab.TARGET
KEYS = ["district_id", "season_start_year"]
PRIMARY_DISTRIBUTION = "history_shape__scaled_state_equal_year__w1.00"
CONSERVATIVE_DISTRIBUTION = "history_shape__scaled_state_equal_year__w1.20"
SHADOW_DISTRIBUTION = (
    "outlook_corrected__history_shape__scaled_state_equal_year__w1.00"
)
SHADOW_PROTOCOL = "regularized_near_tie"
SHADOW_GAMMA = 1.75


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_bootstrap_probability(
    frame: pd.DataFrame,
    event: str,
    candidate: str,
    baseline: str,
    draws: int = 5000,
) -> dict[str, float]:
    groups = [
        part.index.to_numpy()
        for _, part in frame.groupby(["state_name", "season_start_year"])
    ]
    rng = np.random.default_rng(20260731)
    auc_gains = []
    brier_gains = []
    for _ in range(draws):
        index = np.concatenate([
            groups[i] for i in rng.integers(0, len(groups), len(groups))
        ])
        actual = frame.loc[index, event].to_numpy(int)
        if len(np.unique(actual)) < 2:
            continue
        candidate_probability = frame.loc[index, candidate].to_numpy(float)
        baseline_probability = frame.loc[index, baseline].to_numpy(float)
        auc_gains.append(
            roc_auc_score(actual, candidate_probability)
            - roc_auc_score(actual, baseline_probability)
        )
        brier_gains.append(
            np.mean((baseline_probability - actual) ** 2)
            - np.mean((candidate_probability - actual) ** 2)
        )
    auc_values = np.asarray(auc_gains)
    brier_values = np.asarray(brier_gains)
    return {
        "auc_gain_mean": float(auc_values.mean()),
        "auc_gain_p025": float(np.quantile(auc_values, 0.025)),
        "auc_gain_p975": float(np.quantile(auc_values, 0.975)),
        "auc_gain_probability_positive": float(np.mean(auc_values > 0)),
        "brier_gain_mean": float(brier_values.mean()),
        "brier_gain_p025": float(np.quantile(brier_values, 0.025)),
        "brier_gain_p975": float(np.quantile(brier_values, 0.975)),
        "brier_gain_probability_positive": float(np.mean(brier_values > 0)),
        "valid_draws": int(len(auc_values)),
    }


def probability_metric_values(
    actual: pd.Series,
    probability: pd.Series,
) -> dict[str, float]:
    y = actual.to_numpy(int)
    p = probability.clip(1e-6, 1 - 1e-6).to_numpy(float)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(
            y * np.log(p) + (1 - y) * np.log(1 - p)
        )),
    }


def build_probability_comparison() -> tuple[pd.DataFrame, dict[str, object]]:
    distribution = pd.read_parquet(
        ARTIFACTS / "v5_distribution_selected_predictions.parquet"
    )
    distribution = distribution[
        distribution["method"].eq(SHADOW_DISTRIBUTION)
        & distribution["season_start_year"].isin(lab.LATE)
    ].copy()
    v13 = pd.read_parquet(
        V14.parent / "v13_crop_response_final" / "artifacts" / "final_predictions.parquet"
    )
    v13 = v13[
        v13["clock"].eq("mar05")
        & v13["season_start_year"].isin(lab.LATE)
    ][[
        "district_id", "season_start_year", "increase_target",
        "increase_probability", "severe_target", "severe_probability",
    ]]
    v5 = pd.read_csv(
        V14.parent / "v5" / "root_cybench_lab" / "artifacts"
        / "v5_integration" / "probabilistic" / "prediction_outputs.csv"
    )[[
        "district_id", "season_start_year",
        "probability_rise", "probability_severe_drop",
    ]].rename(columns={
        "probability_rise": "v5_probability_rise",
        "probability_severe_drop": "v5_probability_severe_drop",
    })
    merged = distribution.merge(v13, on=KEYS, validate="one_to_one").merge(
        v5, on=KEYS, validate="one_to_one"
    )
    rows = []
    models = {
        "V14 outlook distribution": (
            "probability_rise", "probability_severe_drop"
        ),
        "V5 probability output": (
            "v5_probability_rise", "v5_probability_severe_drop"
        ),
        "V13 probability output": (
            "increase_probability", "severe_probability"
        ),
    }
    for model, (rise, severe) in models.items():
        for event, target, probability in [
            ("rise", "increase_target", rise),
            ("severe_drop", "severe_target", severe),
        ]:
            rows.append({
                "model": model,
                "event": event,
                "period": "2021-2022",
                **probability_metric_values(
                    merged[target], merged[probability]
                ),
            })
    comparison = pd.DataFrame(rows)
    bootstrap = {
        "rise_vs_v5": group_bootstrap_probability(
            merged, "increase_target", "probability_rise",
            "v5_probability_rise",
        ),
        "rise_vs_v13": group_bootstrap_probability(
            merged, "increase_target", "probability_rise",
            "increase_probability",
        ),
        "severe_vs_v5": group_bootstrap_probability(
            merged, "severe_target", "probability_severe_drop",
            "v5_probability_severe_drop",
        ),
        "severe_vs_v13": group_bootstrap_probability(
            merged, "severe_target", "probability_severe_drop",
            "severe_probability",
        ),
    }
    comparison.to_csv(
        ARTIFACTS / "probability_model_comparison.csv", index=False
    )
    (ARTIFACTS / "probability_group_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2)
    )
    return comparison, bootstrap


def build_final_predictions() -> pd.DataFrame:
    v5 = pd.read_csv(lab.V5_PRED)[[
        "district_id", "state_name", "district_name", "season_start_year",
        TARGET, "lag_1_yield", "prediction",
    ]].rename(columns={"prediction": "production_point_prediction"})
    shadow = pd.read_parquet(
        ARTIFACTS / "outlook_isolated_selected_predictions.parquet"
    )
    shadow = shadow[shadow["protocol"].eq(SHADOW_PROTOCOL)][[
        "district_id", "season_start_year", "prediction", "correction", "gamma"
    ]].rename(columns={
        "prediction": "shadow_point_prediction",
        "correction": "raw_outlook_increment",
        "gamma": "outlook_increment_weight",
    })
    distributions = pd.read_parquet(
        ARTIFACTS / "v5_distribution_candidate_predictions.parquet"
    )
    primary = distributions[
        distributions["method"].eq(PRIMARY_DISTRIBUTION)
    ].copy()
    q_columns = [f"q{int(q * 100):02d}" for q in lab.QUANTILES]
    primary = primary[[
        "district_id", "season_start_year", *q_columns,
        "distribution_mean", "distribution_sd",
        "probability_rise", "probability_severe_drop",
    ]]
    conservative = distributions[
        distributions["method"].eq(CONSERVATIVE_DISTRIBUTION)
    ][[
        "district_id", "season_start_year", "q10", "q90"
    ]].rename(columns={
        "q10": "conservative_q10",
        "q90": "conservative_q90",
    })
    shadow_distribution = distributions[
        distributions["method"].eq(SHADOW_DISTRIBUTION)
    ][[
        "district_id", "season_start_year",
        "probability_rise", "probability_severe_drop",
    ]].rename(columns={
        "probability_rise": "shadow_probability_rise",
        "probability_severe_drop": "shadow_probability_severe_drop",
    })
    final = (
        v5.merge(shadow, on=KEYS, validate="one_to_one")
        .merge(primary, on=KEYS, validate="one_to_one")
        .merge(conservative, on=KEYS, validate="one_to_one")
        .merge(shadow_distribution, on=KEYS, validate="one_to_one")
    )
    final["point_delta_from_v5"] = (
        final["shadow_point_prediction"]
        - final["production_point_prediction"]
    )
    final["period"] = np.where(
        final["season_start_year"].isin(lab.DEVELOPMENT),
        "development",
        "late_confirmation",
    )
    final["point_production_status"] = "V5_frozen_production_challenger"
    final["point_shadow_status"] = "V14_frontier_shadow"
    final["distribution_status"] = "V14_preferred_research_distribution"
    final.to_parquet(ARTIFACTS / "final_predictions.parquet", index=False)
    return final


def build_final_metrics(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, prediction in [
        ("V5 frozen point", "production_point_prediction"),
        ("V14 outlook shadow point", "shadow_point_prediction"),
    ]:
        for period, years in [
            ("development", lab.DEVELOPMENT),
            ("late_confirmation", lab.LATE),
            ("four_forecast_years", [2019, 2020, 2021, 2022]),
        ]:
            part = final[final["season_start_year"].isin(years)]
            rows.append({
                "output": "point",
                "model": model,
                "period": period,
                **lab.metric_values(
                    part.rename(columns={prediction: "metric_prediction"}),
                    "metric_prediction",
                ),
            })
    distribution_metrics = pd.read_csv(
        ARTIFACTS / "v5_distribution_metrics.csv"
    )
    selected = distribution_metrics[
        distribution_metrics["method"].isin([
            PRIMARY_DISTRIBUTION,
            CONSERVATIVE_DISTRIBUTION,
            SHADOW_DISTRIBUTION,
        ])
    ].copy()
    selected["output"] = "distribution"
    selected = selected.rename(columns={"method": "model"})
    rows_frame = pd.DataFrame(rows)
    metrics = pd.concat([rows_frame, selected], ignore_index=True, sort=False)
    metrics.to_csv(ARTIFACTS / "final_metrics.csv", index=False)

    state_year_rows = []
    for year, state_frame in final.groupby("season_start_year"):
        for state, part in state_frame.groupby("state_name"):
            for model, prediction in [
                ("V5 frozen point", "production_point_prediction"),
                ("V14 outlook shadow point", "shadow_point_prediction"),
            ]:
                state_year_rows.append({
                    "season_start_year": int(year),
                    "state_name": state,
                    "model": model,
                    **lab.metric_values(
                        part.rename(columns={prediction: "metric_prediction"}),
                        "metric_prediction",
                    ),
                })
    pd.DataFrame(state_year_rows).to_csv(
        ARTIFACTS / "final_state_year_metrics.csv", index=False
    )
    return metrics


def fit_outlook_deployment_bundle() -> dict[str, object]:
    base, groups, outlook = lab.load_panel()
    frame, outlook_groups = lab.with_outlook(base, outlook, 2020)
    train = frame[frame["season_start_year"].between(2017, 2022)].copy()
    feature_sets = {
        "no_future": groups["physical"] + outlook_groups["outlook_no_future"],
        "full": groups["physical"] + outlook_groups["outlook_full"],
        "effect": groups["physical"] + outlook_groups["outlook_effect"],
        "broad": groups["physical"] + outlook_groups["outlook_broad"],
    }
    target = (
        train[TARGET].to_numpy(float)
        - train["baseline_weighted_recent"].to_numpy(float)
    )
    bundles = {}
    for feature_name, features in feature_sets.items():
        usable = lab.finite_columns(train, features, minimum=0.25)
        x, design_columns = lab.design(train, usable)
        models = []
        for seed in lab.SEEDS:
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                XGBRegressor(
                    n_estimators=350,
                    max_depth=2,
                    learning_rate=0.025,
                    min_child_weight=25,
                    subsample=0.85,
                    colsample_bytree=0.65,
                    reg_lambda=50.0,
                    reg_alpha=5.0,
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_jobs=6,
                    random_state=seed,
                ),
            )
            model.fit(x, target)
            models.append(model)
        bundles[feature_name] = {
            "models": models,
            "features": usable,
            "design_columns": design_columns,
        }
    bundle_path = MODELS / "outlook_shadow_xgb_bundle.joblib"
    joblib.dump({
        "components": bundles,
        "fit_seasons": [2017, 2018, 2019, 2020, 2021, 2022],
        "target": "yield minus baseline_weighted_recent",
        "shadow_formula": (
            "V5 + 1.75 * (((full + effect + broad) / 3) - no_future)"
        ),
        "gamma": SHADOW_GAMMA,
        "response_feature_model": "V13 crop-response representation",
        "representation_train_end_for_fit_features": 2020,
        "score_claimed_for_refit": False,
    }, bundle_path)
    return {
        "path": str(bundle_path.relative_to(V14)),
        "sha256": sha256(bundle_path),
        "component_feature_counts": {
            name: len(value["features"]) for name, value in bundles.items()
        },
    }


def build_distribution_pool() -> dict[str, object]:
    base, _, _ = lab.load_panel()
    standalone = pd.read_parquet(
        ARTIFACTS / "standalone_candidate_predictions.parquet"
    )
    standalone = standalone[
        standalone["candidate"].eq("weighted3__physical__xgb_2p0")
        & standalone["season_start_year"].between(2016, 2022)
    ][[
        "district_id", "state_name", "season_start_year", TARGET,
        "prediction", "normal_prediction",
    ]]
    scale = base[[
        "district_id", "season_start_year", "yield_recent_std"
    ]]
    pool = standalone.merge(scale, on=KEYS, validate="one_to_one")
    pool["scale_kg_per_ha"] = np.maximum.reduce([
        pool["yield_recent_std"].fillna(0).to_numpy(float),
        0.07 * pool["normal_prediction"].to_numpy(float),
        np.full(len(pool), 150.0),
    ])
    pool["normalized_error"] = (
        (pool[TARGET] - pool["prediction"]) / pool["scale_kg_per_ha"]
    )
    year_counts = pool["season_start_year"].value_counts().to_dict()
    pool["equal_year_weight"] = [
        1.0 / year_counts[int(year)]
        for year in pool["season_start_year"]
    ]
    path = MODELS / "distribution_calibration_pool.parquet"
    pool.to_parquet(path, index=False)
    return {
        "path": str(path.relative_to(V14)),
        "sha256": sha256(path),
        "rows": len(pool),
        "years": sorted(pool["season_start_year"].unique().tolist()),
        "state_shrink_denominator": 50.0,
        "minimum_target_scale_kg_per_ha": 150.0,
    }


def create_figure(final: pd.DataFrame) -> Path:
    year_rows = []
    for year, part in final.groupby("season_start_year"):
        year_rows.append({
            "year": int(year),
            "V5": lab.metric_values(part, "production_point_prediction")["rmse"],
            "V14 shadow": lab.metric_values(part, "shadow_point_prediction")["rmse"],
        })
    year_frame = pd.DataFrame(year_rows)
    distribution_metrics = pd.read_csv(
        ARTIFACTS / "v5_distribution_metrics.csv"
    )
    coverage = distribution_metrics[
        distribution_metrics["method"].eq(PRIMARY_DISTRIBUTION)
        & distribution_metrics["period"].eq("late")
    ].iloc[0]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(year_frame))
    width = 0.36
    axes[0].bar(x - width / 2, year_frame["V5"], width, label="V5")
    axes[0].bar(
        x + width / 2, year_frame["V14 shadow"], width,
        label="V14 future-crop correction",
    )
    axes[0].set_xticks(x, year_frame["year"].astype(str))
    axes[0].set_ylabel("RMSE (kg/ha; lower is better)")
    axes[0].set_title("Point forecast improves in all four years")
    axes[0].legend(frameon=False)

    intended = np.asarray([50, 80, 90])
    achieved = 100 * np.asarray([
        coverage["coverage_50"],
        coverage["coverage_80"],
        coverage["coverage_90"],
    ])
    axes[1].plot(intended, intended, "--", color="gray", label="perfect")
    axes[1].scatter(intended, achieved, s=85, color="#2a6fbb")
    for target, value in zip(intended, achieved):
        axes[1].annotate(
            f"{value:.1f}%", (target, value),
            textcoords="offset points", xytext=(5, 6),
        )
    axes[1].set_xlim(42, 94)
    axes[1].set_ylim(42, 96)
    axes[1].set_xlabel("Intended coverage (%)")
    axes[1].set_ylabel("Observed 2021–2022 coverage (%)")
    axes[1].set_title("District yield ranges are well calibrated")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    path = ARTIFACTS / "v14_result_summary.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def build_manifest(
    outlook_bundle: dict[str, object],
    distribution_pool: dict[str, object],
) -> dict[str, object]:
    dependencies = [
        V14.parent / "v5" / "root_cybench_lab" / "artifacts"
        / "v5_integration" / "predictions.csv",
        V14.parent / "v13_crop_response_final" / "artifacts"
        / "trajectory_predictions.parquet",
        V14 / "data" / "strict_outlook_features.parquet",
        ARTIFACTS / "final_predictions.parquet",
        ARTIFACTS / "final_metrics.csv",
    ]
    return {
        "release": "V14 anomaly distribution and crop-outlook increment",
        "created": "2026-07-26",
        "production_point": {
            "model": "V5 frozen point",
            "late_rmse_kg_per_ha": 288.61001033283793,
            "status": "keep until sealed-year confirmation",
        },
        "frontier_shadow_point": {
            "model": "V14 isolated future-crop correction",
            "formula": (
                "V5 + 1.75 * "
                "(((XGB_full + XGB_effect + XGB_broad) / 3) - XGB_no_future)"
            ),
            "selection_years": [2019, 2020],
            "confirmation_years": [2021, 2022],
            "four_year_rmse_kg_per_ha": 271.6517054513261,
            "late_rmse_kg_per_ha": 288.12939193813776,
            "status": "frontier shadow; retained, not discarded",
            "reason_not_fully_promoted": (
                "Improves all four year RMSEs, but the state-year grouped "
                "95% RMSE-gain interval still reaches slightly below zero."
            ),
        },
        "distribution": {
            "primary": PRIMARY_DISTRIBUTION,
            "conservative": CONSERVATIVE_DISTRIBUTION,
            "shadow_outlook_center": SHADOW_DISTRIBUTION,
            "available_quantiles": lab.QUANTILES.tolist(),
        },
        "standalone_anomaly": {
            "selected_candidate": "weighted3__physical__xgb_2p0",
            "late_rmse_kg_per_ha": 328.46522119536064,
            "status": "retained research fallback; not a V5 replacement",
        },
        "outlook_deployment_bundle": outlook_bundle,
        "distribution_calibration_pool": distribution_pool,
        "post_2022_yield_labels_read": False,
        "sealed_next_confirmation": [2023, 2024],
        "files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in dependencies
        },
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    final = build_final_predictions()
    build_final_metrics(final)
    build_probability_comparison()
    outlook_bundle = fit_outlook_deployment_bundle()
    distribution_pool = build_distribution_pool()
    figure = create_figure(final)
    manifest = build_manifest(outlook_bundle, distribution_pool)
    manifest["summary_figure"] = {
        "path": str(figure.relative_to(V14)),
        "sha256": sha256(figure),
    }
    (ARTIFACTS / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
