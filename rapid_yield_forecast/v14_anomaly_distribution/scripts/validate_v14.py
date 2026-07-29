#!/usr/bin/env python3
"""Validate V14 row coverage, timing, metrics, distributions, and bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


V14 = Path(__file__).resolve().parents[1]
ROOT = V14.parents[1]
ARTIFACTS = V14 / "artifacts"
MODELS = V14 / "models"
TARGET = "yield_kg_per_ha"
YEARS = [2019, 2020, 2021, 2022]
QUANTILES = np.round(np.arange(0.05, 1.0, 0.05), 2)
PRIMARY_DISTRIBUTION = "history_shape__scaled_state_equal_year__w1.00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(actual: pd.Series, prediction: pd.Series) -> float:
    return float(np.sqrt(np.mean(
        (prediction.to_numpy(float) - actual.to_numpy(float)) ** 2
    )))


def check(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    checks: list[str] = []
    required = [
        ARTIFACTS / "final_predictions.parquet",
        ARTIFACTS / "final_metrics.csv",
        ARTIFACTS / "release_manifest.json",
        ARTIFACTS / "probability_model_comparison.csv",
        ARTIFACTS / "v14_result_summary.png",
        MODELS / "outlook_shadow_xgb_bundle.joblib",
        MODELS / "distribution_calibration_pool.parquet",
        MODELS / "v5_distribution_recipe.json",
        MODELS / "standalone_anomaly_model.joblib",
    ]
    for path in required:
        check(path.exists() and path.stat().st_size > 0, f"exists: {path.name}", checks)

    final = pd.read_parquet(ARTIFACTS / "final_predictions.parquet")
    check(len(final) == 476, "final predictions have 476 rows", checks)
    check(
        final.groupby("season_start_year").size().eq(119).all(),
        "each forecast year has 119 districts",
        checks,
    )
    check(
        sorted(final["season_start_year"].unique().tolist()) == YEARS,
        "final predictions contain exactly 2019-2022",
        checks,
    )
    check(
        not final.duplicated(["district_id", "season_start_year"]).any(),
        "district-year keys are unique",
        checks,
    )
    check(
        final[TARGET].notna().all() and int(final["season_start_year"].max()) == 2022,
        "no post-2022 yield label is present",
        checks,
    )
    expected_shadow = (
        final["production_point_prediction"]
        + 1.75 * final["raw_outlook_increment"]
    )
    check(
        np.allclose(expected_shadow, final["shadow_point_prediction"], atol=1e-7),
        "shadow point exactly follows the locked correction formula",
        checks,
    )

    year_gains = []
    for year, part in final.groupby("season_start_year"):
        base = rmse(part[TARGET], part["production_point_prediction"])
        shadow = rmse(part[TARGET], part["shadow_point_prediction"])
        year_gains.append(base - shadow)
    check(
        all(gain > 0 for gain in year_gains),
        "shadow point improves RMSE in all four forecast years",
        checks,
    )
    late = final[final["season_start_year"].isin([2021, 2022])]
    check(
        abs(rmse(late[TARGET], late["production_point_prediction"]) - 288.61001033283793)
        < 1e-7,
        "V5 late RMSE recomputes to 288.610010",
        checks,
    )
    check(
        abs(rmse(late[TARGET], late["shadow_point_prediction"]) - 288.12939193813776)
        < 1e-7,
        "V14 shadow late RMSE recomputes to 288.129392",
        checks,
    )

    q_columns = [f"q{int(q * 100):02d}" for q in QUANTILES]
    q_values = final[q_columns].to_numpy(float)
    check(np.isfinite(q_values).all(), "all primary quantiles are finite", checks)
    check(
        (np.diff(q_values, axis=1) >= -1e-8).all(),
        "all district quantiles are monotonic",
        checks,
    )
    check(
        ((q_values >= 500) & (q_values <= 7000)).all(),
        "all district quantiles remain inside physical output bounds",
        checks,
    )
    check(
        final[["probability_rise", "probability_severe_drop"]]
        .apply(lambda x: x.between(0, 1).all()).all(),
        "all primary event probabilities lie in [0, 1]",
        checks,
    )

    dist_metrics = pd.read_csv(ARTIFACTS / "v5_distribution_metrics.csv")
    primary_late = dist_metrics[
        dist_metrics["method"].eq(PRIMARY_DISTRIBUTION)
        & dist_metrics["period"].eq("late")
    ].iloc[0]
    check(
        0.75 <= primary_late["coverage_80"] <= 0.85,
        "primary late 80% coverage is within 75%-85%",
        checks,
    )
    check(
        0.85 <= primary_late["coverage_90"] <= 0.95,
        "primary late 90% coverage is within 85%-95%",
        checks,
    )

    standalone_audit = pd.read_csv(ARTIFACTS / "standalone_fit_audit.csv")
    check(
        (standalone_audit["train_year_max"] < standalone_audit["test_year"]).all(),
        "every standalone fit trains only on earlier yield seasons",
        checks,
    )
    outlook_manifest = json.loads(
        (V14 / "data" / "outlook_manifest.json").read_text()
    )
    check(
        outlook_manifest["yield_labels_used"] is False,
        "crop-response representation uses no yield labels",
        checks,
    )
    check(
        outlook_manifest["later_satellite_used_as_input"] is False,
        "later satellite state is a target, never an input",
        checks,
    )
    check(
        outlook_manifest["district_crossfit_groups"] == 3,
        "outlook training features use three district cross-fit groups",
        checks,
    )
    outlook_audit = pd.read_csv(V14 / "data" / "outlook_training_audit.csv")
    check(
        set(outlook_audit.loc[
            outlook_audit["feature_role"].eq("train_district_crossfit"),
            "held_group",
        ].unique()) == {0, 1, 2},
        "every held district group has cross-fitted outlook features",
        checks,
    )

    bundle = joblib.load(MODELS / "outlook_shadow_xgb_bundle.joblib")
    check(
        set(bundle["components"]) == {"no_future", "full", "effect", "broad"},
        "deployment bundle contains all four matched XGBoost components",
        checks,
    )
    check(
        all(len(component["models"]) == 2 for component in bundle["components"].values()),
        "each deployment XGBoost component contains both locked seeds",
        checks,
    )
    pool = pd.read_parquet(MODELS / "distribution_calibration_pool.parquet")
    check(len(pool) == 833, "distribution calibration pool has 833 OOF rows", checks)
    check(
        sorted(pool["season_start_year"].unique().tolist())
        == list(range(2016, 2023)),
        "distribution pool spans 2016-2022",
        checks,
    )
    check(
        np.isfinite(pool["normalized_error"]).all(),
        "distribution calibration errors are finite",
        checks,
    )

    manifest = json.loads((ARTIFACTS / "release_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        check(sha256(path) == expected, f"hash matches: {relative}", checks)
    check(
        sha256(V14 / manifest["outlook_deployment_bundle"]["path"])
        == manifest["outlook_deployment_bundle"]["sha256"],
        "outlook deployment bundle hash matches",
        checks,
    )
    check(
        sha256(V14 / manifest["distribution_calibration_pool"]["path"])
        == manifest["distribution_calibration_pool"]["sha256"],
        "distribution pool hash matches",
        checks,
    )

    validation = {
        "status": "pass",
        "checks_passed": len(checks),
        "checks": checks,
        "point_year_rmse_gains_kg_per_ha": dict(zip(YEARS, year_gains)),
        "post_2022_yield_labels_read": False,
    }
    (ARTIFACTS / "validation.json").write_text(
        json.dumps(validation, indent=2)
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
