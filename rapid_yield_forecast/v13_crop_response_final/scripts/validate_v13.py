#!/usr/bin/env python3
"""Validate V13 artifacts, promotion decisions, and saved deployment models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


V13 = Path(__file__).resolve().parents[1]
OUT = V13 / "artifacts"
MODELS = V13 / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: dict[str, object] = {}
    final = pd.read_parquet(OUT / "final_predictions.parquet")
    assert len(final) == 119 * 4 * 3
    assert not final.duplicated(["district_id", "season_start_year", "clock"]).any()
    assert sorted(final["season_start_year"].unique().tolist()) == [2019, 2020, 2021, 2022]
    assert sorted(final["clock"].unique().tolist()) == ["feb15", "jan15", "mar05"]
    assert final["actual"].notna().all() and final["prediction"].notna().all()
    assert final["increase_probability"].between(0, 1).all()
    assert final["severe_probability"].between(0, 1).all()
    checks["final_prediction_rows"] = len(final)

    late = final[final["period"].eq("late")]
    assert late["lo80_integrated"].notna().all()
    assert late["hi80_integrated"].notna().all()
    assert (late["hi80_integrated"] > late["lo80_integrated"]).all()
    checks["late_interval_rows"] = len(late)

    point = pd.read_csv(OUT / "point_selected_metrics.csv")
    assert not point["promotion_pass"].any()
    assert np.allclose(final["prediction"], final["anchor"])
    checks["point_promoted_clocks"] = []

    direction = pd.read_csv(OUT / "direction_selected_metrics.csv")
    promoted = sorted(direction.loc[direction["promotion_pass"], "clock"].unique().tolist())
    assert promoted == ["mar05"]
    march_late = direction[
        direction["clock"].eq("mar05") & direction["period"].eq("late")
    ].iloc[0]
    assert march_late["auc"] > march_late["baseline_auc"]
    assert march_late["brier"] < march_late["baseline_brier"]
    assert march_late["p025"] > 0
    assert abs(float(march_late["v13_weight"]) - 0.15) < 1e-10
    checks["direction_promoted_clocks"] = promoted

    non_march = final[~final["clock"].eq("mar05")]
    assert np.allclose(
        non_march["increase_probability"],
        non_march["increase_baseline_probability"],
    )
    march = final[final["clock"].eq("mar05")]
    assert not np.allclose(
        march["increase_probability"],
        march["increase_baseline_probability"],
    )

    transition_manifest = json.loads((OUT / "transition_manifest.json").read_text())
    assert transition_manifest["rows"] == 1428
    assert transition_manifest["invalid_psri_cells_removed"] == 96
    assert transition_manifest["yield_used_in_response_pretraining"] is False
    assert transition_manifest["future_satellite_used_as_input"] is False
    checks["invalid_psri_cells_removed"] = 96

    trajectory = pd.read_csv(OUT / "trajectory_uncertainty.csv")
    dev_future = trajectory[
        trajectory["period"].eq("development")
        & trajectory["comparison"].eq("no_future_minus_full")
    ].iloc[0]
    late_future = trajectory[
        trajectory["period"].eq("late")
        & trajectory["comparison"].eq("no_future_minus_full")
    ].iloc[0]
    assert dev_future["p025"] > 0
    assert late_future["p025"] < 0 < late_future["p975"]

    policy = json.loads((OUT / "final_policy.json").read_text())
    assert policy["future_weather_crop_trajectory_point_contract_pass"] is True
    assert policy["future_weather_crop_trajectory_promoted"] is False
    assert policy["post_2022_yield_labels_read"] is False
    assert policy["increase_probability"]["mar05"] == "V13 crop-response blend"
    checks["future_weather_strictly_promoted"] = False

    deployment = json.loads((OUT / "deployment_manifest.json").read_text())
    assert deployment["blend_rule"].startswith(
        "March p(increase) = 0.85 * V12"
    )
    assert len(deployment["tabular_feature_order"]) == 108

    model_paths = sorted(MODELS.glob("*"))
    assert len(model_paths) == 5
    for path in model_paths:
        if path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert payload["train_end"] == 2022
            assert payload["uses_yield_labels"] is False
            assert len(payload["state_dict"]) > 0
        elif path.suffix == ".joblib":
            model = joblib.load(path)
            assert "model" in model.named_steps
        else:
            raise AssertionError(f"Unexpected model artifact: {path}")
    checks["deployment_model_files"] = [path.name for path in model_paths]

    required_docs = [
        V13 / "ARCHITECTURE_PLAN.md",
        V13 / "RESULT.md",
        V13 / "METHODOLOGY.md",
        V13 / "AUDIT.md",
        V13 / "README.md",
        V13 / "RUNBOOK.md",
    ]
    assert all(path.exists() and path.stat().st_size > 200 for path in required_docs)
    checks["documentation_files"] = [path.name for path in required_docs]

    key_artifacts = [
        OUT / "final_predictions.parquet",
        OUT / "final_metrics.csv",
        OUT / "final_policy.json",
        OUT / "deployment_manifest.json",
        OUT / "trajectory_metrics.csv",
        OUT / "trajectory_uncertainty.csv",
        OUT / "direction_selected_metrics.csv",
        OUT / "point_selected_metrics.csv",
        *model_paths,
    ]
    validation = {
        "status": "pass",
        "checks": checks,
        "sha256": {
            str(path.relative_to(V13)): sha256(path)
            for path in key_artifacts
        },
    }
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

