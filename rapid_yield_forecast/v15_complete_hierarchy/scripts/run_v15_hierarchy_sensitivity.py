#!/usr/bin/env python3
"""Repeat the hierarchy with stable fixed normals instead of a rolling selector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


V15 = Path(__file__).resolve().parents[1]
ROOT = V15.parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACTS = V15 / "artifacts"
DATA = V15 / "data"

from rapid_yield_forecast.v15_complete_hierarchy.scripts import run_v15_hierarchy as h  # noqa: E402


NORMALS = ("weighted3", "xgb1", "extra")


def main() -> None:
    base = pd.read_parquet(h.BASE_PATH)
    normal = pd.read_parquet(
        ARTIFACTS / "normal_candidate_predictions.parquet"
    )
    encoder = pd.read_parquet(h.ENCODER_PATH)
    predictions = []
    audits = []
    panels = []
    for normal_name in NORMALS:
        selected = normal[
            normal["normal_candidate"].eq(normal_name)
        ].copy()
        selected["selected_normal_candidate"] = normal_name
        selected["actual_log_anomaly"] = np.log(
            selected[h.TARGET] / selected["normal_prediction"]
        )
        panel = base.merge(
            selected[[
                "district_id", "season_start_year", "normal_prediction",
                "selected_normal_candidate", "actual_log_anomaly",
            ]],
            on=["district_id", "season_start_year"], validate="one_to_one",
        )
        panel = panel[panel["actual_log_anomaly"].notna()].copy()
        panel = h.add_state_targets(panel)
        panel["normal_source"] = normal_name
        panels.append(panel)
        groups = h.base_feature_groups(panel)
        pred, audit = h.hierarchy_predictions(panel, encoder, groups)
        pred["normal_source"] = normal_name
        pred["candidate"] = normal_name + "__" + pred["candidate"]
        audit["normal_source"] = normal_name
        predictions.append(pred)
        audits.append(audit)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_parquet(
        ARTIFACTS / "hierarchy_sensitivity_predictions.parquet", index=False
    )
    pd.concat(audits, ignore_index=True).to_csv(
        ARTIFACTS / "hierarchy_sensitivity_training_audit.csv", index=False
    )
    metrics = h.candidate_metrics(prediction_frame)
    metrics.to_csv(
        ARTIFACTS / "hierarchy_sensitivity_metrics.csv", index=False
    )
    grid, selected_metrics, selected_predictions = h.blend_with_v5(
        prediction_frame
    )
    grid.to_csv(
        ARTIFACTS / "hierarchy_sensitivity_v5_grid.csv", index=False
    )
    selected_metrics.to_csv(
        ARTIFACTS / "hierarchy_sensitivity_v5_metrics.csv", index=False
    )
    selected_predictions.to_parquet(
        ARTIFACTS / "hierarchy_sensitivity_v5_predictions.parquet", index=False
    )
    best_dev = (
        metrics[metrics["period"].eq("development")]
        .sort_values(["selection_score", "rmse"]).iloc[0].to_dict()
    )
    summary = {
        "normals": list(NORMALS),
        "candidates": int(prediction_frame["candidate"].nunique()),
        "best_development": best_dev,
        "v5_blend": selected_metrics.to_dict("records"),
        "post_2022_yield_labels_read": False,
    }
    with (ARTIFACTS / "hierarchy_sensitivity_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
