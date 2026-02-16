from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from codex_v2.src.data.build_dataset_v2 import DatasetBundle
from codex_v2.src.eval.metrics_v2 import regression_metrics
from codex_v2.src.training.train_loop_v2 import TrainingOutput


def _split_lookup(bundle: DatasetBundle) -> np.ndarray:
    out = np.array(["unknown"] * len(bundle.sample_years), dtype=object)
    out[bundle.train_idx] = "train"
    out[bundle.val_idx] = "val"
    out[bundle.test_idx] = "test"
    return out


def build_predictions_df(
    bundle: DatasetBundle,
    training_output: TrainingOutput,
    mode: str,
    target_mode: str,
    horizon_days: int,
    seed: int,
    run_name: str,
) -> pd.DataFrame:
    split_name = _split_lookup(bundle)

    district_ids = bundle.district_df["district_id"].astype(str).to_numpy()
    state_names = bundle.district_df["state_name"].astype(str).to_numpy()
    district_names = bundle.district_df["district_name"].astype(str).to_numpy()

    rows: List[dict] = []
    s_count, n_nodes = training_output.pred_target.shape
    for s in range(s_count):
        for n in range(n_nodes):
            actual_raw = float(bundle.y_raw[s, n])
            pred_raw = float(training_output.pred_raw[s, n])
            rows.append(
                {
                    "run_name": str(run_name),
                    "mode": str(mode),
                    "target_mode": str(target_mode),
                    "horizon_days": int(horizon_days),
                    "seed": int(seed),
                    "split": str(split_name[s]),
                    "season_year": int(bundle.sample_years[s]),
                    "operational_date": str(bundle.sample_operational_dates[s]),
                    "issue_date": str(bundle.sample_issue_dates[s]),
                    "district_id": str(district_ids[n]),
                    "state_name": str(state_names[n]),
                    "district_name": str(district_names[n]),
                    "actual_target": float(bundle.y_target[s, n]),
                    "pred_target": float(training_output.pred_target[s, n]),
                    "actual_yield_kg_per_ha": actual_raw,
                    "predicted_yield_kg_per_ha": pred_raw,
                    "error_kg_per_ha": float(pred_raw - actual_raw),
                    "abs_error_kg_per_ha": float(abs(pred_raw - actual_raw)),
                }
            )
    return pd.DataFrame(rows)


def metrics_per_opdate(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    gcols = ["run_name", "mode", "target_mode", "horizon_days", "seed", "split", "operational_date"]
    for keys, grp in pred_df.groupby(gcols, dropna=False):
        m = regression_metrics(
            grp["actual_yield_kg_per_ha"].to_numpy(dtype=np.float32),
            grp["predicted_yield_kg_per_ha"].to_numpy(dtype=np.float32),
        )
        row = {k: v for k, v in zip(gcols, keys)}
        row.update(
            {
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "mape": float(m["mape"]),
                "r2": float(m["r2"]),
                "n_rows": int(len(grp)),
                "n_years": int(grp["season_year"].nunique()),
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    order = {
        "12-05": 1,
        "12-15": 2,
        "12-25": 3,
        "01-04": 4,
        "01-14": 5,
        "01-24": 6,
        "02-05": 7,
        "02-15": 8,
        "02-25": 9,
        "03-05": 10,
    }
    out["_ord"] = out["operational_date"].map(lambda x: order.get(str(x), 999))
    out = out.sort_values(["split", "_ord"]).drop(columns=["_ord"]).reset_index(drop=True)
    return out


def metrics_summary_seeded(pred_df: pd.DataFrame, metrics_op: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    base_cols = ["run_name", "mode", "target_mode", "horizon_days", "seed", "split"]
    for keys, grp in pred_df.groupby(base_cols, dropna=False):
        m = regression_metrics(
            grp["actual_yield_kg_per_ha"].to_numpy(dtype=np.float32),
            grp["predicted_yield_kg_per_ha"].to_numpy(dtype=np.float32),
        )
        row = {k: v for k, v in zip(base_cols, keys)}
        row.update(
            {
                "summary_level": "global_rows",
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "mape": float(m["mape"]),
                "r2": float(m["r2"]),
                "n_rows": int(len(grp)),
                "n_opdates": int(grp["operational_date"].nunique()),
                "n_years": int(grp["season_year"].nunique()),
            }
        )
        rows.append(row)

    if not metrics_op.empty:
        grp_cols = ["run_name", "mode", "target_mode", "horizon_days", "seed", "split"]
        for keys, grp in metrics_op.groupby(grp_cols, dropna=False):
            row = {k: v for k, v in zip(grp_cols, keys)}
            row.update(
                {
                    "summary_level": "mean_over_opdates",
                    "rmse": float(grp["rmse"].mean()),
                    "mae": float(grp["mae"].mean()),
                    "mape": float(grp["mape"].mean()),
                    "r2": float(grp["r2"].mean()),
                    "n_rows": int(grp["n_rows"].sum()),
                    "n_opdates": int(grp["operational_date"].nunique()),
                    "n_years": int(grp["n_years"].sum()),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def write_run_artifacts(
    out_dir: Path,
    bundle: DatasetBundle,
    training_output: TrainingOutput,
    mode: str,
    target_mode: str,
    horizon_days: int,
    seed: int,
    run_name: str,
    resolved_config: Dict[str, object],
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = build_predictions_df(
        bundle=bundle,
        training_output=training_output,
        mode=mode,
        target_mode=target_mode,
        horizon_days=horizon_days,
        seed=seed,
        run_name=run_name,
    )
    metrics_op = metrics_per_opdate(pred_df)
    metrics_summary = metrics_summary_seeded(pred_df, metrics_op)

    pred_path = out_dir / f"predictions_{mode}.csv"
    metrics_path = out_dir / "metrics_per_opdate.csv"
    summary_path = out_dir / "metrics_summary_seeded.csv"
    coverage_path = out_dir / "feature_coverage_report.csv"
    sangrur_path = out_dir / "sangrur_merge_weights.csv"
    config_path = out_dir / "model_config_resolved.json"

    pred_df.to_csv(pred_path, index=False)
    metrics_op.to_csv(metrics_path, index=False)
    metrics_summary.to_csv(summary_path, index=False)

    if bundle.coverage_report.empty:
        pd.DataFrame(columns=["season_year", "operational_date", "state_name"]).to_csv(coverage_path, index=False)
    else:
        bundle.coverage_report.to_csv(coverage_path, index=False)

    if bundle.sangrur_weights_report.empty:
        pd.DataFrame(
            columns=["season_year", "sangrur_area_before", "malerkotla_area_added", "weight_sangrur", "weight_malerkotla"]
        ).to_csv(sangrur_path, index=False)
    else:
        bundle.sangrur_weights_report.to_csv(sangrur_path, index=False)

    with config_path.open("w") as fh:
        json.dump(resolved_config, fh, indent=2)

    return {
        "predictions": pred_path,
        "metrics_per_opdate": metrics_path,
        "metrics_summary_seeded": summary_path,
        "feature_coverage_report": coverage_path,
        "sangrur_merge_weights": sangrur_path,
        "model_config_resolved": config_path,
    }
