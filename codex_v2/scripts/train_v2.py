#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import yaml

# Allow running from repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codex_v2.src.data.build_dataset_v2 import build_dataset_v2
from codex_v2.src.eval.reporting_v2 import write_run_artifacts
from codex_v2.src.training.train_loop_v2 import train_model_v2


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def run_training_job(
    mode: str,
    target_mode: str,
    horizon_days: int,
    operational_dates: Sequence[str],
    seed: int,
    config_data: Path,
    config_model: Path,
    config_train: Path,
    out_dir: Path,
    apply_sat_mask_fix: bool = True,
    use_engineered_weather: bool = True,
    use_engineered_satellite: bool = True,
    use_missingness_indicators: bool = True,
    fusion_mode: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, object]:
    mode = str(mode).strip().lower()
    if mode not in {"shared", "per_date"}:
        raise ValueError(f"Unsupported mode={mode}")

    model_cfg = _load_yaml(config_model)
    train_cfg = _load_yaml(config_train)

    if fusion_mode is not None:
        model_cfg["fusion_mode"] = str(fusion_mode)

    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_name or f"{mode}_h{horizon_days}_s{seed}"

    run_results: List[dict] = []
    run_dirs: List[Path] = []

    op_groups: List[List[str]]
    if mode == "shared":
        op_groups = [list(operational_dates)]
    else:
        op_groups = [[op] for op in operational_dates]

    for idx, op_group in enumerate(op_groups):
        op_key = "all" if mode == "shared" else op_group[0].replace("/", "-")
        sub_name = run_name if mode == "shared" else f"{run_name}_op_{op_key}"
        run_dir = out_dir / sub_name
        run_dirs.append(run_dir)

        bundle = build_dataset_v2(
            data_config_path=config_data,
            mode=mode,
            target_mode=target_mode,
            horizon_days=horizon_days,
            operational_dates=op_group,
            apply_sat_mask_fix=apply_sat_mask_fix,
            use_engineered_weather=use_engineered_weather,
            use_engineered_satellite=use_engineered_satellite,
            use_missingness_indicators=use_missingness_indicators,
        )

        training_output = train_model_v2(
            bundle=bundle,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            seed=seed,
            out_dir=run_dir,
            mode=mode,
            target_mode=target_mode,
            horizon_days=horizon_days,
        )

        resolved_config = {
            "run_name": sub_name,
            "mode": mode,
            "target_mode": target_mode,
            "horizon_days": int(horizon_days),
            "seed": int(seed),
            "operational_dates": [str(x) for x in op_group],
            "flags": {
                "apply_sat_mask_fix": bool(apply_sat_mask_fix),
                "use_engineered_weather": bool(use_engineered_weather),
                "use_engineered_satellite": bool(use_engineered_satellite),
                "use_missingness_indicators": bool(use_missingness_indicators),
                "fusion_mode": str(model_cfg.get("fusion_mode", "concat_gate")),
            },
            "model": model_cfg,
            "train": train_cfg,
            "data": bundle.config_resolved,
            "model_stats": {
                "model_total_params": int(training_output.model_total_params),
                "model_trainable_params": int(training_output.model_trainable_params),
                "epochs_ran": int(training_output.epochs_ran),
                "best_epoch": int(training_output.best_epoch),
                "train_seconds": float(training_output.train_seconds),
            },
        }

        artifact_paths = write_run_artifacts(
            out_dir=run_dir,
            bundle=bundle,
            training_output=training_output,
            mode=mode,
            target_mode=target_mode,
            horizon_days=horizon_days,
            seed=seed,
            run_name=sub_name,
            resolved_config=resolved_config,
        )

        run_results.append(
            {
                "run_name": sub_name,
                "run_dir": str(run_dir),
                "metrics_per_opdate": str(artifact_paths["metrics_per_opdate"]),
                "metrics_summary_seeded": str(artifact_paths["metrics_summary_seeded"]),
                "predictions": str(artifact_paths["predictions"]),
                "feature_coverage_report": str(artifact_paths["feature_coverage_report"]),
                "model_config_resolved": str(artifact_paths["model_config_resolved"]),
            }
        )

    # Top-level consolidated outputs for this command invocation.
    metrics_frames = []
    summary_frames = []
    pred_frames = []
    coverage_frames = []
    log_rows = []
    cfg_rows = []

    for rr in run_results:
        run_dir = Path(rr["run_dir"])
        p_metrics = run_dir / "metrics_per_opdate.csv"
        p_summary = run_dir / "metrics_summary_seeded.csv"
        p_preds = run_dir / f"predictions_{mode}.csv"
        p_cov = run_dir / "feature_coverage_report.csv"
        p_cfg = run_dir / "model_config_resolved.json"
        p_log = run_dir / "training_log.jsonl"

        if p_metrics.exists():
            metrics_frames.append(pd.read_csv(p_metrics))
        if p_summary.exists():
            summary_frames.append(pd.read_csv(p_summary))
        if p_preds.exists():
            pred_frames.append(pd.read_csv(p_preds))
        if p_cov.exists():
            coverage_frames.append(pd.read_csv(p_cov))
        if p_cfg.exists():
            cfg_rows.append(json.loads(p_cfg.read_text()))
        if p_log.exists():
            for line in p_log.read_text().splitlines():
                if line.strip():
                    log_rows.append(json.loads(line))

    metrics_all = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    preds_all = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    cov_all = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()

    metrics_all.to_csv(out_dir / "metrics_per_opdate.csv", index=False)
    summary_all.to_csv(out_dir / "metrics_summary_seeded.csv", index=False)
    preds_all.to_csv(out_dir / f"predictions_{mode}.csv", index=False)
    cov_all.to_csv(out_dir / "feature_coverage_report.csv", index=False)

    sangrur_rows = []
    for rr in run_results:
        p = Path(rr["run_dir"]) / "sangrur_merge_weights.csv"
        if p.exists():
            sangrur_rows.append(pd.read_csv(p))
    if sangrur_rows:
        sangrur_all = pd.concat(sangrur_rows, ignore_index=True).drop_duplicates().sort_values("season_year")
    else:
        sangrur_all = pd.DataFrame(
            columns=["season_year", "sangrur_area_before", "malerkotla_area_added", "weight_sangrur", "weight_malerkotla"]
        )
    sangrur_all.to_csv(out_dir / "sangrur_merge_weights.csv", index=False)

    with (out_dir / "training_log.jsonl").open("w") as fh:
        for row in log_rows:
            fh.write(json.dumps(row) + "\n")

    with (out_dir / "model_config_resolved.json").open("w") as fh:
        json.dump(
            {
                "run_name": run_name,
                "mode": mode,
                "target_mode": target_mode,
                "horizon_days": int(horizon_days),
                "seed": int(seed),
                "operational_dates": [str(x) for x in operational_dates],
                "runs": cfg_rows,
            },
            fh,
            indent=2,
        )

    return {
        "out_dir": str(out_dir),
        "mode": mode,
        "target_mode": target_mode,
        "seed": int(seed),
        "horizon_days": int(horizon_days),
        "run_name": run_name,
        "sub_runs": run_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Codex V2 Transformer+GAT model.")
    parser.add_argument("--mode", choices=["shared", "per_date"], required=True)
    parser.add_argument(
        "--target-mode",
        choices=["raw", "district_demeaned", "district_zscore"],
        required=True,
    )
    parser.add_argument("--horizon-days", type=int, required=True)
    parser.add_argument("--operational-dates", nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config-model", type=str, required=True)
    parser.add_argument("--config-train", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--config-data", type=str, default="codex_v2/configs/data_v2.yaml")
    parser.add_argument("--no-sat-mask-fix", action="store_true")
    parser.add_argument("--no-engineered-weather", action="store_true")
    parser.add_argument("--no-engineered-satellite", action="store_true")
    parser.add_argument("--no-missingness-indicators", action="store_true")
    parser.add_argument("--fusion-mode", choices=["concat_gate", "cross_attention"], default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_training_job(
        mode=args.mode,
        target_mode=args.target_mode,
        horizon_days=int(args.horizon_days),
        operational_dates=[str(x) for x in args.operational_dates],
        seed=int(args.seed),
        config_data=Path(args.config_data),
        config_model=Path(args.config_model),
        config_train=Path(args.config_train),
        out_dir=Path(args.out_dir),
        apply_sat_mask_fix=not bool(args.no_sat_mask_fix),
        use_engineered_weather=not bool(args.no_engineered_weather),
        use_engineered_satellite=not bool(args.no_engineered_satellite),
        use_missingness_indicators=not bool(args.no_missingness_indicators),
        fusion_mode=args.fusion_mode,
        run_name=args.run_name,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
