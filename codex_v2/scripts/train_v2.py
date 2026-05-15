#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

# Allow running from repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codex_v2.src.data.build_dataset_v2 import apply_target_transform, build_dataset_v2
from codex_v2.src.data.opdate_profiles_v2 import PROFILE_MANUAL, SUPPORTED_OPDATE_PROFILES, opdates_for_profile
from codex_v2.src.eval.calibration_v2 import apply_rise_bias_calibrator, fit_rise_bias_calibrator
from codex_v2.src.eval.reporting_v2 import write_run_artifacts
from codex_v2.src.training.train_loop_v2 import train_model_v2


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _split_names_array(n_samples: int, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    out = np.array(["unknown"] * int(n_samples), dtype=object)
    out[train_idx] = "train"
    out[val_idx] = "val"
    out[test_idx] = "test"
    return out


def run_training_job(
    mode: str,
    target_mode: str,
    horizon_days: int,
    operational_dates: Optional[Sequence[str]],
    seed: int,
    config_data: Path,
    config_model: Path,
    config_train: Path,
    out_dir: Path,
    opdate_profile: str = PROFILE_MANUAL,
    allow_manual_opdates_override: bool = False,
    apply_sat_mask_fix: bool = True,
    use_engineered_weather: bool = True,
    use_engineered_satellite: bool = True,
    use_missingness_indicators: bool = True,
    enable_e6: bool = False,
    enable_e7: bool = False,
    enable_token_time_features: bool = False,
    climate_normals_temp_csv: Optional[Path] = None,
    climate_normals_rain_csv: Optional[Path] = None,
    signed_log_pos_gain: float = 1.15,
    signed_log_neg_gain: float = 1.0,
    loss: Optional[str] = None,
    rise_under_w: float = 0.8,
    drop_miss_w: float = 0.2,
    huber_delta: float = 1.0,
    use_weighted_sampler: Optional[bool] = None,
    sample_pos_weight: float = 2.0,
    checkpoint_objective: str = "rmse",
    min_drop_recall: float = 0.777,
    enable_rise_calibration: bool = False,
    neutral_eps: Optional[float] = None,
    neutral_weight: float = 0.35,
    class_loss_weight: float = 1.0,
    magnitude_loss_weight: float = 1.0,
    magnitude_huber_delta: float = 0.5,
    fusion_mode: Optional[str] = None,
    run_name: Optional[str] = None,
    state_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    mode = str(mode).strip().lower()
    if mode not in {"shared", "per_date"}:
        raise ValueError(f"Unsupported mode={mode}")

    model_cfg = _load_yaml(config_model)
    train_cfg = _load_yaml(config_train)
    optimization_cfg = train_cfg.setdefault("optimization", {})
    training_cfg = train_cfg.setdefault("training", {})

    if loss is not None:
        optimization_cfg["loss"] = str(loss)
    optimization_cfg["rise_under_w"] = float(rise_under_w)
    optimization_cfg["drop_miss_w"] = float(drop_miss_w)
    optimization_cfg["huber_delta"] = float(huber_delta)
    if use_weighted_sampler is not None:
        training_cfg["use_weighted_sampler"] = bool(use_weighted_sampler)
    training_cfg["sample_pos_weight"] = float(sample_pos_weight)
    training_cfg["checkpoint_objective"] = str(checkpoint_objective)
    training_cfg["min_drop_recall"] = float(min_drop_recall)
    if neutral_eps is not None:
        training_cfg["neutral_eps"] = float(neutral_eps)
    training_cfg["neutral_weight"] = float(neutral_weight)
    training_cfg["class_loss_weight"] = float(class_loss_weight)
    training_cfg["magnitude_loss_weight"] = float(magnitude_loss_weight)
    training_cfg["magnitude_huber_delta"] = float(magnitude_huber_delta)

    if fusion_mode is not None:
        model_cfg["fusion_mode"] = str(fusion_mode)

    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_name or f"{mode}_h{horizon_days}_s{seed}"

    op_profile = str(opdate_profile).strip().lower()
    if op_profile not in SUPPORTED_OPDATE_PROFILES:
        raise ValueError(
            f"Unknown opdate_profile={opdate_profile}. Supported={sorted(SUPPORTED_OPDATE_PROFILES)}"
        )

    if op_profile != PROFILE_MANUAL and not allow_manual_opdates_override:
        effective_operational_dates = opdates_for_profile(op_profile)
    else:
        effective_operational_dates = [str(x) for x in (operational_dates or [])]
        if not effective_operational_dates:
            if op_profile != PROFILE_MANUAL:
                effective_operational_dates = opdates_for_profile(op_profile)
            else:
                data_cfg = _load_yaml(config_data)
                op_cfg = data_cfg.get("operational_dates", {})
                primary = [str(x) for x in op_cfg.get("primary", [])]
                secondary = [str(x) for x in op_cfg.get("secondary", [])]
                effective_operational_dates = primary + secondary

    run_results: List[dict] = []
    run_dirs: List[Path] = []

    op_groups: List[List[str]]
    if mode == "shared":
        op_groups = [list(effective_operational_dates)]
    else:
        if not effective_operational_dates:
            raise RuntimeError("per_date mode requires at least one operational date after resolution.")
        op_groups = [[op] for op in effective_operational_dates]

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
            operational_dates=op_group if op_group else None,
            opdate_profile=op_profile,
            allow_manual_opdates_override=allow_manual_opdates_override,
            apply_sat_mask_fix=apply_sat_mask_fix,
            use_engineered_weather=use_engineered_weather,
            use_engineered_satellite=use_engineered_satellite,
            use_missingness_indicators=use_missingness_indicators,
            enable_e6=enable_e6,
            enable_e7=enable_e7,
            enable_token_time_features=enable_token_time_features,
            climate_normals_temp_csv=climate_normals_temp_csv,
            climate_normals_rain_csv=climate_normals_rain_csv,
            signed_log_pos_gain=float(signed_log_pos_gain),
            signed_log_neg_gain=float(signed_log_neg_gain),
            state_names=state_names,
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

        rise_calibration_meta: Dict[str, object] = {}
        if enable_rise_calibration:
            split_names = _split_names_array(
                n_samples=int(len(bundle.sample_years)),
                train_idx=bundle.train_idx,
                val_idx=bundle.val_idx,
                test_idx=bundle.test_idx,
            )
            rise_calibration_meta = fit_rise_bias_calibrator(
                pred_raw=training_output.pred_raw,
                actual_raw=bundle.y_raw,
                sample_opdates=bundle.sample_operational_dates.tolist(),
                sample_splits=split_names.tolist(),
                target_mean=bundle.target_mean,
            )
            pred_raw_cal = apply_rise_bias_calibrator(
                pred_raw=training_output.pred_raw,
                sample_opdates=bundle.sample_operational_dates.tolist(),
                target_mean=bundle.target_mean,
                calibrator=rise_calibration_meta,
            )
            pred_target_cal = apply_target_transform(
                y_raw=pred_raw_cal,
                target_mode=bundle.target_mode,
                target_mean=bundle.target_mean,
                target_std=bundle.target_std,
                signed_log_pos_gain=float(bundle.signed_log_pos_gain),
                signed_log_neg_gain=float(bundle.signed_log_neg_gain),
            )
            training_output.pred_raw = pred_raw_cal.astype(np.float32)
            training_output.pred_target = pred_target_cal.astype(np.float32)

        resolved_config = {
            "run_name": sub_name,
            "mode": mode,
            "target_mode": target_mode,
            "horizon_days": int(horizon_days),
            "seed": int(seed),
            "operational_dates": [str(x) for x in op_group],
            "state_filter": [str(x) for x in state_names] if state_names else [],
            "flags": {
                "opdate_profile": op_profile,
                "allow_manual_opdates_override": bool(allow_manual_opdates_override),
                "apply_sat_mask_fix": bool(apply_sat_mask_fix),
                "use_engineered_weather": bool(use_engineered_weather),
                "use_engineered_satellite": bool(use_engineered_satellite),
                "use_missingness_indicators": bool(use_missingness_indicators),
                "enable_e6": bool(enable_e6),
                "enable_e7": bool(enable_e7),
                "enable_token_time_features": bool(enable_token_time_features),
                "climate_normals_temp_csv": str(climate_normals_temp_csv) if climate_normals_temp_csv else "",
                "climate_normals_rain_csv": str(climate_normals_rain_csv) if climate_normals_rain_csv else "",
                "signed_log_pos_gain": float(signed_log_pos_gain),
                "signed_log_neg_gain": float(signed_log_neg_gain),
                "loss": str(optimization_cfg.get("loss", "huber")),
                "rise_under_w": float(optimization_cfg.get("rise_under_w", rise_under_w)),
                "drop_miss_w": float(optimization_cfg.get("drop_miss_w", drop_miss_w)),
                "huber_delta": float(optimization_cfg.get("huber_delta", huber_delta)),
                "use_weighted_sampler": bool(training_cfg.get("use_weighted_sampler", False)),
                "sample_pos_weight": float(training_cfg.get("sample_pos_weight", sample_pos_weight)),
                "checkpoint_objective": str(training_cfg.get("checkpoint_objective", checkpoint_objective)),
                "min_drop_recall": float(training_cfg.get("min_drop_recall", min_drop_recall)),
                "neutral_eps": float(training_cfg.get("neutral_eps", -1.0)),
                "neutral_weight": float(training_cfg.get("neutral_weight", neutral_weight)),
                "class_loss_weight": float(training_cfg.get("class_loss_weight", class_loss_weight)),
                "magnitude_loss_weight": float(training_cfg.get("magnitude_loss_weight", magnitude_loss_weight)),
                "magnitude_huber_delta": float(training_cfg.get("magnitude_huber_delta", magnitude_huber_delta)),
                "enable_rise_calibration": bool(enable_rise_calibration),
                "rise_calibration": rise_calibration_meta if rise_calibration_meta else {},
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
        resolved_opdates = [str(x) for x in effective_operational_dates]
        if cfg_rows:
            nested_data = cfg_rows[0].get("data", {}) if isinstance(cfg_rows[0], dict) else {}
            nested_opdates = nested_data.get("operational_dates")
            if isinstance(nested_opdates, list) and nested_opdates:
                resolved_opdates = [str(x) for x in nested_opdates]
        json.dump(
            {
                "run_name": run_name,
                "mode": mode,
                "target_mode": target_mode,
                "horizon_days": int(horizon_days),
                "seed": int(seed),
                "opdate_profile": op_profile,
                "operational_dates": resolved_opdates,
                "state_filter": [str(x) for x in state_names] if state_names else [],
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
        "opdate_profile": op_profile,
        "operational_dates": [str(x) for x in effective_operational_dates],
        "sub_runs": run_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Codex V2 Transformer+GAT model.")
    parser.add_argument("--mode", choices=["shared", "per_date"], required=True)
    parser.add_argument(
        "--target-mode",
        choices=["raw", "district_demeaned", "district_zscore", "district_signed_log", "district_signed_log_asym"],
        required=True,
    )
    parser.add_argument("--horizon-days", type=int, required=True)
    parser.add_argument("--operational-dates", nargs="+", default=None)
    parser.add_argument(
        "--opdate-profile",
        choices=sorted(SUPPORTED_OPDATE_PROFILES),
        default=PROFILE_MANUAL,
    )
    parser.add_argument("--allow-manual-opdates-override", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config-model", type=str, required=True)
    parser.add_argument("--config-train", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--config-data", type=str, default="codex_v2/configs/data_v2.yaml")
    parser.add_argument("--no-sat-mask-fix", action="store_true")
    parser.add_argument("--no-engineered-weather", action="store_true")
    parser.add_argument("--no-engineered-satellite", action="store_true")
    parser.add_argument("--no-missingness-indicators", action="store_true")
    parser.add_argument("--enable-e6", action="store_true")
    parser.add_argument("--enable-e7", action="store_true")
    parser.add_argument("--enable-token-time-features", action="store_true")
    parser.add_argument("--climate-normals-temp-csv", type=str, default=None)
    parser.add_argument("--climate-normals-rain-csv", type=str, default=None)
    parser.add_argument("--loss", choices=["mse", "huber", "mae_mse", "asym_huber"], default=None)
    parser.add_argument("--rise-under-w", type=float, default=0.8)
    parser.add_argument("--drop-miss-w", type=float, default=0.2)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--sample-pos-weight", type=float, default=2.0)
    parser.add_argument("--pos-gain", type=float, default=1.15)
    parser.add_argument("--neg-gain", type=float, default=1.0)
    parser.add_argument("--checkpoint-objective", choices=["rmse", "drop_constrained_rise"], default="rmse")
    parser.add_argument("--min-drop-recall", type=float, default=0.777)
    parser.add_argument("--neutral-eps", type=float, default=None)
    parser.add_argument("--neutral-weight", type=float, default=0.35)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--magnitude-loss-weight", type=float, default=1.0)
    parser.add_argument("--magnitude-huber-delta", type=float, default=0.5)
    parser.add_argument("--use-weighted-sampler", dest="use_weighted_sampler", action="store_true")
    parser.add_argument("--no-weighted-sampler", dest="use_weighted_sampler", action="store_false")
    parser.set_defaults(use_weighted_sampler=None)
    parser.add_argument("--enable-rise-calibration", dest="enable_rise_calibration", action="store_true")
    parser.add_argument("--no-rise-calibration", dest="enable_rise_calibration", action="store_false")
    parser.set_defaults(enable_rise_calibration=False)
    parser.add_argument("--fusion-mode", choices=["concat_gate", "cross_attention"], default=None)
    parser.add_argument("--state-names", nargs="+", default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    if bool(args.enable_e6):
        if not args.climate_normals_temp_csv or not args.climate_normals_rain_csv:
            parser.error(
                "--enable-e6 requires both --climate-normals-temp-csv and --climate-normals-rain-csv."
            )
    return args


def main() -> None:
    args = parse_args()

    result = run_training_job(
        mode=args.mode,
        target_mode=args.target_mode,
        horizon_days=int(args.horizon_days),
        operational_dates=[str(x) for x in args.operational_dates] if args.operational_dates else None,
        opdate_profile=str(args.opdate_profile),
        allow_manual_opdates_override=bool(args.allow_manual_opdates_override),
        seed=int(args.seed),
        config_data=Path(args.config_data),
        config_model=Path(args.config_model),
        config_train=Path(args.config_train),
        out_dir=Path(args.out_dir),
        apply_sat_mask_fix=not bool(args.no_sat_mask_fix),
        use_engineered_weather=not bool(args.no_engineered_weather),
        use_engineered_satellite=not bool(args.no_engineered_satellite),
        use_missingness_indicators=not bool(args.no_missingness_indicators),
        enable_e6=bool(args.enable_e6),
        enable_e7=bool(args.enable_e7),
        enable_token_time_features=bool(args.enable_token_time_features),
        climate_normals_temp_csv=Path(args.climate_normals_temp_csv) if args.climate_normals_temp_csv else None,
        climate_normals_rain_csv=Path(args.climate_normals_rain_csv) if args.climate_normals_rain_csv else None,
        signed_log_pos_gain=float(args.pos_gain),
        signed_log_neg_gain=float(args.neg_gain),
        loss=args.loss,
        rise_under_w=float(args.rise_under_w),
        drop_miss_w=float(args.drop_miss_w),
        huber_delta=float(args.huber_delta),
        use_weighted_sampler=args.use_weighted_sampler,
        sample_pos_weight=float(args.sample_pos_weight),
        checkpoint_objective=str(args.checkpoint_objective),
        min_drop_recall=float(args.min_drop_recall),
        enable_rise_calibration=bool(args.enable_rise_calibration),
        neutral_eps=float(args.neutral_eps) if args.neutral_eps is not None else None,
        neutral_weight=float(args.neutral_weight),
        class_loss_weight=float(args.class_loss_weight),
        magnitude_loss_weight=float(args.magnitude_loss_weight),
        magnitude_huber_delta=float(args.magnitude_huber_delta),
        fusion_mode=args.fusion_mode,
        run_name=args.run_name,
        state_names=[str(x) for x in args.state_names] if args.state_names else None,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
