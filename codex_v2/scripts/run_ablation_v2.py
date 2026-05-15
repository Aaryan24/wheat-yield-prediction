#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Allow running from repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codex_v2.scripts.train_v2 import run_training_job
from codex_v2.scripts.export_reports_v2 import compile_all_runs


DEFAULT_OP_DATES = [
    "12-05",
    "12-15",
    "12-25",
    "01-04",
    "01-14",
    "01-24",
    "02-05",
    "02-15",
    "02-25",
    "03-05",
]


ABLATIONS: Dict[str, Dict[str, object]] = {
    "A0": {
        "mode": "per_date",
        "target_mode": "raw",
        "apply_sat_mask_fix": False,
        "use_engineered_weather": False,
        "use_engineered_satellite": False,
        "use_missingness_indicators": False,
        "fusion_mode": "concat_gate",
    },
    "A1": {
        "mode": "per_date",
        "target_mode": "raw",
        "apply_sat_mask_fix": True,
        "use_engineered_weather": False,
        "use_engineered_satellite": False,
        "use_missingness_indicators": False,
        "fusion_mode": "concat_gate",
    },
    "A2": {
        "mode": "per_date",
        "target_mode": "district_demeaned",
        "apply_sat_mask_fix": True,
        "use_engineered_weather": False,
        "use_engineered_satellite": False,
        "use_missingness_indicators": False,
        "fusion_mode": "concat_gate",
    },
    "B0": {
        "mode": "shared",
        "target_mode": "raw",
        "apply_sat_mask_fix": False,
        "use_engineered_weather": False,
        "use_engineered_satellite": False,
        "use_missingness_indicators": False,
        "fusion_mode": "concat_gate",
    },
    "B1": {
        "mode": "shared",
        "target_mode": "raw",
        "apply_sat_mask_fix": True,
        "use_engineered_weather": False,
        "use_engineered_satellite": False,
        "use_missingness_indicators": False,
        "fusion_mode": "concat_gate",
    },
    "B2": {
        "mode": "shared",
        "target_mode": "district_demeaned",
        "apply_sat_mask_fix": True,
        "use_engineered_weather": False,
        "use_engineered_satellite": False,
        "use_missingness_indicators": False,
        "fusion_mode": "concat_gate",
    },
    "B3": {
        "mode": "shared",
        "target_mode": "district_demeaned",
        "apply_sat_mask_fix": True,
        "use_engineered_weather": True,
        "use_engineered_satellite": True,
        "use_missingness_indicators": True,
        "fusion_mode": "concat_gate",
    },
    "B4": {
        "mode": "shared",
        "target_mode": "district_demeaned",
        "apply_sat_mask_fix": True,
        "use_engineered_weather": True,
        "use_engineered_satellite": True,
        "use_missingness_indicators": True,
        "fusion_mode": "cross_attention",
    },
}


def _resolve_ablation_set(ablation_set: str) -> List[str]:
    key = str(ablation_set).strip().lower()
    if key == "full":
        return ["A0", "A1", "A2", "B0", "B1", "B2", "B3", "B4"]
    if key == "quick":
        return ["A0", "B3"]

    picked = [x.strip().upper() for x in str(ablation_set).split(",") if x.strip()]
    bad = [x for x in picked if x not in ABLATIONS]
    if bad:
        raise ValueError(f"Unknown ablation ids: {bad}. Valid: {sorted(ABLATIONS.keys())}")
    return picked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Codex V2 ablation matrix.")
    parser.add_argument("--ablation-set", type=str, required=True, help="full|quick|A0,A1,...")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", required=True)
    parser.add_argument("--operational-dates", nargs="+", default=None)
    parser.add_argument(
        "--opdate-profile",
        type=str,
        choices=["manual", "five_day_dec1_apr30"],
        default="manual",
    )
    parser.add_argument("--allow-manual-opdates-override", action="store_true")
    parser.add_argument("--enable-e6", action="store_true")
    parser.add_argument("--enable-e7", action="store_true")
    parser.add_argument("--climate-normals-temp-csv", type=str, default=None)
    parser.add_argument("--climate-normals-rain-csv", type=str, default=None)
    parser.add_argument("--out-root", type=str, required=True)

    parser.add_argument("--config-data", type=str, default="codex_v2/configs/data_v2.yaml")
    parser.add_argument("--config-model", type=str, default="codex_v2/configs/model_3m.yaml")
    parser.add_argument("--config-train-shared", type=str, default="codex_v2/configs/train_shared.yaml")
    parser.add_argument("--config-train-per-date", type=str, default="codex_v2/configs/train_per_date.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablations = _resolve_ablation_set(args.ablation_set)
    op_dates = [str(x) for x in (args.operational_dates or DEFAULT_OP_DATES)]
    if str(args.opdate_profile).strip().lower() != "manual" and not args.operational_dates:
        op_dates = []
    if bool(args.enable_e6) and (not args.climate_normals_temp_csv or not args.climate_normals_rain_csv):
        raise RuntimeError(
            "--enable-e6 requires both --climate-normals-temp-csv and --climate-normals-rain-csv"
        )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    registry_rows: List[dict] = []
    total = len(ablations) * len(args.horizons) * len(args.seeds)
    run_i = 0

    for ab_id in ablations:
        ab_cfg = ABLATIONS[ab_id]
        mode = str(ab_cfg["mode"])

        train_cfg_path = Path(args.config_train_shared if mode == "shared" else args.config_train_per_date)

        for horizon in args.horizons:
            for seed in args.seeds:
                run_i += 1
                run_name = f"{ab_id}_{mode}_h{int(horizon)}_s{int(seed)}"
                run_dir = out_root / run_name
                print(f"[{run_i}/{total}] {run_name}", flush=True)

                result = run_training_job(
                    mode=mode,
                    target_mode=str(ab_cfg["target_mode"]),
                    horizon_days=int(horizon),
                    operational_dates=op_dates,
                    opdate_profile=str(args.opdate_profile),
                    allow_manual_opdates_override=bool(args.allow_manual_opdates_override),
                    seed=int(seed),
                    config_data=Path(args.config_data),
                    config_model=Path(args.config_model),
                    config_train=train_cfg_path,
                    out_dir=run_dir,
                    apply_sat_mask_fix=bool(ab_cfg["apply_sat_mask_fix"]),
                    use_engineered_weather=bool(ab_cfg["use_engineered_weather"]),
                    use_engineered_satellite=bool(ab_cfg["use_engineered_satellite"]),
                    use_missingness_indicators=bool(ab_cfg["use_missingness_indicators"]),
                    enable_e6=bool(args.enable_e6),
                    enable_e7=bool(args.enable_e7),
                    climate_normals_temp_csv=Path(args.climate_normals_temp_csv) if args.climate_normals_temp_csv else None,
                    climate_normals_rain_csv=Path(args.climate_normals_rain_csv) if args.climate_normals_rain_csv else None,
                    fusion_mode=str(ab_cfg["fusion_mode"]),
                    run_name=run_name,
                )

                row = {
                    "ablation_id": ab_id,
                    "run_name": run_name,
                    "mode": mode,
                    "target_mode": str(ab_cfg["target_mode"]),
                    "horizon_days": int(horizon),
                    "seed": int(seed),
                    "apply_sat_mask_fix": bool(ab_cfg["apply_sat_mask_fix"]),
                    "use_engineered_weather": bool(ab_cfg["use_engineered_weather"]),
                    "use_engineered_satellite": bool(ab_cfg["use_engineered_satellite"]),
                    "use_missingness_indicators": bool(ab_cfg["use_missingness_indicators"]),
                    "opdate_profile": str(args.opdate_profile),
                    "enable_e6": bool(args.enable_e6),
                    "enable_e7": bool(args.enable_e7),
                    "fusion_mode": str(ab_cfg["fusion_mode"]),
                    "result_out_dir": str(result["out_dir"]),
                }
                registry_rows.append(row)

                # Print concise run-final metrics only.
                summary_path = Path(result["out_dir"]) / "metrics_summary_seeded.csv"
                if summary_path.exists():
                    sdf = pd.read_csv(summary_path)
                    # Prefer mean_over_opdates rows for comparability.
                    def _pick(split_name: str) -> pd.Series | None:
                        sub = sdf[
                            (sdf["split"] == split_name)
                            & (sdf["summary_level"] == "mean_over_opdates")
                        ]
                        if sub.empty:
                            sub = sdf[sdf["split"] == split_name]
                        if sub.empty:
                            return None
                        return sub.iloc[0]

                    val_row = _pick("val")
                    test_row = _pick("test")
                    msg = [f"{run_name}"]
                    if val_row is not None:
                        msg.append(
                            f"val_rmse={float(val_row['rmse']):.3f}, val_r2={float(val_row['r2']):.3f}"
                        )
                    if test_row is not None:
                        msg.append(
                            f"test_rmse={float(test_row['rmse']):.3f}, test_r2={float(test_row['r2']):.3f}"
                        )
                    print(" | ".join(msg), flush=True)

    registry = pd.DataFrame(registry_rows)
    registry_path = out_root / "ablation_registry.csv"
    registry.to_csv(registry_path, index=False)

    compiled = compile_all_runs(out_root=out_root, write_report=True)

    print(json.dumps({
        "registry": str(registry_path),
        "compiled_csv": str(compiled["csv_path"]),
        "compiled_md": str(compiled["md_path"]),
        "runs": int(len(registry_rows)),
    }, indent=2))


if __name__ == "__main__":
    main()
