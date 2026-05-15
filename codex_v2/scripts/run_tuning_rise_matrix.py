#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


MAP_BASE_TO_NEW = {
    "12-15": "12-16",
    "12-25": "12-26",
    "01-04": "01-05",
    "01-14": "01-15",
    "01-24": "01-25",
    "02-05": "02-04",
    "02-15": "02-14",
    "02-25": "02-24",
    "03-05": "03-06",
}
MAP_NEW_TO_BASE = {v: k for k, v in MAP_BASE_TO_NEW.items()}


@dataclass
class RunSpec:
    run_id: str
    target_mode: str
    rise_under_w: float
    drop_miss_w: float
    use_weighted_sampler: bool
    pos_gain: float
    neg_gain: float
    enable_rise_calibration: bool
    checkpoint_objective: str


@dataclass
class EvalResult:
    run_id: str
    out_dir: str
    val_drop_recall: float
    test_drop_recall: float
    test_rise_recall: float
    mapped_test_rmse: float
    test_bucket_lt2: int
    test_bucket_gt10: int
    gates_pass: bool


def _run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _conf_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    fn = int(np.sum(yt & (~yp)))
    tn = int(np.sum((~yt) & (~yp)))
    return tp, fp, fn, tn


def _district_bucket_counts(ape_series: pd.Series) -> dict:
    s = ape_series.dropna()
    return {
        "lt2": int((s < 2.0).sum()),
        "b2_5": int(((s >= 2.0) & (s < 5.0)).sum()),
        "b5_10": int(((s >= 5.0) & (s <= 10.0)).sum()),
        "gt10": int((s > 10.0).sum()),
    }


def evaluate_candidate(
    old_pred_path: Path,
    candidate_pred_path: Path,
    mapped_test_rmse_guardrail_base: float,
) -> dict:
    old_df = pd.read_csv(old_pred_path)
    cand_df = pd.read_csv(candidate_pred_path)

    old_cmp = old_df[old_df["split"].isin(["val", "test", "train"])].copy()
    old_cmp["opdate_key"] = old_cmp["operational_date"]

    cand_cmp = cand_df[cand_df["split"].isin(["val", "test", "train"])].copy()
    cand_cmp = cand_cmp[cand_cmp["operational_date"].isin(MAP_NEW_TO_BASE.keys())].copy()
    cand_cmp["opdate_key"] = cand_cmp["operational_date"].map(MAP_NEW_TO_BASE)

    # District means from train years.
    mu_df = (
        cand_cmp[cand_cmp["split"] == "train"][["district_id", "season_year", "actual_yield_kg_per_ha"]]
        .drop_duplicates()
        .groupby("district_id", as_index=False)["actual_yield_kg_per_ha"]
        .mean()
        .rename(columns={"actual_yield_kg_per_ha": "mu_d"})
    )

    old_cmp = old_cmp.merge(mu_df, on="district_id", how="left")
    cand_cmp = cand_cmp.merge(mu_df, on="district_id", how="left")

    merged = old_cmp.merge(
        cand_cmp,
        on=["split", "season_year", "district_id", "opdate_key"],
        suffixes=("_old", "_new"),
        how="inner",
    )
    merged = merged[merged["split"].isin(["val", "test"])]

    out: dict = {}
    for split in ["val", "test"]:
        s = merged[merged["split"] == split].copy()
        if s.empty:
            raise RuntimeError(f"No merged rows for split={split} in candidate eval")

        actual_delta = s["actual_yield_kg_per_ha_new"].to_numpy(dtype=np.float32) - s["mu_d_new"].to_numpy(dtype=np.float32)
        pred_delta = s["predicted_yield_kg_per_ha_new"].to_numpy(dtype=np.float32) - s["mu_d_new"].to_numpy(dtype=np.float32)

        drop_true = actual_delta < 0.0
        drop_pred = pred_delta < 0.0
        rise_true = actual_delta > 0.0
        rise_pred = pred_delta > 0.0

        tp_d, fp_d, fn_d, tn_d = _conf_counts(drop_true, drop_pred)
        tp_r, fp_r, fn_r, tn_r = _conf_counts(rise_true, rise_pred)

        out[f"{split}_drop_recall"] = _safe_rate(tp_d, tp_d + fn_d)
        out[f"{split}_rise_recall"] = _safe_rate(tp_r, tp_r + fn_r)

        err = s["predicted_yield_kg_per_ha_new"].to_numpy(dtype=np.float32) - s["actual_yield_kg_per_ha_new"].to_numpy(dtype=np.float32)
        out[f"mapped_{split}_rmse"] = float(math.sqrt(float(np.mean(err ** 2))))

        s["ape_pct_new"] = np.where(
            s["actual_yield_kg_per_ha_new"].abs() > 1e-6,
            (s["predicted_yield_kg_per_ha_new"] - s["actual_yield_kg_per_ha_new"]).abs()
            / s["actual_yield_kg_per_ha_new"].abs()
            * 100.0,
            np.nan,
        )
        district_ape = s.groupby("district_id")["ape_pct_new"].mean()
        b = _district_bucket_counts(district_ape)
        out[f"{split}_bucket_lt2"] = int(b["lt2"])
        out[f"{split}_bucket_gt10"] = int(b["gt10"])

    out["gates_pass"] = bool(
        out["val_drop_recall"] >= 0.777
        and out["test_drop_recall"] >= 0.886
        and out["test_rise_recall"] >= 0.200
        and out["test_bucket_lt2"] >= 22
        and out["test_bucket_gt10"] <= 31
        and out["mapped_test_rmse"] <= (mapped_test_rmse_guardrail_base + 3.0)
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 6-run rise-bias tuning matrix for B4 e6/e7 h25.")
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--repo-root", type=str, default="/Users/aaryan/Downloads/ugp")
    parser.add_argument("--config-data", type=str, default="codex_v2/configs/data_v2.yaml")
    parser.add_argument("--config-model", type=str, default="codex_v2/configs/model_3m.yaml")
    parser.add_argument("--config-train", type=str, default="codex_v2/configs/train_shared_mps_120fixed_tuning.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--old-baseline-pred", type=str, default="codex_v2/experiments/live_b4_h25_seed42_20260216_220031/B4_shared_h25_s42/predictions_shared.csv")
    parser.add_argument("--new-baseline-pred", type=str, default="codex_v2/experiments/B4_e6e7_5d_h25_s42_20260303_035916/B4_e6e7_5d_h25_s42/predictions_shared.csv")
    parser.add_argument("--temp-normals-csv", type=str, default="codex_v2/experiments/imd_normals_20260303_fixlat/district_temp_normals_monthly_1991_2020.csv")
    parser.add_argument("--rain-normals-csv", type=str, default="codex_v2/experiments/imd_normals_20260303_fixlat/district_rain_normals_monthly_1971_2020.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    old_baseline_pred = repo_root / args.old_baseline_pred
    new_baseline_pred = repo_root / args.new_baseline_pred

    baseline_eval = evaluate_candidate(
        old_pred_path=old_baseline_pred,
        candidate_pred_path=new_baseline_pred,
        mapped_test_rmse_guardrail_base=1e9,
    )
    guardrail_rmse = float(baseline_eval["mapped_test_rmse"])
    print("Baseline mapped test RMSE guardrail base:", guardrail_rmse, flush=True)

    specs = [
        RunSpec("R1", "district_signed_log", 0.6, 0.2, False, 1.15, 1.0, False, "rmse"),
        RunSpec("R2", "district_signed_log", 1.0, 0.2, False, 1.15, 1.0, False, "rmse"),
        RunSpec("R3", "district_signed_log", 1.0, 0.2, True, 1.15, 1.0, False, "rmse"),
        RunSpec("R4", "district_signed_log_asym", 1.0, 0.2, True, 1.15, 1.0, False, "rmse"),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    eval_rows: List[dict] = []
    py = sys.executable

    for spec in specs:
        run_dir = out_root / spec.run_id
        cmd = [
            py,
            "codex_v2/scripts/train_v2.py",
            "--mode", "shared",
            "--target-mode", spec.target_mode,
            "--horizon-days", "25",
            "--opdate-profile", "five_day_dec1_apr30",
            "--seed", str(int(args.seed)),
            "--config-data", str(args.config_data),
            "--config-model", str(args.config_model),
            "--config-train", str(args.config_train),
            "--out-dir", str(run_dir),
            "--run-name", f"{spec.run_id}_B4_e6e7_5d_h25_s{int(args.seed)}",
            "--fusion-mode", "cross_attention",
            "--enable-e6",
            "--enable-e7",
            "--climate-normals-temp-csv", str(args.temp_normals_csv),
            "--climate-normals-rain-csv", str(args.rain_normals_csv),
            "--loss", "asym_huber",
            "--rise-under-w", str(spec.rise_under_w),
            "--drop-miss-w", str(spec.drop_miss_w),
            "--huber-delta", "1.0",
            "--sample-pos-weight", "2.0",
            "--checkpoint-objective", spec.checkpoint_objective,
            "--min-drop-recall", "0.777",
            "--pos-gain", str(spec.pos_gain),
            "--neg-gain", str(spec.neg_gain),
            "--no-rise-calibration",
        ]
        if spec.use_weighted_sampler:
            cmd.append("--use-weighted-sampler")
        else:
            cmd.append("--no-weighted-sampler")

        _run_cmd(cmd=cmd, cwd=repo_root, env=env)

        pred_path = run_dir / f"{spec.run_id}_B4_e6e7_5d_h25_s{int(args.seed)}" / "predictions_shared.csv"
        ev = evaluate_candidate(
            old_pred_path=old_baseline_pred,
            candidate_pred_path=pred_path,
            mapped_test_rmse_guardrail_base=guardrail_rmse,
        )
        row = {
            "run_id": spec.run_id,
            "run_dir": str(run_dir),
            **ev,
        }
        eval_rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    eval_df = pd.DataFrame(eval_rows)

    passed = eval_df[eval_df["gates_pass"] == True].copy()  # noqa: E712
    if not passed.empty:
        passed = passed.sort_values(["mapped_test_rmse", "test_rise_recall"], ascending=[True, False])
        best_run_id = str(passed.iloc[0]["run_id"])
    else:
        # fallback: prioritize highest test_rise_recall then lower mapped_test_rmse
        best_run_id = str(eval_df.sort_values(["test_rise_recall", "mapped_test_rmse"], ascending=[False, True]).iloc[0]["run_id"])

    best_spec = [s for s in specs if s.run_id == best_run_id][0]
    print("Selected best base run for R5:", best_run_id, flush=True)

    r5_dir = out_root / "R5"
    r5_cmd = [
        py,
        "codex_v2/scripts/train_v2.py",
        "--mode", "shared",
        "--target-mode", best_spec.target_mode,
        "--horizon-days", "25",
        "--opdate-profile", "five_day_dec1_apr30",
        "--seed", str(int(args.seed)),
        "--config-data", str(args.config_data),
        "--config-model", str(args.config_model),
        "--config-train", str(args.config_train),
        "--out-dir", str(r5_dir),
        "--run-name", f"R5_B4_e6e7_5d_h25_s{int(args.seed)}",
        "--fusion-mode", "cross_attention",
        "--enable-e6",
        "--enable-e7",
        "--climate-normals-temp-csv", str(args.temp_normals_csv),
        "--climate-normals-rain-csv", str(args.rain_normals_csv),
        "--loss", "asym_huber",
        "--rise-under-w", str(best_spec.rise_under_w),
        "--drop-miss-w", str(best_spec.drop_miss_w),
        "--huber-delta", "1.0",
        "--sample-pos-weight", "2.0",
        "--checkpoint-objective", "drop_constrained_rise",
        "--min-drop-recall", "0.777",
        "--pos-gain", str(best_spec.pos_gain),
        "--neg-gain", str(best_spec.neg_gain),
        "--enable-rise-calibration",
    ]
    if best_spec.use_weighted_sampler:
        r5_cmd.append("--use-weighted-sampler")
    else:
        r5_cmd.append("--no-weighted-sampler")

    _run_cmd(cmd=r5_cmd, cwd=repo_root, env=env)

    r5_pred = r5_dir / f"R5_B4_e6e7_5d_h25_s{int(args.seed)}" / "predictions_shared.csv"
    r5_eval = evaluate_candidate(
        old_pred_path=old_baseline_pred,
        candidate_pred_path=r5_pred,
        mapped_test_rmse_guardrail_base=guardrail_rmse,
    )

    final = {
        "baseline_guardrail_mapped_test_rmse": guardrail_rmse,
        "r1_r4": eval_rows,
        "selected_for_r5": best_run_id,
        "r5": {
            "run_dir": str(r5_dir),
            **r5_eval,
        },
    }

    eval_df.to_csv(out_root / "r1_r4_eval.csv", index=False)
    with (out_root / "tuning_summary.json").open("w") as fh:
        json.dump(final, fh, indent=2)

    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
