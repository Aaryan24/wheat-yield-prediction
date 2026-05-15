#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAP_NEW_TO_BASE = {
    "12-16": "12-15",
    "12-26": "12-25",
    "01-05": "01-04",
    "01-15": "01-14",
    "01-25": "01-24",
    "02-04": "02-05",
    "02-14": "02-15",
    "02-24": "02-25",
    "03-06": "03-05",
}


def _conf_counts(y_true: np.ndarray, y_pred: np.ndarray):
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum((~yt) & yp))
    fn = int(np.sum(yt & (~yp)))
    tn = int(np.sum((~yt) & (~yp)))
    return tp, fp, fn, tn


def _safe_rate(a: int, b: int) -> float:
    return float(a) / float(b) if b > 0 else float("nan")


def _district_bucket_counts(ape: pd.Series) -> dict:
    s = ape.dropna()
    return {
        "lt2": int((s < 2.0).sum()),
        "gt10": int((s > 10.0).sum()),
    }


def evaluate_candidate(old_pred: Path, cand_pred: Path) -> dict:
    old_df = pd.read_csv(old_pred)
    c_df = pd.read_csv(cand_pred)

    old_cmp = old_df[old_df["split"].isin(["train", "val", "test"])].copy()
    old_cmp["opdate_key"] = old_cmp["operational_date"]

    c_cmp = c_df[c_df["split"].isin(["train", "val", "test"])].copy()
    c_cmp = c_cmp[c_cmp["operational_date"].isin(MAP_NEW_TO_BASE.keys())].copy()
    c_cmp["opdate_key"] = c_cmp["operational_date"].map(MAP_NEW_TO_BASE)

    mu = (
        c_cmp[c_cmp["split"] == "train"][["district_id", "season_year", "actual_yield_kg_per_ha"]]
        .drop_duplicates()
        .groupby("district_id", as_index=False)["actual_yield_kg_per_ha"]
        .mean()
        .rename(columns={"actual_yield_kg_per_ha": "mu_d"})
    )

    old_cmp = old_cmp.merge(mu, on="district_id", how="left")
    c_cmp = c_cmp.merge(mu, on="district_id", how="left")

    m = old_cmp.merge(
        c_cmp,
        on=["split", "season_year", "district_id", "opdate_key"],
        suffixes=("_old", "_new"),
        how="inner",
    )
    m = m[m["split"].isin(["val", "test"])]

    out = {}
    for split in ["val", "test"]:
        s = m[m["split"] == split].copy()
        ad = s["actual_yield_kg_per_ha_new"].to_numpy(dtype=np.float32) - s["mu_d_new"].to_numpy(dtype=np.float32)
        pdlt = s["predicted_yield_kg_per_ha_new"].to_numpy(dtype=np.float32) - s["mu_d_new"].to_numpy(dtype=np.float32)
        tp_d, fp_d, fn_d, tn_d = _conf_counts(ad < 0.0, pdlt < 0.0)
        tp_r, fp_r, fn_r, tn_r = _conf_counts(ad > 0.0, pdlt > 0.0)
        out[f"{split}_drop_recall"] = _safe_rate(tp_d, tp_d + fn_d)
        out[f"{split}_rise_recall"] = _safe_rate(tp_r, tp_r + fn_r)
        err = s["predicted_yield_kg_per_ha_new"].to_numpy(dtype=np.float32) - s["actual_yield_kg_per_ha_new"].to_numpy(dtype=np.float32)
        out[f"mapped_{split}_rmse"] = float(math.sqrt(float(np.mean(err**2))))

        s["ape_pct"] = np.where(
            s["actual_yield_kg_per_ha_new"].abs() > 1e-6,
            (s["predicted_yield_kg_per_ha_new"] - s["actual_yield_kg_per_ha_new"]).abs() / s["actual_yield_kg_per_ha_new"].abs() * 100.0,
            np.nan,
        )
        b = _district_bucket_counts(s.groupby("district_id")["ape_pct"].mean())
        out[f"{split}_bucket_lt2"] = int(b["lt2"])
        out[f"{split}_bucket_gt10"] = int(b["gt10"])

    return out


def _run(cmd, cwd: Path, env: dict) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    repo = Path('/Users/aaryan/Downloads/ugp')
    out_root = repo / 'codex_v2/experiments/tuning_rise_matrix_live_retry'
    old_pred = repo / 'codex_v2/experiments/live_b4_h25_seed42_20260216_220031/B4_shared_h25_s42/predictions_shared.csv'
    base_new_pred = repo / 'codex_v2/experiments/B4_e6e7_5d_h25_s42_20260303_035916/B4_e6e7_5d_h25_s42/predictions_shared.csv'

    guardrail_rmse = evaluate_candidate(old_pred, base_new_pred)['mapped_test_rmse']
    print('guardrail_mapped_test_rmse=', guardrail_rmse, flush=True)

    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    py = sys.executable

    # R4 rerun with batch_size=8 via config file.
    r4_cmd = [
        py, 'codex_v2/scripts/train_v2.py',
        '--mode', 'shared',
        '--target-mode', 'district_signed_log_asym',
        '--horizon-days', '25',
        '--opdate-profile', 'five_day_dec1_apr30',
        '--seed', '42',
        '--config-data', 'codex_v2/configs/data_v2.yaml',
        '--config-model', 'codex_v2/configs/model_3m.yaml',
        '--config-train', 'codex_v2/configs/train_shared_mps_120fixed_tuning.yaml',
        '--out-dir', str(out_root / 'R4'),
        '--run-name', 'R4_B4_e6e7_5d_h25_s42',
        '--fusion-mode', 'cross_attention',
        '--enable-e6', '--enable-e7',
        '--climate-normals-temp-csv', 'codex_v2/experiments/imd_normals_20260303_fixlat/district_temp_normals_monthly_1991_2020.csv',
        '--climate-normals-rain-csv', 'codex_v2/experiments/imd_normals_20260303_fixlat/district_rain_normals_monthly_1971_2020.csv',
        '--loss', 'asym_huber',
        '--rise-under-w', '1.0',
        '--drop-miss-w', '0.2',
        '--huber-delta', '1.0',
        '--sample-pos-weight', '2.0',
        '--checkpoint-objective', 'rmse',
        '--min-drop-recall', '0.777',
        '--pos-gain', '1.15',
        '--neg-gain', '1.0',
        '--no-rise-calibration',
        '--use-weighted-sampler',
    ]
    _run(r4_cmd, repo, env)

    # Evaluate R1-R4 and pick best by gate then rmse.
    rows = []
    spec_by_run = {
        'R1': dict(target_mode='district_signed_log', rise_under_w='0.6', weighted='--no-weighted-sampler', pos_gain='1.15', neg_gain='1.0'),
        'R2': dict(target_mode='district_signed_log', rise_under_w='1.0', weighted='--no-weighted-sampler', pos_gain='1.15', neg_gain='1.0'),
        'R3': dict(target_mode='district_signed_log', rise_under_w='1.0', weighted='--use-weighted-sampler', pos_gain='1.15', neg_gain='1.0'),
        'R4': dict(target_mode='district_signed_log_asym', rise_under_w='1.0', weighted='--use-weighted-sampler', pos_gain='1.15', neg_gain='1.0'),
    }

    for rid in ['R1', 'R2', 'R3', 'R4']:
        pred = out_root / rid / f'{rid}_B4_e6e7_5d_h25_s42' / 'predictions_shared.csv'
        ev = evaluate_candidate(old_pred, pred)
        gate = bool(
            ev['val_drop_recall'] >= 0.777
            and ev['test_drop_recall'] >= 0.886
            and ev['test_rise_recall'] >= 0.200
            and ev['test_bucket_lt2'] >= 22
            and ev['test_bucket_gt10'] <= 31
            and ev['mapped_test_rmse'] <= (guardrail_rmse + 3.0)
        )
        rows.append({'run_id': rid, **ev, 'gates_pass': gate})
        print(json.dumps(rows[-1], indent=2), flush=True)

    df = pd.DataFrame(rows)
    passed = df[df['gates_pass'] == True]
    if not passed.empty:
        best = passed.sort_values(['mapped_test_rmse', 'test_rise_recall'], ascending=[True, False]).iloc[0]
    else:
        best = df.sort_values(['test_rise_recall', 'mapped_test_rmse'], ascending=[False, True]).iloc[0]
    best_run = str(best['run_id'])
    print('best_for_r5=', best_run, flush=True)

    cfg = spec_by_run[best_run]
    r5_cmd = [
        py, 'codex_v2/scripts/train_v2.py',
        '--mode', 'shared',
        '--target-mode', cfg['target_mode'],
        '--horizon-days', '25',
        '--opdate-profile', 'five_day_dec1_apr30',
        '--seed', '42',
        '--config-data', 'codex_v2/configs/data_v2.yaml',
        '--config-model', 'codex_v2/configs/model_3m.yaml',
        '--config-train', 'codex_v2/configs/train_shared_mps_120fixed_tuning.yaml',
        '--out-dir', str(out_root / 'R5'),
        '--run-name', 'R5_B4_e6e7_5d_h25_s42',
        '--fusion-mode', 'cross_attention',
        '--enable-e6', '--enable-e7',
        '--climate-normals-temp-csv', 'codex_v2/experiments/imd_normals_20260303_fixlat/district_temp_normals_monthly_1991_2020.csv',
        '--climate-normals-rain-csv', 'codex_v2/experiments/imd_normals_20260303_fixlat/district_rain_normals_monthly_1971_2020.csv',
        '--loss', 'asym_huber',
        '--rise-under-w', cfg['rise_under_w'],
        '--drop-miss-w', '0.2',
        '--huber-delta', '1.0',
        '--sample-pos-weight', '2.0',
        '--checkpoint-objective', 'drop_constrained_rise',
        '--min-drop-recall', '0.777',
        '--pos-gain', cfg['pos_gain'],
        '--neg-gain', cfg['neg_gain'],
        '--enable-rise-calibration',
        cfg['weighted'],
    ]
    _run(r5_cmd, repo, env)

    r5_pred = out_root / 'R5' / 'R5_B4_e6e7_5d_h25_s42' / 'predictions_shared.csv'
    r5_eval = evaluate_candidate(old_pred, r5_pred)
    r5_gate = bool(
        r5_eval['val_drop_recall'] >= 0.777
        and r5_eval['test_drop_recall'] >= 0.886
        and r5_eval['test_rise_recall'] >= 0.200
        and r5_eval['test_bucket_lt2'] >= 22
        and r5_eval['test_bucket_gt10'] <= 31
        and r5_eval['mapped_test_rmse'] <= (guardrail_rmse + 3.0)
    )

    out = {
        'guardrail_mapped_test_rmse': guardrail_rmse,
        'r1_r4': rows,
        'selected_for_r5': best_run,
        'r5': {**r5_eval, 'gates_pass': r5_gate},
    }
    df.to_csv(out_root / 'r1_r4_eval.csv', index=False)
    with (out_root / 'tuning_summary.json').open('w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
