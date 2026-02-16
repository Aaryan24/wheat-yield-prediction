#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


DEFAULT_OPDATES = [
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


@dataclass
class Job:
    ablation: str
    horizon: int
    seed: int
    gpu: int
    run_name: str


@dataclass
class ActiveJob:
    job: Job
    proc: subprocess.Popen
    log_path: Path
    run_dir: Path


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _read_last_jsonl(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        text = path.read_text().strip()
        if not text:
            return None
        line = text.splitlines()[-1]
        return json.loads(line)
    except Exception:
        return None


def _read_summary_metrics(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path)
        if df.empty:
            return None
        # Prefer mean_over_opdates rows.
        val = df[(df["split"] == "val") & (df["summary_level"] == "mean_over_opdates")]
        test = df[(df["split"] == "test") & (df["summary_level"] == "mean_over_opdates")]
        if val.empty:
            val = df[df["split"] == "val"]
        if test.empty:
            test = df[df["split"] == "test"]
        out: Dict[str, float] = {}
        if not val.empty:
            out["val_rmse"] = float(val.iloc[0]["rmse"])
            out["val_r2"] = float(val.iloc[0]["r2"])
        if not test.empty:
            out["test_rmse"] = float(test.iloc[0]["rmse"])
            out["test_r2"] = float(test.iloc[0]["r2"])
        return out if out else None
    except Exception:
        return None


def _bar(done: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    n = int((done / total) * width)
    return "[" + ("#" * n) + ("-" * (width - n)) + "]"


def _print_status(active: List[ActiveJob], total_epochs: int, wave_seed: int, start_ts: float) -> None:
    elapsed = int(time.time() - start_ts)
    h = elapsed // 3600
    m = (elapsed % 3600) // 60
    s = elapsed % 60
    print(f"\nSeed {wave_seed} | elapsed {h:02d}:{m:02d}:{s:02d}", flush=True)

    for aj in active:
        tlog = aj.run_dir / "training_log.jsonl"
        last = _read_last_jsonl(tlog)
        if last is None:
            line = (
                f"GPU{aj.job.gpu} {aj.job.run_name:<20} "
                f"{_bar(0, total_epochs)} 000/{total_epochs} waiting"
            )
        else:
            ep = int(last.get("epoch", 0))
            tr = float(last.get("train_loss", float("nan")))
            vr = float(last.get("val_rmse", float("nan")))
            r2 = float(last.get("val_r2", float("nan")))
            line = (
                f"GPU{aj.job.gpu} {aj.job.run_name:<20} "
                f"{_bar(ep, total_epochs)} {ep:03d}/{total_epochs} "
                f"train={tr:.3f} val_rmse={vr:.3f} val_r2={r2:.3f}"
            )
        print(line, flush=True)


def _launch_job(
    repo_root: Path,
    out_root: Path,
    config_data: Path,
    config_model: Path,
    config_train_shared: Path,
    config_train_per_date: Path,
    opdates: List[str],
    job: Job,
) -> ActiveJob:
    run_dir = out_root / job.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "launcher_stdout.log"
    lf = log_path.open("w")

    cmd = [
        sys.executable,
        "codex_v2/scripts/run_ablation_v2.py",
        "--ablation-set",
        job.ablation,
        "--seeds",
        str(job.seed),
        "--horizons",
        str(job.horizon),
        "--operational-dates",
        *opdates,
        "--out-root",
        str(out_root),
        "--config-data",
        str(config_data),
        "--config-model",
        str(config_model),
        "--config-train-shared",
        str(config_train_shared),
        "--config-train-per-date",
        str(config_train_per_date),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=lf,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return ActiveJob(job=job, proc=proc, log_path=log_path, run_dir=run_dir)


def _run_export(repo_root: Path, out_root: Path) -> None:
    cmd = [
        sys.executable,
        "codex_v2/scripts/export_reports_v2.py",
        "--out-root",
        str(out_root),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch B3/B4 matrix on 4 GPUs (T4 recommended) with live epoch progress summaries."
        )
    )
    parser.add_argument("--out-root", type=str, default="codex_v2/experiments/lightning_t4_b3b4")
    parser.add_argument("--config-data", type=str, default="codex_v2/configs/data_v2.yaml")
    parser.add_argument("--config-model", type=str, default="codex_v2/configs/model_3m.yaml")
    parser.add_argument("--config-train-shared", type=str, default="codex_v2/configs/train_shared.yaml")
    parser.add_argument("--config-train-per-date", type=str, default="codex_v2/configs/train_per_date.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99])
    parser.add_argument("--opdates", nargs="+", default=DEFAULT_OPDATES)
    parser.add_argument("--refresh-seconds", type=int, default=20)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    config_data = Path(args.config_data)
    config_model = Path(args.config_model)
    config_train_shared = Path(args.config_train_shared)
    config_train_per_date = Path(args.config_train_per_date)

    shared_cfg = _load_yaml(config_train_shared)
    total_epochs = int(shared_cfg.get("training", {}).get("epochs", 120))

    if len(args.gpus) < 4:
        raise RuntimeError("Need 4 GPU ids for this launcher (e.g., --gpus 0 1 2 3).")

    print("Plan: B3/B4 x horizons(25,46) x seeds", args.seeds, flush=True)
    print("Operational dates:", " ".join(args.opdates), flush=True)
    print("Out root:", out_root, flush=True)

    wave_jobs_template = [("B3", 25), ("B3", 46), ("B4", 25), ("B4", 46)]

    if args.dry_run:
        for seed in args.seeds:
            for gpu, (ab, hz) in zip(args.gpus[:4], wave_jobs_template):
                run_name = f"{ab}_shared_h{hz}_s{seed}"
                print(f"GPU{gpu}: {run_name}")
        return

    for seed in args.seeds:
        print(f"\n=== Seed wave {seed} ===", flush=True)
        active: List[ActiveJob] = []

        for gpu, (ab, hz) in zip(args.gpus[:4], wave_jobs_template):
            run_name = f"{ab}_shared_h{hz}_s{seed}"
            run_dir = out_root / run_name
            done_path = run_dir / "metrics_summary_seeded.csv"
            if done_path.exists():
                print(f"GPU{gpu}: skip {run_name} (already complete)", flush=True)
                continue

            job = Job(ablation=ab, horizon=hz, seed=seed, gpu=gpu, run_name=run_name)
            aj = _launch_job(
                repo_root=repo_root,
                out_root=out_root,
                config_data=config_data,
                config_model=config_model,
                config_train_shared=config_train_shared,
                config_train_per_date=config_train_per_date,
                opdates=[str(x) for x in args.opdates],
                job=job,
            )
            active.append(aj)
            print(f"GPU{gpu}: started {run_name}", flush=True)

        if not active:
            print(f"Seed {seed}: nothing to run", flush=True)
            _run_export(repo_root, out_root)
            continue

        t0 = time.time()
        while True:
            alive = [aj for aj in active if aj.proc.poll() is None]
            _print_status(active=active, total_epochs=total_epochs, wave_seed=seed, start_ts=t0)
            if not alive:
                break
            time.sleep(max(5, int(args.refresh_seconds)))

        # Final status for the wave.
        print(f"\nSeed {seed} complete. Final metrics:", flush=True)
        for aj in active:
            code = aj.proc.returncode
            if code != 0:
                print(
                    f"- {aj.job.run_name}: FAILED (exit={code}), log={aj.log_path}",
                    flush=True,
                )
                continue

            summary = _read_summary_metrics(aj.run_dir / "metrics_summary_seeded.csv")
            if summary is None:
                print(f"- {aj.job.run_name}: completed, summary missing. log={aj.log_path}", flush=True)
            else:
                print(
                    (
                        f"- {aj.job.run_name}: "
                        f"val_rmse={summary.get('val_rmse', float('nan')):.3f}, "
                        f"val_r2={summary.get('val_r2', float('nan')):.3f}, "
                        f"test_rmse={summary.get('test_rmse', float('nan')):.3f}, "
                        f"test_r2={summary.get('test_r2', float('nan')):.3f}"
                    ),
                    flush=True,
                )

        _run_export(repo_root, out_root)

    print("\nAll seed waves completed.", flush=True)
    print(f"Compiled CSV: {out_root / 'v2_all_runs_compiled.csv'}", flush=True)
    print(f"Compiled MD : {out_root / 'v2_all_runs_report.md'}", flush=True)


if __name__ == "__main__":
    main()
