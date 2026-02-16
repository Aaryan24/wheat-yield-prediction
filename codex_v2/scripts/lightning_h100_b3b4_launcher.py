#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run B3/B4 ablations sequentially on a single GPU (H100 recommended) "
            "with live progress output."
        )
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--out-root", type=str, default="codex_v2/experiments/lightning_h100_b3b4")
    parser.add_argument("--config-data", type=str, default="codex_v2/configs/data_v2.yaml")
    parser.add_argument("--config-model", type=str, default="codex_v2/configs/model_3m.yaml")
    parser.add_argument("--config-train-shared", type=str, default="codex_v2/configs/train_shared.yaml")
    parser.add_argument("--config-train-per-date", type=str, default="codex_v2/configs/train_per_date.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99])
    parser.add_argument("--horizons", type=int, nargs="+", default=[25, 46])
    parser.add_argument("--opdates", nargs="+", default=DEFAULT_OPDATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "codex_v2/scripts/run_ablation_v2.py",
        "--ablation-set",
        "B3,B4",
        "--seeds",
        *[str(x) for x in args.seeds],
        "--horizons",
        *[str(x) for x in args.horizons],
        "--operational-dates",
        *[str(x) for x in args.opdates],
        "--out-root",
        str(out_root),
        "--config-data",
        str(args.config_data),
        "--config-model",
        str(args.config_model),
        "--config-train-shared",
        str(args.config_train_shared),
        "--config-train-per-date",
        str(args.config_train_per_date),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    print("Running on single GPU launcher:", flush=True)
    print(f"  GPU_ID={args.gpu_id}", flush=True)
    print(f"  out_root={out_root}", flush=True)
    print("  ablations=B3,B4", flush=True)
    print(f"  horizons={args.horizons}", flush=True)
    print(f"  seeds={args.seeds}", flush=True)
    print(f"  opdates={' '.join(args.opdates)}", flush=True)
    print("", flush=True)

    # Stream output directly so tqdm/progress bars are visible live.
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)

    print("", flush=True)
    print("Completed.", flush=True)
    print(f"Compiled CSV: {out_root / 'v2_all_runs_compiled.csv'}", flush=True)
    print(f"Compiled MD : {out_root / 'v2_all_runs_report.md'}", flush=True)


if __name__ == "__main__":
    main()
