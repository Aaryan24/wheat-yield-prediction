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


def _mps_is_available() -> bool:
    try:
        import torch

        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Apple Silicon MPS launcher for Codex V2 (B4 only, horizon 25 only)."
    )
    parser.add_argument("--out-root", type=str, default="codex_v2/experiments/local_mps_b4_h25")
    parser.add_argument("--config-data", type=str, default="codex_v2/configs/data_v2.yaml")
    parser.add_argument("--config-model", type=str, default="codex_v2/configs/model_3m.yaml")
    parser.add_argument("--config-train-shared", type=str, default="codex_v2/configs/train_shared_mps.yaml")
    parser.add_argument("--config-train-per-date", type=str, default="codex_v2/configs/train_per_date.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99])
    parser.add_argument("--opdates", nargs="+", default=DEFAULT_OPDATES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not _mps_is_available():
        raise RuntimeError(
            "MPS is not available on this machine/runtime. "
            "Install Apple Silicon PyTorch and verify with: "
            "python -c \"import torch; print(torch.backends.mps.is_available())\""
        )

    cmd = [
        sys.executable,
        "codex_v2/scripts/run_ablation_v2.py",
        "--ablation-set",
        "B4",
        "--seeds",
        *[str(x) for x in args.seeds],
        "--horizons",
        "25",
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

    print("Plan: B4 only, horizon=25 only, seeds=", [int(x) for x in args.seeds], flush=True)
    print("Operational dates:", " ".join([str(x) for x in args.opdates]), flush=True)
    print("Out root:", out_root, flush=True)
    print("Runtime device: mps", flush=True)

    if args.dry_run:
        print("Dry run command:")
        print(" ".join(cmd))
        return

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


if __name__ == "__main__":
    main()
