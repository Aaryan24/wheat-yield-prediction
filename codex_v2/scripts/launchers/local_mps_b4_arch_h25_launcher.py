#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_TEMP_NORMALS = (
    "codex_v2/experiments/imd_normals_20260303_fixlat/"
    "district_temp_normals_monthly_1991_2020.csv"
)
DEFAULT_RAIN_NORMALS = (
    "codex_v2/experiments/imd_normals_20260303_fixlat/"
    "district_rain_normals_monthly_1971_2020.csv"
)


def _mps_is_available() -> bool:
    try:
        import torch

        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first architecture-upgrade B4 shared h25 job on Apple MPS."
    )
    parser.add_argument("--out-dir", type=str, default="codex_v2/experiments/B4_arch_v1_e6e7_5d_h25_s42")
    parser.add_argument("--config-data", type=str, default="codex_v2/configs/data_v2.yaml")
    parser.add_argument("--config-model", type=str, default="codex_v2/configs/arch/model_3m_arch_v1.yaml")
    parser.add_argument("--config-train", type=str, default="codex_v2/configs/arch/train_shared_mps_arch.yaml")
    parser.add_argument("--temp-normals-csv", type=str, default=DEFAULT_TEMP_NORMALS)
    parser.add_argument("--rain-normals-csv", type=str, default=DEFAULT_RAIN_NORMALS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not _mps_is_available():
        raise RuntimeError("MPS is not available in this runtime.")

    cmd = [
        sys.executable,
        "codex_v2/scripts/train_v2.py",
        "--mode",
        "shared",
        "--target-mode",
        "district_signed_log",
        "--horizon-days",
        "25",
        "--opdate-profile",
        "five_day_dec1_apr30",
        "--seed",
        "42",
        "--config-data",
        str(args.config_data),
        "--config-model",
        str(args.config_model),
        "--config-train",
        str(args.config_train),
        "--out-dir",
        str(out_dir),
        "--enable-e6",
        "--enable-e7",
        "--enable-token-time-features",
        "--climate-normals-temp-csv",
        str(args.temp_normals_csv),
        "--climate-normals-rain-csv",
        str(args.rain_normals_csv),
        "--run-name",
        "B4_arch_v1_e6e7_5d_h25_s42",
    ]

    print("Run:", "B4_arch_v1_e6e7_5d_h25_s42", flush=True)
    print("Device:", "mps", flush=True)
    print("Out dir:", out_dir, flush=True)

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
