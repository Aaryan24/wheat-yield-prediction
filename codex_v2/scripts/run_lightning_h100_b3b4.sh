#!/usr/bin/env bash
set -euo pipefail

python codex_v2/scripts/lightning_h100_b3b4_launcher.py "$@"
