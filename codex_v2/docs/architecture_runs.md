# Architecture V1 Runs

Canonical files for the first shared architecture-upgrade runs:

- Model config: `codex_v2/configs/arch/model_3m_arch_v1.yaml`
- Train config: `codex_v2/configs/arch/train_shared_mps_arch.yaml`
- Launcher: `codex_v2/scripts/launchers/local_mps_b4_arch_h25_launcher.py`

Canonical run names:

- Shared all-state: `B4_arch_v1_e6e7_5d_h25_s42`
- Punjab+Haryana regional: `B4_arch_v1_e6e7_5d_h25_s42_PH`
- Uttar Pradesh regional: `B4_arch_v1_e6e7_5d_h25_s42_UP`

These runs keep the observed-normal anomaly features unchanged for now and add:

- sequence-level bidirectional cross-attention
- attention pooling on both branches
- token-level calendar / lag encodings
- sign + magnitude output head

Experiments are stored under `codex_v2/experiments/` and should not be committed by default.
