# Codex V2: Transformer + GAT Yield Modeling

This folder is an isolated V2 training stack for district-level wheat yield prediction.

- Temporal encoder: **standard Transformer encoder** (no Informer distillation)
- Spatial encoder: **dense GAT** over district adjacency
- Supports both training styles:
  - `shared`: one model across multiple operational dates (with op-date embedding)
  - `per_date`: one model per operational date (ablation baseline)
- Default model size target: **~3.0M trainable parameters**

## Folder Layout

- `configs/`
  - `data_v2.yaml`: baseline data paths and split defaults
  - `model_3m.yaml`: 3M parameter model config
  - `train_shared.yaml`, `train_per_date.yaml`: optimization/training defaults
- `src/data/`
  - `build_dataset_v2.py`: dataset assembly, scaling, target transforms, split-safe stats
  - `feature_engineering_v2.py`: weather/satellite engineered features + missingness indicators
  - `satellite_alignment_v2.py`: RS alignment, Sangrur/Malerkotla merge, mask-fix imputation
- `src/models/`
  - `temporal_transformer.py`: branch Transformer encoder
  - `dual_channel_transformer_gat_v2.py`: dual-branch fusion + GAT + regression head
- `src/training/`
  - `losses_v2.py`, `train_loop_v2.py`
- `src/eval/`
  - `metrics_v2.py`, `reporting_v2.py`
- `scripts/`
  - `train_v2.py`: single training job interface
  - `run_ablation_v2.py`: full A0..B4 matrix runner
  - `export_reports_v2.py`: compile all runs into global CSV/MD reports
- `tests/`
  - mask correctness, split leakage, parameter-budget tests
- `experiments/`
  - output root for runs and compiled reports

## Required Baseline Data (already expected in repo)

Configured in `codex_v2/configs/data_v2.yaml`:

- `data/processed/s2s_district/districts.parquet`
- `data/processed/s2s_district/s2s_district_daily_*.parquet`
- `data/yields/apy_query_report_model_ready_119.csv`
- `data/yields/apy_query_report_sangrur_malerkotla_audit.csv`
- `Remote sensing data/sentinel2_wheat_pipeline/output/merged/*.csv`
- `configs/data_config.yaml` (boundary metadata for adjacency)

## Architecture (Current Default)

From `configs/model_3m.yaml`:

- Weather branch Transformer: `d_model=160`, `layers=3`, `heads=8`, `d_ff=640`
- Satellite branch Transformer: `d_model=160`, `layers=3`, `heads=8`, `d_ff=640`
- Fusion block: `concat_gate` (or `cross_attention` for B4)
- GAT stack: hidden `192`, heads `4`, layers `2`
- Op-date embedding: `16` dims (shared mode)
- Expected trainable params: `2.8M - 3.2M` (test-enforced)

## V2 Data/Training Behavior

- Satellite fix: state-mean imputed active RS steps are marked valid in mask.
- Sangrur/Malerkotla: Malerkotla contribution is merged into Sangrur in RS branch using audit weights.
- Target modes:
  - `raw`
  - `district_demeaned`
  - `district_zscore`
- Train-only stats:
  - feature scalers fit on train years only
  - district mean/std for anomaly targets fit on train years only
- Default split:
  - train: `2017-2020`
  - val: `2021`
  - test: `2022`

## Quick Start

Run one shared model (single seed, multiple op-dates):

```bash
python codex_v2/scripts/train_v2.py \
  --mode shared \
  --target-mode district_demeaned \
  --horizon-days 25 \
  --operational-dates 12-05 12-15 12-25 01-04 01-14 01-24 02-05 \
  --seed 42 \
  --config-model codex_v2/configs/model_3m.yaml \
  --config-train codex_v2/configs/train_shared.yaml \
  --out-dir codex_v2/experiments/shared_example
```

Run per-date baseline (same CLI; script trains one model per op-date):

```bash
python codex_v2/scripts/train_v2.py \
  --mode per_date \
  --target-mode raw \
  --horizon-days 25 \
  --operational-dates 12-05 12-15 12-25 01-04 01-14 01-24 02-05 \
  --seed 42 \
  --config-model codex_v2/configs/model_3m.yaml \
  --config-train codex_v2/configs/train_per_date.yaml \
  --out-dir codex_v2/experiments/per_date_example
```

Run full ablation matrix (A0..B4, multi-seed, horizons):

```bash
python codex_v2/scripts/run_ablation_v2.py \
  --ablation-set full \
  --seeds 7 42 99 \
  --horizons 25 46 \
  --operational-dates 12-05 12-15 12-25 01-04 01-14 01-24 02-05 02-15 02-25 03-05 \
  --out-root codex_v2/experiments
```

Compile reports from finished runs:

```bash
python codex_v2/scripts/export_reports_v2.py --out-root codex_v2/experiments
```

## Lightning (4x T4) Launcher

For your narrowed run scope (`B3`, `B4`; horizons `25`, `46`; seeds `7,42,99`; op-dates excluding `12-05`), use:

```bash
bash codex_v2/scripts/run_lightning_t4_b3b4.sh \
  --out-root codex_v2/experiments/lightning_t4_b3b4 \
  --gpus 0 1 2 3 \
  --refresh-seconds 20
```

What it does:

- Runs 3 seed waves (`7`, `42`, `99`).
- In each wave, launches 4 parallel jobs:
  - `B3_h25`, `B3_h46`, `B4_h25`, `B4_h46`
- Assigns one job per GPU via `CUDA_VISIBLE_DEVICES`.
- Prints live epoch progress snapshots in terminal (every `--refresh-seconds`).
- Prints concise final metrics per run:
  - `val_rmse`, `val_r2`, `test_rmse`, `test_r2`
- Writes logs to each run folder as `launcher_stdout.log`.
- Recompiles global reports after each wave.

Dry-run plan preview:

```bash
python codex_v2/scripts/lightning_t4_b3b4_launcher.py --dry-run
```

## Local Mac M-Series (MPS) Launcher

For local Apple Silicon runs (for example MacBook Air M4) with your strict scope (`B4` only, `25d` only), use:

```bash
bash codex_v2/scripts/run_local_mps_b4_h25.sh \
  --out-root codex_v2/experiments/local_mps_b4_h25
```

What it does:

- Forces `runtime.device: mps` via `codex_v2/configs/train_shared_mps.yaml`.
- Runs ablation set `B4` only.
- Runs horizon `25` only.
- Uses default seed waves `7`, `42`, `99`.
- Uses op-dates excluding `12-05` (same narrowed set as Lightning launcher).
- Writes run artifacts and compiled reports under the provided `--out-root`.

Dry-run plan preview:

```bash
python codex_v2/scripts/local_mps_b4_h25_launcher.py --dry-run
```

## Output Artifacts

Each run directory writes:

- `metrics_per_opdate.csv`
- `metrics_summary_seeded.csv`
- `predictions_<mode>.csv`
- `feature_coverage_report.csv`
- `model_config_resolved.json`
- `training_log.jsonl`
- `sangrur_merge_weights.csv` (extra traceability artifact)

Global compiled outputs:

- `codex_v2/experiments/v2_all_runs_compiled.csv`
- `codex_v2/experiments/v2_all_runs_report.md`

## Tests

Run:

```bash
pytest -q codex_v2/tests
```

Covers:

- satellite mask validity after imputation
- train-only target/scaler leakage checks
- 3M parameter budget enforcement
