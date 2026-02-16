# Model Training Guide (Dual-Channel Informer + GAT)

This document explains how to run the current yield model on another machine.

## 1. What This Model Is

Target:
- Predict district-level wheat yield (`yield_kg_per_ha`).

Current architecture:
- Dual temporal encoders (Informer-style), one per modality:
  - Weather channel input: `x_w in R^{B x N x H x 10}`
  - Satellite channel input: `x_s in R^{B x N x T_s x 4}`
- Temporal embeddings are fused and passed through graph attention over districts:
  - Node embedding per district
  - GAT message passing on district adjacency graph
  - MLP regression head -> one yield prediction per district

Notation:
- `B`: number of seasons in batch (in this pipeline, full seasons are stacked)
- `N`: number of districts (119 in current yield panel)
- `H`: forecast horizon days (for weather lead days, e.g. 25 or 46)
- `T_s`: satellite sequence length (43 five-day steps)

## 2. File Paths Used by Current Pipeline

Main training script:
- `scripts/train_informer_gat_operational.py`

Model definition:
- `src/models/informer_gat.py`

Default data inputs used by training script:
- District table: `data/processed/s2s_district/districts.parquet`
- Yield labels: `data/yields/apy_query_report_model_ready_119.csv`
- Weather tensors: `data/processed/s2s_district/s2s_district_daily_YYYY.parquet`
- Satellite merged files: `Remote sensing data/sentinel2_wheat_pipeline/output/merged/*.csv`
- Sangrur/Malerkotla merge weights:
  - `data/yields/apy_query_report_sangrur_malerkotla_audit.csv`

Baseline data committed in repo for reproducible training:
- `data/yields/**` (yield raw + model-ready outputs)
- `data/processed/s2s_district/districts.parquet`
- `data/processed/s2s_district/weights.parquet`
- `data/processed/s2s_district/s2s_district_daily_*.parquet`
- `data/processed/s2s_district/s2s_district_temp_6h_*.parquet`
- `data/boundaries/World Bank Official Boundaries - Admin 2.gpkg`

Not committed by design:
- Generated `.pt` files (coworker can generate locally)
- Raw S2S GRIB archive under `data/s2s/**`
- Clean duplicate weather outputs under `data/processed/s2s_district_clean/**`

## 3. Environment Setup

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional: generate local `.pt` files from baseline parquet data:

```bash
python scripts/export_s2s_parquet_to_pt.py \
  --input-dir data/processed/s2s_district \
  --output-dir data/processed/s2s_district \
  --start-year 2017 \
  --end-year 2023
```

## 4. Train Command (Current 10M Parameter Config)

Example: operational dates `02-15`, `02-25`, `03-05`, horizon `25d`, `120` epochs.

```bash
python scripts/train_informer_gat_operational.py \
  --operational-dates 02-15 02-25 03-05 \
  --forecast-horizon 25 \
  --epochs 120 \
  --patience 120 \
  --weather-d-model 256 \
  --sat-d-model 256 \
  --weather-heads 8 \
  --sat-heads 8 \
  --weather-layers 4 \
  --sat-layers 4 \
  --weather-d-ff 1536 \
  --sat-d-ff 1536 \
  --gat-hidden 256 \
  --gat-heads 4 \
  --gat-layers 2 \
  --no-distil \
  --seed 42 \
  --split-seed 42 \
  --device auto \
  --save-predictions \
  --out-dir experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep
```

What it prints live:
- Progress bar per epoch
- `train_mse`, `val_mse`
- End-of-date metrics (`RMSE`, `MAE`, `MAPE`, `R2`)

## 5. Data Split Logic

Default split mode is `fixed`:
- Train years: `2017, 2018, 2019, 2020`
- Validation year: `2021`
- Test year: `2022`

Alternative random split exists (`--split-mode random`) and is controlled by `--split-seed`.

## 6. Operational Date Logic

Operational labels are `MM-DD` (example `02-15`).

For season start year `Y`:
- If month >= 9, target date year is `Y`
- Else target date year is `Y + 1`

Examples:
- Season `2017`, op-date `12-15` -> target date `2017-12-15`
- Season `2017`, op-date `02-15` -> target date `2018-02-15`

## 7. Satellite + Sangrur/Malerkotla Handling

In `src/models`, the network always has a satellite branch.

In data assembly (`scripts/train_informer_gat_operational.py`):
- Punjab Sangrur is merged with Malerkotla satellite signal for consistency with 119-district yield labels.
- Merge uses area-weighted combination from:
  - `data/yields/apy_query_report_sangrur_malerkotla_audit.csv`
- Satellite rows are clipped to `end_date <= operational_date`.

## 8. Output Artifacts

Inside `--out-dir`:
- `operational_date_metrics.csv`
- `operational_date_metrics.json`
- `embeddings/informer_gat_embeddings_opdate_<MM-DD>.npz`
- `predictions/predictions_opdate_<MM-DD>.csv` (if `--save-predictions`)

## 9. Current Model Blocks (High-Level)

1. Weather encoder:
- Linear projection -> positional encoding -> Transformer encoder stack (Informer-style)
- Optional distillation between layers (disabled with `--no-distil`)
- Masked temporal mean pooling

2. Satellite encoder:
- Same pattern as weather branch, with independent parameters

3. Fusion:
- Concatenate weather + satellite embeddings
- Fusion MLP
- Learnable gate to blend weather-priority vs satellite-priority projections

4. Spatial graph learning:
- Multi-layer dense GAT on district adjacency matrix
- Nonlinearity between GAT blocks

5. Regression head:
- MLP -> scalar yield prediction per district

## 10. Device Notes

Use:
- `--device auto` (recommended)
- `--device cpu`, `--device cuda`, or `--device mps`

If `mps` is not supported on your macOS/PyTorch build, use `--device cpu` or `--device auto`.
