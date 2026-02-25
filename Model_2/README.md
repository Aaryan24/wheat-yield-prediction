# Model_2 — Dual-Channel Informer + GAT Wheat Yield Prediction

Self-contained pipeline implementing the Informer + Graph Attention Network
architecture for district-level wheat yield prediction.

## Quick Start

### Step 1: Prepare the dataset

Combines S2S weather forecasts, Sentinel-2 satellite data, and yield labels
into a single `dataset.npz`:

```bash
# From repo root
python Model_2/prepare_dataset.py
```

This reads data from:
- `data/processed/s2s_district/s2s_district_daily_YYYY.parquet` (weather)
- `Remote sensing data/sentinel2_wheat_pipeline/output/merged/*.csv` (satellite)
- `data/yields/apy_query_report_model_ready_119.csv` (yield labels)
- `data/boundaries/World Bank Official Boundaries - Admin 2.gpkg` (adjacency)

### Step 2: Train the model

```bash
python Model_2/train.py
```

Quick smoke test:
```bash
python Model_2/train.py --epochs 3 --patience 3
```

### Step 3: Check the results

All analysis artifacts are auto-generated in `Model_2/analysis/`:

| File | Description |
|------|-------------|
| `metrics.json` | RMSE, MAE, MAPE, R² for train/val/test |
| `predictions.csv` | Per-district actual vs predicted yield |
| `training_curves.png` | Train / val loss over epochs |
| `error_analysis.png` | Scatter plot + error histogram |
| `model_params.json` | Architecture summary & param counts |

## Architecture

```
Weather Seq ──→ [Informer Encoder] ──┐
                                      ├─→ [Gated Fusion] ──→ [GAT Layers] ──→ [MLP Head] ──→ Yield
Satellite Seq ─→ [Informer Encoder] ──┘         ↑
                                          District Adjacency
```

- **Informer Encoder**: Transformer with distilling convolutions for sequence compression
- **Gated Fusion**: Learnable gate blending weather vs satellite embeddings
- **GAT**: Multi-head graph attention over 119-district adjacency graph
- **Head**: MLP → scalar yield (kg/ha) per district

## Configuration

Edit `config.yaml` to change hyperparameters, data paths, or split settings.

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | All hyperparameters and paths |
| `prepare_dataset.py` | Combine raw data → `dataset.npz` |
| `informer_gat_model.py` | Model definition (self-contained) |
| `train.py` | Train + auto-generate analysis |
| `analysis/` | Auto-populated results folder |
