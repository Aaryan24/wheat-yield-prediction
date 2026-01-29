# Wheat Yield Prediction using Dual-Channel Informer and Graph Attention Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning framework for predicting district-level wheat yields in the Indo-Gangetic Plain (Punjab, Haryana, Uttar Pradesh) using multi-source data and spatiotemporal modeling.

## 🎯 Key Features

- **Dual-Channel Informer**: Separate encoders for weather (S2S) and satellite (Sentinel-2) data
- **Graph Attention Network**: Models spatial dependencies between districts
- **S2S Forecasts**: Up to 46-day weather forecasts at 36km resolution
- **Sentinel-2 Imagery**: 10m resolution NDVI/LAI time series
- **District-Level Predictions**: Covers 100+ districts across Punjab, Haryana, and UP

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │   S2S Weather       │      │   Sentinel-2        │          │
│  │   (36km, 46 days)   │      │   (10m, NDVI/LAI)   │          │
│  │   6 variables       │      │   Time series       │          │
│  └──────────┬──────────┘      └──────────┬──────────┘          │
│             │                            │                      │
│             ▼                            ▼                      │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │  Weather Informer   │      │ Satellite Informer  │          │
│  │  (Temporal Encoder) │      │ (Temporal Encoder)  │          │
│  └──────────┬──────────┘      └──────────┬──────────┘          │
│             │                            │                      │
│             └──────────┬─────────────────┘                      │
│                        ▼                                        │
│             ┌─────────────────────┐                             │
│             │   Feature Fusion    │                             │
│             └──────────┬──────────┘                             │
│                        ▼                                        │
│             ┌─────────────────────┐                             │
│             │  Graph Attention    │                             │
│             │  Network (GAT)      │                             │
│             │  District Graph     │                             │
│             └──────────┬──────────┘                             │
│                        ▼                                        │
│             ┌─────────────────────┐                             │
│             │   Yield Prediction  │                             │
│             │   (kg/ha per dist)  │                             │
│             └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
wheat-yield-prediction/
├── configs/                    # Configuration files
│   ├── data_config.yaml       # Data paths and parameters
│   ├── model_config.yaml      # Model hyperparameters
│   └── train_config.yaml      # Training settings
│
├── data/                       # Data directory (git-ignored)
│   ├── s2s/                   # S2S weather forecasts (.grib)
│   ├── sentinel2/             # Sentinel-2 imagery (.tif)
│   ├── yields/                # Yield statistics (.csv)
│   ├── boundaries/            # District shapefiles (.shp)
│   └── processed/             # Preprocessed tensors (.pt)
│
├── src/                        # Source code
│   ├── data/                  # Data handling
│   │   ├── download_s2s.py    # S2S ECMWF download
│   │   ├── download_sentinel.py
│   │   ├── preprocess_s2s.py
│   │   ├── preprocess_sentinel.py
│   │   └── dataset.py         # PyTorch Dataset classes
│   │
│   ├── models/                # Model implementations
│   │   ├── informer.py        # Informer encoder
│   │   ├── gat.py             # Graph Attention Network
│   │   ├── dual_channel.py    # Dual-channel architecture
│   │   └── yield_predictor.py # Full model
│   │
│   ├── training/              # Training utilities
│   │   ├── trainer.py         # Training loop
│   │   ├── losses.py          # Loss functions
│   │   └── metrics.py         # Evaluation metrics
│   │
│   └── utils/                 # Utilities
│       ├── config.py          # Config loading
│       ├── geo_utils.py       # Geospatial helpers
│       └── visualization.py   # Plotting functions
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_results_analysis.ipynb
│
├── experiments/                # Experiment outputs
│   ├── plots/                 # Generated figures
│   └── logs/                  # Training logs
│
├── scripts/                    # CLI scripts
│   ├── download_data.sh       # Download all data
│   ├── preprocess.sh          # Run preprocessing
│   └── train.sh               # Train model
│
├── tests/                      # Unit tests
│   └── test_models.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/wheat-yield-prediction.git
cd wheat-yield-prediction
```

### 2. Setup Environment
```bash
conda create -n wheat python=3.10 -y
conda activate wheat
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
# ECMWF API (for S2S data)
cat > ~/.ecmwfapirc << EOF
{
    "url": "https://api.ecmwf.int/v1",
    "key": "YOUR_KEY",
    "email": "your@email.com"
}
EOF

# Copernicus Data Space (for Sentinel-2)
# Register at: https://dataspace.copernicus.eu/
```

### 4. Download Data
```bash
# S2S weather forecasts (2017-2023)
python src/data/download_s2s.py

# Sentinel-2 imagery
python src/data/download_sentinel.py
```

### 5. Train Model
```bash
python -m src.training.trainer --config configs/train_config.yaml
```

## 📦 Data Sources

| Data | Source | Resolution | Period |
|------|--------|------------|--------|
| Weather Forecasts | S2S ECMWF | 36 km, 0-46 days | 2017-2023 |
| Satellite Imagery | Sentinel-2 | 10 m | 2017-2023 |
| Yield Statistics | Govt. of India | District-level | 2017-2023 |
| Boundaries | GADM | Admin Level 2 | - |

### Estimated Data Sizes
| Dataset | Size |
|---------|------|
| S2S (7 years, 6 vars) | ~3-5 GB |
| Sentinel-2 (NDVI only) | ~10-20 GB |
| Processed Tensors | ~2-3 GB |

## 🔬 Model Details

### Dual-Channel Informer
- **Weather Channel**: Processes 6 S2S variables (Tmax, Tmin, Precip, Solar, Wind U/V)
- **Satellite Channel**: Processes NDVI/LAI time series
- **Architecture**: Informer with ProbSparse self-attention

### Graph Attention Network
- **Nodes**: District centroids
- **Edges**: Spatial adjacency (shared boundaries)
- **Attention**: Multi-head attention for neighbor aggregation

## 📈 Expected Results

Based on similar literature:
- **RMSE**: ~0.5 t/ha
- **R²**: ~0.75-0.85
- **Lead Time**: 25-40 days before harvest

## 📚 References

1. Peng et al. (2024). "A Deep-Learning Network for Wheat Yield Prediction Combining Weather Forecast and Remote Sensing Data"
2. Zhou et al. (2021). "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
3. Veličković et al. (2018). "Graph Attention Networks"

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
