# Wheat Yield Prediction Snapshot, 2010-2022

This folder is a clean GitHub snapshot of the current wheat-yield forecasting work. It is organized around the current experimental protocol:

- Train years: 2010-2018
- Test years: 2019-2022
- Target: district-level wheat yield in kg/ha
- Direction/sign target: whether yield is above or below the trend baseline for that district-year
- Main operating setup: h25 S2S weather horizon, lag-1/trend baselines, agri/economic inputs, remote-sensing inputs, and post-hoc tabular/classification stacking

## Current Leaderboard

All numbers below are on the 2019-2022 test window, 476 district-year rows.

| Model / variant | RMSE | MAE | R2 | Sign accuracy | Drop recall | Rise recall | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Trend baseline | 396.95 | 299.40 | 0.5741 | NA | NA | NA | Deterministic trend; sign is not meaningful because prediction equals trend. |
| Lag-1 baseline | 362.77 | 267.33 | 0.6443 | 70.80% | 58.33% | 82.26% | Strong simple baseline. |
| Old SOTA sign-magnitude stack | 334.35 | 246.87 | 0.6979 | 71.43% | 61.84% | 80.24% | Best result before the latest meta-stack work. |
| Hybrid meta-stack with old-SOTA sign and weighted magnitude | 321.16 | 236.15 | 0.7212 | 71.43% | 61.84% | 80.24% | Best general RMSE model before classification refinement. |
| Logistic old-SOTA flip-router, h25 meta-only, max sign | 319.90 | 234.51 | 0.7234 | 73.53% | 60.09% | 85.89% | Best full-coverage sign model. Flips 18 test cases; 14 flips are correct. |
| Logistic old-SOTA flip-router, h25 meta-only, RMSE trade-off | 319.31 | 233.73 | 0.7244 | 73.32% | 60.09% | 85.48% | Slightly lower RMSE, slightly lower sign accuracy. |
| CatBoost lag-1 flip-router, h25 meta-only | 318.77 | 234.42 | 0.7254 | 72.06% | 55.26% | 87.50% | Best RMSE among classifier-sign variants that still beat baseline sign. |

## Main Interpretation

The best current direction model is not a direct rise/drop classifier. It is a conservative flip-router:

1. Start from the old SOTA sign prediction.
2. Estimate whether that sign is likely to be wrong.
3. Flip the sign only at high confidence.
4. Reuse the final hybrid model's magnitude around the trend baseline.

This works better because the old SOTA sign is already fairly strong. The classifier only has to identify a small number of high-value corrections instead of relearning the entire rise/drop process.

The direct classification models using weather/agri/economic variables did not beat the meta-only flip-router. The best direct classifiers were around 70-71% sign accuracy. The useful classification signal is currently concentrated in model disagreement, prediction margins, lag-1/trend distance, and ensemble vote structure.

## Why 90% Full-Coverage Sign Accuracy Is Hard

The current binary sign label is noisy near the trend baseline. When actual yield is close to the trend line, tiny target noise can change the sign label even if the numeric forecast is reasonable.

Observed test behavior:

- All test rows: best classifier sign accuracy is 73.53%.
- Predicted margin >= 100 kg/ha: sign accuracy rises to about 84.6% on 51.9% coverage.
- Predicted margin >= 200 kg/ha: sign accuracy rises to about 86.5% on 26.5% coverage.
- Predicted margin >= 400 kg/ha: sign accuracy rises to about 95.2% on 4.4% coverage.

So 90%+ is reachable for high-confidence subsets, but not yet for all district-years under the current sign definition.

## Data Included

The snapshot includes processed, district-level data needed for the 2010-2022 experiments:

| Folder | Files | Size | Contents |
|---|---:|---:|---|
| `data/weather_s2s_daily_2010_2022/` | 13 | 393.94 MiB | District-level daily S2S weather features for 2010-2022. Variables include `tmax`, `tmin`, precipitation, solar radiation, and wind. |
| `data/weather_s2s_temp6h_2010_2022/` | 13 | 526.64 MiB | District-level 6-hour max/min temperature features for 2010-2022. Useful for heat-stress and hot-night features. |
| `data/climatology/` | 6 | 12.66 MiB | Reforecast climatology files including h25, h46, and five-day variants. |
| `data/yields/` | 6 | 0.85 MiB | DES/APY wheat yield panel, model-ready district-year files, and audit metadata. |
| `data/agri_economics/` | 4 | 0.12 MiB | ICRISAT district-level land-use, fertilizer, irrigated-area, and soil variables. |
| `data/remote_sensing_landsat_compat/` | 119 | 8.12 MiB | District-level remote-sensing compatibility inputs. |
| `data/district_metadata/` | 2 | 0.03 MiB | District metadata and aggregation weights. |

Large raw GRIB files, `.pt` tensors, model checkpoints, and temporary diagnostics are intentionally not included.

## Experiment Artifacts Included

| Folder | Purpose |
|---|---|
| `experiments/hybrid_meta_stack_lgbm_catboost_2010_2018_test2019_2022/` | Latest RMSE-focused hybrid/meta-stack results. |
| `experiments/classification_refinement_h25_2010_2018_test2019_2022/` | Latest classification-focused sign-router sweep and best predictions. |
| `experiments/classification_focused_h25_2010_2018_test2019_2022/` | Earlier broad classification sweep. |
| `experiments/sign_breakthrough_2010_2018_test2019_2022/` | Sign-feature and router diagnostics for h25. |
| `experiments/sign_breakthrough_h46_2010_2018_test2019_2022/` | Sign-feature and router diagnostics for h46. |
| `experiments/observed_weather_proxy_sign_2010_2018_test2019_2022_retry/` | Observed/reanalysis-style weather proxy diagnostics. |
| `experiments/tabular_residual_h46_2010_2018_test2019_2022/` | h46 residual tabular search. |
| `experiments/sota_search_2010_2018_test2019_2022/` | Earlier SOTA comparison and final sign-magnitude ensemble artifacts. |

## Code Included

The `code/` folder contains the scripts/configs used for the latest searches:

- `run_hybrid_meta_stack_search.py`
- `run_classification_refinement_search.py`
- `run_classification_focused_search.py`
- `run_observed_weather_proxy_search.py`
- `run_sign_breakthrough_search.py`
- `run_tabular_residual_search.py`
- `build_reforecast_climatology.py`
- `configs/codex_data_v2_2010.yaml`
- `configs/data_config_2010.yaml`
- `configs/arch/`

## Reproducibility Notes

The files in this folder are a curated snapshot, not a raw-data dump. The purpose is to make the current progress reviewable on GitHub while keeping the folder focused on the 2010-2022 modeling path.

Important caveat: several model choices were selected while repeatedly inspecting the 2019-2022 holdout. For a professor update, these are valid as a current progress snapshot. For a final paper claim, the same modeling choices should be checked with rolling-year or leave-one-year-out validation.

## Manifest

`MANIFEST.csv` lists every file in this snapshot, except the manifest itself, with byte size and SHA-256 hash.
