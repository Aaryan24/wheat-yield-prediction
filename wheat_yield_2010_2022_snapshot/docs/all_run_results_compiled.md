# Consolidated Experiment Results

Generated on: 2026-02-16

## Scope
- Operational metrics files parsed: `22`
- Operational-day smoke metrics files parsed: `2`
- Total normalized metric rows: `50`
- Full normalized CSV: `experiments/all_runs_compiled_metrics.csv`

## Grouped Run Summary (By Parent Experiment)
| run_group | subruns | rows | model_total_params | horizons | seeds | best_label | best_test_rmse | best_test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments/informer_v1_local_progress | 1 | 7 |  | 46 |  | 12-25 | 392.515 | 0.564 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16 | 4 | 4 | 1612673.0 | 15,25,35,46 | 42 | 01-24 | 409.697 | 0.524 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16 | 4 | 4 | 1612673.0 | 15,25,35,46 | 42 | 01-24 | 409.697 | 0.524 |
| experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | 1 | 3 | 10100993.0 | 25 | 42 | 02-15 | 474.914 | 0.361 |
| experiments/informer_large_transformer_01-24 | 1 | 1 | 1612673.0 | 46 |  | 01-24 | 496.642 | 0.301 |
| experiments/seed_sensitivity_46d_large_transformer_2026-02-16 | 2 | 2 | 1612673.0 | 46 | 7,99 | 01-24 | 533.768 | 0.193 |
| experiments/informer_v1_local_pred_01-24 | 1 | 1 |  | 46 |  | 01-24 | 602.130 | -0.027 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16 | 4 | 4 | 10100993.0 | 15,25,35,46 | 42 | 01-24 | 604.762 | -0.036 |
| experiments/informer_base_transformer_01-24 | 1 | 1 | 265665.0 | 46 |  | 01-24 | 622.182 | -0.097 |
| experiments/informer_runtime_probe_cpu | 1 | 1 |  | 46 |  | 12-05 | 4089.006 | -46.370 |
| experiments/informer_gat_smoke | 1 | 1 |  | 25 |  | dayminus-40 | 4094.232 | -46.491 |
| experiments/informer_gat_smoke_all | 1 | 7 |  | 10,20,25 |  | dayminus-20 | 4094.999 | -46.509 |
| experiments/informer_gat_smoke_dates_v2 | 1 | 7 |  | 46 |  | 01-24 | 4095.005 | -46.509 |
| experiments/informer_gat_smoke_dates | 1 | 7 |  | 46 |  | 12-15 | 4095.136 | -46.512 |

## Run-Level Summary (By Metrics File Folder)
| run_id | rows | model_total_params | horizons | seeds | epochs_ran | best_label | best_test_rmse | best_test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments/informer_v1_local_progress | 7 |  | 46 |  | 88-120 | 12-25 | 392.515 | 0.564 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_15d | 1 | 1612673.0 | 15 | 42 | 119 | 01-24 | 409.697 | 0.524 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_15d | 1 | 1612673.0 | 15 | 42 | 120 | 01-24 | 409.697 | 0.524 |
| experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | 3 | 10100993.0 | 25 | 42 | 120 | 02-15 | 474.914 | 0.361 |
| experiments/informer_large_transformer_01-24 | 1 | 1612673.0 | 46 |  | 120 | 01-24 | 496.642 | 0.301 |
| experiments/seed_sensitivity_46d_large_transformer_2026-02-16/seed_99 | 1 | 1612673.0 | 46 | 99 | 120 | 01-24 | 533.768 | 0.193 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_25d | 1 | 1612673.0 | 25 | 42 | 114 | 01-24 | 543.991 | 0.162 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_25d | 1 | 1612673.0 | 25 | 42 | 120 | 01-24 | 543.991 | 0.162 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_46d | 1 | 1612673.0 | 46 | 42 | 65 | 01-24 | 599.442 | -0.018 |
| experiments/informer_v1_local_pred_01-24 | 1 |  | 46 |  | 96 | 01-24 | 602.130 | -0.027 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_25d | 1 | 10100993.0 | 25 | 42 | 120 | 01-24 | 604.762 | -0.036 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_46d | 1 | 10100993.0 | 46 | 42 | 120 | 01-24 | 604.881 | -0.037 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_15d | 1 | 10100993.0 | 15 | 42 | 120 | 01-24 | 605.166 | -0.038 |
| experiments/seed_sensitivity_46d_large_transformer_2026-02-16/seed_7 | 1 | 1612673.0 | 46 | 7 | 58 | 01-24 | 606.813 | -0.043 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_35d | 1 | 1612673.0 | 35 | 42 | 64 | 01-24 | 607.115 | -0.044 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_35d | 1 | 10100993.0 | 35 | 42 | 120 | 01-24 | 609.174 | -0.051 |
| experiments/informer_base_transformer_01-24 | 1 | 265665.0 | 46 |  | 88 | 01-24 | 622.182 | -0.097 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_35d | 1 | 1612673.0 | 35 | 42 | 120 | 01-24 | 748.473 | -0.587 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_46d | 1 | 1612673.0 | 46 | 42 | 120 | 01-24 | 1019.956 | -1.947 |
| experiments/informer_runtime_probe_cpu | 1 |  | 46 |  |  | 12-05 | 4089.006 | -46.370 |
| experiments/informer_gat_smoke | 1 |  | 25 |  |  | dayminus-40 | 4094.232 | -46.491 |
| experiments/informer_gat_smoke_all | 7 |  | 10,20,25 |  |  | dayminus-20 | 4094.999 | -46.509 |
| experiments/informer_gat_smoke_dates_v2 | 7 |  | 46 |  |  | 01-24 | 4095.005 | -46.509 |
| experiments/informer_gat_smoke_dates | 7 |  | 46 |  |  | 12-15 | 4095.136 | -46.512 |

## Parameter-Count Comparison
| model_total_params | rows | horizons | min_epochs | max_epochs | best_test_rmse | mean_test_rmse | best_test_r2 | mean_test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1612673.0 | 11 | 15,25,35,46 | 58.000 | 120.000 | 409.697 | 592.690 | 0.524 | -0.070 |
| 10100993.0 | 7 | 15,25,35,46 | 120.000 | 120.000 | 474.914 | 586.965 | 0.361 | 0.018 |
| 265665.0 | 1 | 46 | 88.000 | 88.000 | 622.182 | 622.182 | -0.097 | -0.097 |

## Top 20 Rows By Test RMSE
| run_group | run_id | operational_label | horizon_days | model_total_params | epochs_ran | seed | val_rmse | test_rmse | test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | 12-25 | 46 |  | 120.000 |  | 351.780 | 392.515 | 0.564 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_15d | 01-24 | 15 | 1612673.0 | 120.000 | 42.000 | 345.233 | 409.697 | 0.524 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_15d | 01-24 | 15 | 1612673.0 | 119.000 | 42.000 | 345.233 | 409.697 | 0.524 |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | 12-05 | 46 |  | 120.000 |  | 352.164 | 416.707 | 0.508 |
| experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | 02-15 | 25 | 10100993.0 | 120.000 | 42.000 | 372.786 | 474.914 | 0.361 |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | 01-24 | 46 |  | 120.000 |  | 342.439 | 494.710 | 0.307 |
| experiments/informer_large_transformer_01-24 | experiments/informer_large_transformer_01-24 | 01-24 | 46 | 1612673.0 | 120.000 |  | 408.105 | 496.642 | 0.301 |
| experiments/seed_sensitivity_46d_large_transformer_2026-02-16 | experiments/seed_sensitivity_46d_large_transformer_2026-02-16/seed_99 | 01-24 | 46 | 1612673.0 | 120.000 | 99.000 | 437.368 | 533.768 | 0.193 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_25d | 01-24 | 25 | 1612673.0 | 114.000 | 42.000 | 320.021 | 543.991 | 0.162 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_25d | 01-24 | 25 | 1612673.0 | 120.000 | 42.000 | 320.021 | 543.991 | 0.162 |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | 01-04 | 46 |  | 93.000 |  | 464.526 | 597.533 | -0.012 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_46d | 01-24 | 46 | 1612673.0 | 65.000 | 42.000 | 462.650 | 599.442 | -0.018 |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | 12-15 | 46 |  | 91.000 |  | 462.618 | 600.718 | -0.022 |
| experiments/informer_v1_local_pred_01-24 | experiments/informer_v1_local_pred_01-24 | 01-24 | 46 |  | 96.000 |  | 460.381 | 602.130 | -0.027 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16 | experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_25d | 01-24 | 25 | 10100993.0 | 120.000 | 42.000 | 461.308 | 604.762 | -0.036 |
| experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | 02-25 | 25 | 10100993.0 | 120.000 | 42.000 | 461.304 | 604.856 | -0.037 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16 | experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_46d | 01-24 | 46 | 10100993.0 | 120.000 | 42.000 | 461.286 | 604.881 | -0.037 |
| experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | 03-05 | 25 | 10100993.0 | 120.000 | 42.000 | 461.304 | 605.005 | -0.037 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16 | experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_15d | 01-24 | 15 | 10100993.0 | 120.000 | 42.000 | 461.297 | 605.166 | -0.038 |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | 02-05 | 46 |  | 88.000 |  | 461.368 | 605.254 | -0.038 |

## Full Row-Level Results (All Parsed Rows)
| run_group | run_id | source_type | operational_label | days_before_harvest | horizon_days | model_total_params | seed | epochs_ran | best_epoch | train_rmse | val_rmse | test_rmse | train_r2 | val_r2 | test_r2 | train_mape | val_mape | test_mape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16 | experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_15d | operational_date_metrics | 01-24 |  | 15 | 10100993.0 | 42.000 | 120.000 | 115.000 | 734.949 | 461.297 | 605.166 | -0.092 | 0.000 | -0.038 | 14.537 | 9.932 | 12.382 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16 | experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_25d | operational_date_metrics | 01-24 |  | 25 | 10100993.0 | 42.000 | 120.000 | 94.000 | 734.316 | 461.308 | 604.762 | -0.090 | -0.000 | -0.036 | 14.533 | 9.939 | 12.375 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16 | experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_35d | operational_date_metrics | 01-24 |  | 35 | 10100993.0 | 42.000 | 120.000 | 41.000 | 740.830 | 461.745 | 609.174 | -0.110 | -0.002 | -0.051 | 14.578 | 9.880 | 12.451 |
| experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16 | experiments/horizon_sweep_01-24_seed42_10m_full120_live_2026-02-16/horizon_46d | operational_date_metrics | 01-24 |  | 46 | 10100993.0 | 42.000 | 120.000 | 107.000 | 734.521 | 461.286 | 604.881 | -0.091 | 0.000 | -0.037 | 14.534 | 9.936 | 12.377 |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_15d | operational_date_metrics | 01-24 |  | 15 | 1612673.0 | 42.000 | 119.000 | 94.000 | 567.280 | 345.233 | 409.697 | 0.349 | 0.440 | 0.524 |  |  |  |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_25d | operational_date_metrics | 01-24 |  | 25 | 1612673.0 | 42.000 | 114.000 | 89.000 | 614.677 | 320.021 | 543.991 | 0.236 | 0.519 | 0.162 |  |  |  |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_35d | operational_date_metrics | 01-24 |  | 35 | 1612673.0 | 42.000 | 64.000 | 39.000 | 737.856 | 461.431 | 607.115 | -0.101 | -0.001 | -0.044 |  |  |  |
| experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_2026-02-16/horizon_46d | operational_date_metrics | 01-24 |  | 46 | 1612673.0 | 42.000 | 65.000 | 40.000 | 725.437 | 462.650 | 599.442 | -0.064 | -0.006 | -0.018 |  |  |  |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_15d | operational_date_metrics | 01-24 |  | 15 | 1612673.0 | 42.000 | 120.000 | 94.000 | 567.280 | 345.233 | 409.697 | 0.349 | 0.440 | 0.524 |  |  |  |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_25d | operational_date_metrics | 01-24 |  | 25 | 1612673.0 | 42.000 | 120.000 | 89.000 | 614.677 | 320.021 | 543.991 | 0.236 | 0.519 | 0.162 |  |  |  |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_35d | operational_date_metrics | 01-24 |  | 35 | 1612673.0 | 42.000 | 120.000 | 117.000 | 694.805 | 361.240 | 748.473 | 0.024 | 0.387 | -0.587 |  |  |  |
| experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16 | experiments/horizon_sweep_01-24_seed42_large_transformer_full120_2026-02-16/horizon_46d | operational_date_metrics | 01-24 |  | 46 | 1612673.0 | 42.000 | 120.000 | 88.000 | 808.211 | 387.798 | 1019.956 | -0.321 | 0.293 | -1.947 |  |  |  |
| experiments/informer_base_transformer_01-24 | experiments/informer_base_transformer_01-24 | operational_date_metrics | 01-24 |  | 46 | 265665.0 |  | 88.000 | 63.000 | 757.909 | 466.620 | 622.182 | -0.161 | -0.023 | -0.097 |  |  |  |
| experiments/informer_gat_smoke | experiments/informer_gat_smoke | operational_day_metrics | dayminus-40 | 40.000 | 25 |  |  |  |  | 4208.378 | 3963.307 | 4094.232 | -34.811 | -72.813 | -46.491 |  |  |  |
| experiments/informer_gat_smoke_all | experiments/informer_gat_smoke_all | operational_day_metrics | dayminus-10 | 10.000 | 10 |  |  |  |  | 4209.220 | 3964.137 | 4095.052 | -34.825 | -72.844 | -46.510 |  |  |  |
| experiments/informer_gat_smoke_all | experiments/informer_gat_smoke_all | operational_day_metrics | dayminus-20 | 20.000 | 20 |  |  |  |  | 4209.160 | 3964.080 | 4094.999 | -34.824 | -72.842 | -46.509 |  |  |  |
| experiments/informer_gat_smoke_all | experiments/informer_gat_smoke_all | operational_day_metrics | dayminus-30 | 30.000 | 25 |  |  |  |  | 4209.407 | 3964.329 | 4095.248 | -34.828 | -72.851 | -46.514 |  |  |  |
| experiments/informer_gat_smoke_all | experiments/informer_gat_smoke_all | operational_day_metrics | dayminus-40 | 40.000 | 25 |  |  |  |  | 4209.373 | 3964.288 | 4095.220 | -34.828 | -72.850 | -46.514 |  |  |  |
| experiments/informer_gat_smoke_all | experiments/informer_gat_smoke_all | operational_day_metrics | dayminus-50 | 50.000 | 25 |  |  |  |  | 4209.353 | 3964.274 | 4095.199 | -34.827 | -72.849 | -46.513 |  |  |  |
| experiments/informer_gat_smoke_all | experiments/informer_gat_smoke_all | operational_day_metrics | dayminus-60 | 60.000 | 25 |  |  |  |  | 4209.501 | 3964.430 | 4095.345 | -34.830 | -72.855 | -46.517 |  |  |  |
| experiments/informer_gat_smoke_all | experiments/informer_gat_smoke_all | operational_day_metrics | dayminus-70 | 70.000 | 25 |  |  |  |  | 4209.296 | 3964.224 | 4095.134 | -34.826 | -72.847 | -46.512 |  |  |  |
| experiments/informer_gat_smoke_dates | experiments/informer_gat_smoke_dates | operational_date_metrics | 01-04 |  | 46 |  |  |  |  | 4209.484 | 3964.407 | 4095.315 | -34.830 | -72.854 | -46.516 |  |  |  |
| experiments/informer_gat_smoke_dates | experiments/informer_gat_smoke_dates | operational_date_metrics | 01-14 |  | 46 |  |  |  |  | 4209.333 | 3964.258 | 4095.174 | -34.827 | -72.849 | -46.513 |  |  |  |
| experiments/informer_gat_smoke_dates | experiments/informer_gat_smoke_dates | operational_date_metrics | 01-24 |  | 46 |  |  |  |  | 4209.513 | 3964.440 | 4095.354 | -34.830 | -72.855 | -46.517 |  |  |  |
| experiments/informer_gat_smoke_dates | experiments/informer_gat_smoke_dates | operational_date_metrics | 02-03 |  | 46 |  |  |  |  | 4209.465 | 3964.388 | 4095.305 | -34.829 | -72.853 | -46.516 |  |  |  |
| experiments/informer_gat_smoke_dates | experiments/informer_gat_smoke_dates | operational_date_metrics | 12-05 |  | 46 |  |  |  |  | 4209.459 | 3964.384 | 4095.300 | -34.829 | -72.853 | -46.516 |  |  |  |
| experiments/informer_gat_smoke_dates | experiments/informer_gat_smoke_dates | operational_date_metrics | 12-15 |  | 46 |  |  |  |  | 4209.297 | 3964.223 | 4095.136 | -34.826 | -72.847 | -46.512 |  |  |  |
| experiments/informer_gat_smoke_dates | experiments/informer_gat_smoke_dates | operational_date_metrics | 12-25 |  | 46 |  |  |  |  | 4209.541 | 3964.469 | 4095.380 | -34.830 | -72.856 | -46.517 |  |  |  |
| experiments/informer_gat_smoke_dates_v2 | experiments/informer_gat_smoke_dates_v2 | operational_date_metrics | 01-04 |  | 46 |  |  |  |  | 4209.352 | 3964.268 | 4095.190 | -34.827 | -72.849 | -46.513 |  |  |  |
| experiments/informer_gat_smoke_dates_v2 | experiments/informer_gat_smoke_dates_v2 | operational_date_metrics | 01-14 |  | 46 |  |  |  |  | 4209.490 | 3964.412 | 4095.336 | -34.830 | -72.854 | -46.516 |  |  |  |
| experiments/informer_gat_smoke_dates_v2 | experiments/informer_gat_smoke_dates_v2 | operational_date_metrics | 01-24 |  | 46 |  |  |  |  | 4209.163 | 3964.089 | 4095.005 | -34.824 | -72.842 | -46.509 |  |  |  |
| experiments/informer_gat_smoke_dates_v2 | experiments/informer_gat_smoke_dates_v2 | operational_date_metrics | 02-05 |  | 46 |  |  |  |  | 4209.202 | 3964.127 | 4095.048 | -34.825 | -72.844 | -46.510 |  |  |  |
| experiments/informer_gat_smoke_dates_v2 | experiments/informer_gat_smoke_dates_v2 | operational_date_metrics | 12-05 |  | 46 |  |  |  |  | 4209.264 | 3964.186 | 4095.103 | -34.826 | -72.846 | -46.511 |  |  |  |
| experiments/informer_gat_smoke_dates_v2 | experiments/informer_gat_smoke_dates_v2 | operational_date_metrics | 12-15 |  | 46 |  |  |  |  | 4209.241 | 3964.162 | 4095.085 | -34.825 | -72.845 | -46.511 |  |  |  |
| experiments/informer_gat_smoke_dates_v2 | experiments/informer_gat_smoke_dates_v2 | operational_date_metrics | 12-25 |  | 46 |  |  |  |  | 4209.449 | 3964.381 | 4095.288 | -34.829 | -72.853 | -46.515 |  |  |  |
| experiments/informer_large_transformer_01-24 | experiments/informer_large_transformer_01-24 | operational_date_metrics | 01-24 |  | 46 | 1612673.0 |  | 120.000 | 102.000 | 748.094 | 408.105 | 496.642 | -0.132 | 0.217 | 0.301 |  |  |  |
| experiments/informer_runtime_probe_cpu | experiments/informer_runtime_probe_cpu | operational_date_metrics | 12-05 |  | 46 |  |  |  |  | 4203.186 | 3958.068 | 4089.006 | -34.722 | -72.618 | -46.370 |  |  |  |
| experiments/informer_v1_local_pred_01-24 | experiments/informer_v1_local_pred_01-24 | operational_date_metrics | 01-24 |  | 46 |  |  | 96.000 | 71.000 | 732.329 | 460.381 | 602.130 | -0.084 | 0.004 | -0.027 |  |  |  |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | operational_date_metrics | 01-04 |  | 46 |  |  | 93.000 | 68.000 | 721.553 | 464.526 | 597.533 | -0.053 | -0.014 | -0.012 |  |  |  |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | operational_date_metrics | 01-14 |  | 46 |  |  | 91.000 | 66.000 | 742.868 | 462.262 | 611.030 | -0.116 | -0.004 | -0.058 |  |  |  |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | operational_date_metrics | 01-24 |  | 46 |  |  | 120.000 | 111.000 | 656.026 | 342.439 | 494.710 | 0.130 | 0.449 | 0.307 |  |  |  |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | operational_date_metrics | 02-05 |  | 46 |  |  | 88.000 | 63.000 | 734.917 | 461.368 | 605.254 | -0.092 | -0.000 | -0.038 |  |  |  |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | operational_date_metrics | 12-05 |  | 46 |  |  | 120.000 | 108.000 | 553.974 | 352.164 | 416.707 | 0.379 | 0.417 | 0.508 |  |  |  |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | operational_date_metrics | 12-15 |  | 46 |  |  | 91.000 | 66.000 | 727.832 | 462.618 | 600.718 | -0.071 | -0.006 | -0.022 |  |  |  |
| experiments/informer_v1_local_progress | experiments/informer_v1_local_progress | operational_date_metrics | 12-25 |  | 46 |  |  | 120.000 | 104.000 | 563.714 | 351.780 | 392.515 | 0.357 | 0.418 | 0.564 |  |  |  |
| experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | operational_date_metrics | 02-15 |  | 25 | 10100993.0 | 42.000 | 120.000 | 79.000 | 598.497 | 372.786 | 474.914 | 0.276 | 0.347 | 0.361 | 11.848 | 7.807 | 9.924 |
| experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | operational_date_metrics | 02-25 |  | 25 | 10100993.0 | 42.000 | 120.000 | 116.000 | 734.462 | 461.304 | 604.856 | -0.091 | 0.000 | -0.037 | 14.534 | 9.937 | 12.377 |
| experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | experiments/opdate_sweep_0215_0225_0305_h25_10m_120ep_2026-02-16 | operational_date_metrics | 03-05 |  | 25 | 10100993.0 | 42.000 | 120.000 | 87.000 | 734.688 | 461.304 | 605.005 | -0.091 | 0.000 | -0.037 | 14.536 | 9.935 | 12.380 |
| experiments/seed_sensitivity_46d_large_transformer_2026-02-16 | experiments/seed_sensitivity_46d_large_transformer_2026-02-16/seed_7 | operational_date_metrics | 01-24 |  | 46 | 1612673.0 | 7.000 | 58.000 | 33.000 | 737.407 | 461.391 | 606.813 | -0.100 | -0.000 | -0.043 |  |  |  |
| experiments/seed_sensitivity_46d_large_transformer_2026-02-16 | experiments/seed_sensitivity_46d_large_transformer_2026-02-16/seed_99 | operational_date_metrics | 01-24 |  | 46 | 1612673.0 | 99.000 | 120.000 | 112.000 | 759.553 | 437.368 | 533.768 | -0.167 | 0.101 | 0.193 |  |  |  |

## Non-Metric/Utility Artifacts
- `experiments/informer_concat_smoke_2017_12-05/run_summary.json`: smoke forward-pass only (no train/val/test RMSE).
  - `operational_date_label=12-05`, `forecast_horizon_days=46`, `timing_total_sec=1.0338`

## Partial/Interrupted Runs
- `experiments/horizon_sweep_01-24_seed42_10m_full120_2026-02-16`: KeyboardInterrupt during epoch 30/120

## Experiment Dirs With No Parsed Metrics
- `experiments/horizon_sweep_01-24_seed42_10m_full120_2026-02-16`
