# V13 runbook

Run from the repository root.

## Full research rerun

```bash
V13_EPOCHS=60 V13_SEEDS=42,73 V13_DEVICE=cpu \
python rapid_yield_forecast/v13_crop_response_final/scripts/run_v13_final.py
```

Expected work:

- strict weather pretraining at historical cutoffs;
- 18 two-seed/variant/cutoff crop-response fits;
- 936 downstream linear/logistic fits;
- point, direction, and trajectory promotion tests;
- through-2022 deployment refits.

## Compile intervals and final metrics

```bash
python rapid_yield_forecast/v13_crop_response_final/scripts/finalize_v13.py
```

## Validate

```bash
python rapid_yield_forecast/v13_crop_response_final/scripts/validate_v13.py
```

## Main artifacts

- `artifacts/final_predictions.parquet`: one row per district, test season, and
  clock, with point, ranges, trend probability, and severe probability.
- `artifacts/final_metrics.csv`: pooled strict metrics.
- `artifacts/state_year_final_metrics.csv`: local stability.
- `artifacts/trajectory_metrics.csv`: crop-response ablations.
- `artifacts/trajectory_uncertainty.csv`: paired grouped uncertainty.
- `artifacts/direction_selected_metrics.csv`: March promotion evidence.
- `artifacts/point_selected_metrics.csv`: rejected point candidates.
- `artifacts/final_policy.json`: final clock-specific policy.
- `artifacts/deployment_manifest.json`: live feature order and blend rule.
- `models/`: two-seed response weights and promoted March direction head.

