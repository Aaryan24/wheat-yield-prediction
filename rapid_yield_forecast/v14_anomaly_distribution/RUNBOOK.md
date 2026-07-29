# V14 runbook

Run commands from:

```text
/Users/aaryan/Downloads/ugp
```

## Full reproduction

Build strict cross-fitted future-crop features:

```bash
python rapid_yield_forecast/v14_anomaly_distribution/scripts/build_v14_outlooks.py
```

Run the base anomaly, matched-XGBoost, and distribution laboratory:

```bash
python rapid_yield_forecast/v14_anomaly_distribution/scripts/run_v14_lab.py
```

Run multi-year shrinkage, isolated-outlook, and probability extensions:

```bash
python rapid_yield_forecast/v14_anomaly_distribution/scripts/run_v14_extensions.py
```

Create deployment bundles, final tables, comparisons, hashes, and the figure:

```bash
python rapid_yield_forecast/v14_anomaly_distribution/scripts/finalize_v14.py
```

Validate:

```bash
python rapid_yield_forecast/v14_anomaly_distribution/scripts/validate_v14.py
```

Expected final message:

```text
"status": "pass"
```

## Main output files

| File | Purpose |
|---|---|
| `artifacts/final_predictions.parquet` | All district points, quantiles, and risks |
| `artifacts/final_metrics.csv` | Point and distribution headline metrics |
| `artifacts/final_state_year_metrics.csv` | State-year stability |
| `artifacts/outlook_year_state_audit.csv` | Every future-crop gain and loss |
| `artifacts/anomaly_protocol_metrics.csv` | All historical-selection protocols |
| `artifacts/probability_model_comparison.csv` | V14 vs V5 vs V13 probabilities |
| `artifacts/release_manifest.json` | Locked policy and hashes |
| `artifacts/validation.json` | Machine-readable checks |

## Deployment objects

`models/outlook_shadow_xgb_bundle.joblib` stores:

- no-future XGBoost;
- compact full-future XGBoost;
- isolated-effect XGBoost;
- broad full-future XGBoost;
- both seeds and exact feature columns; and
- the locked 1.75 correction formula.

`models/distribution_calibration_pool.parquet` stores the strictly historical,
normalised error pool needed to create new district percentiles.

`models/v5_distribution_recipe.json` stores distribution choices and
quantiles.

The four refitted XGBoost components are trained through 2022 for operational
use. Their historical score is not claimed; reported scores always come from
the strict rolling predictions.

## Inputs required for a live March 5 prediction

For each district:

1. district ID and state;
2. at least three earlier yields, ideally five to ten;
3. frozen V5 inputs and component forecasts;
4. experienced crop-stage weather available before March 5;
5. strictly lagged economic fields;
6. static soil, drainage, crop area, and phenology;
7. current wheat-only satellite state;
8. ten dated future-weather summaries from a valid earlier issue; and
9. the local recent-yield variability used to scale ranges.

If the future-crop feature is missing, return the V5 point and V14 ranges.
Never invent a future-crop correction.

## Live output contract

Return:

- safe V5 yield in kg/ha;
- V14 shadow yield in kg/ha;
- 5th through 95th percentiles;
- 50%, 80%, and 90% ranges;
- probability of an increase;
- probability of a decline of at least 10%;
- conservative 80% range; and
- data-availability and model-version flags.

