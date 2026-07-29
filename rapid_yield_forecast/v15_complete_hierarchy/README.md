# V15 complete hierarchy

V15 implements all seven parts of the proposed design and keeps only the parts
that survived strict testing.

The best research point forecast is:

```text
V15 = V14 future-weather shadow
      + 1.25 × (MODIS-pretrained current-crop XGBoost
                − matching XGBoost without that crop representation)
```

In plain language: start with V5, add the small future-weather signal found in
V14, then add a second small correction based on what Sentinel says about the
crop's current condition and expected near-term trajectory.

## Best honest result

| Test period | V5 RMSE | V14 RMSE | V15 RMSE | V15 gain vs V5 |
|---|---:|---:|---:|---:|
| 2019–2020, used for selection | 257.00 | 254.11 | **250.60** | **6.39 kg/ha** |
| 2021–2022, untouched confirmation | 288.61 | 288.13 | **287.20** | **1.41 kg/ha** |
| 2019–2022, all four years | 273.26 | 271.65 | **269.52** | **3.74 kg/ha** |

V15 improves three of the four individual years and eight of twelve
state-years. It is a genuine frontier challenger, but four independent years
are still too few to call the gain final. V5 remains the conservative
production anchor until more unseen seasons are available.

![V15 result summary](/Users/aaryan/Downloads/ugp/rapid_yield_forecast/v15_complete_hierarchy/artifacts/v15_result_summary.png)

## The seven requested stages

| Stage | Built? | Result |
|---|---|---|
| 1. Learn district normal from 10–20 years | Yes | Useful control; unstable in 2021–22, not promoted |
| 2. Predict percentage/log yield anomaly | Yes | Useful direction signal; independent yield level not promoted |
| 3. Extend satellite learning to 2000 | Yes | MODIS 2000–2022 pretraining; promoted |
| 4. Predict common state shock first | Yes | Explicit state model; unstable late, kept as shadow |
| 5. Predict district exposure | Yes | Shrunk district response coefficient; kept as shadow |
| 6. Fine-tune on Sentinel | Yes | Promoted crop-condition encoder |
| 7. Produce district probability distribution | Yes | q05–q95, rise probability, severe-drop probability |

The machine-readable evidence is in
[`seven_stage_evidence.csv`](artifacts/seven_stage_evidence.csv).

## Main outputs

- `artifacts/final_predictions.parquet`: final point, q05–q95, and event probabilities.
- `artifacts/point_model_ablation_metrics.csv`: V5, V14 and V15 comparison.
- `artifacts/point_year_state_audit.csv`: every year and state-year.
- `artifacts/point_grouped_bootstrap.csv`: uncertainty in the measured gain.
- `artifacts/v15_distribution_metrics.csv`: probability-range calibration and event AUC.
- `artifacts/validation.json`: fail-closed validation result.
- `artifacts/release_manifest.json`: file hashes and release facts.
- `models/v15_deployment_recipe.json`: exact deployment formula.
- `RESULT.md`: simple interpretation of every important result.
- `METHODOLOGY.md`: complete experiment design.
- `RUNBOOK.md`: inputs and operational steps.
- `AUDIT.md`: leakage and failure audit.

## Reproduce

Run from `/Users/aaryan/Downloads/ugp`:

```bash
python rapid_yield_forecast/v15_complete_hierarchy/scripts/build_v15_data.py
python rapid_yield_forecast/v15_complete_hierarchy/scripts/train_v15_encoder.py
python rapid_yield_forecast/v15_complete_hierarchy/scripts/run_v15_hierarchy.py
python rapid_yield_forecast/v15_complete_hierarchy/scripts/run_v15_learned_normal.py
python rapid_yield_forecast/v15_complete_hierarchy/scripts/run_v15_hierarchy_sensitivity.py
python rapid_yield_forecast/v15_complete_hierarchy/scripts/run_v15_integration.py
python rapid_yield_forecast/v15_complete_hierarchy/scripts/run_v15_distribution.py
python rapid_yield_forecast/v15_complete_hierarchy/scripts/audit_v15.py
python rapid_yield_forecast/v15_complete_hierarchy/scripts/finalize_v15_models.py
python rapid_yield_forecast/v15_complete_hierarchy/scripts/validate_v15.py
```

The encoder scripts use the Apple MPS GPU when available. Reported scores come
only from the strict held-year models. The through-2022 deployment refit has
`score_claimed_for_refit=false`.
