# V13 crop-response final model

V13 is the final strict consolidation of the rapid wheat-yield research line.
It preserves the best V5/V7 point forecast, V7 uncertainty ranges, and V11/V12
risk components, and promotes one new V13 component: the March 5 probability
that yield increases.

Start with:

- `RESULT.md` for the plain-language result;
- `METHODOLOGY.md` for the model and mathematics;
- `AUDIT.md` for leakage, instability, and rejection checks;
- `artifacts/final_policy.json` for the machine-readable policy;
- `artifacts/deployment_manifest.json` for the exact live input/output contract.

Main reproducible commands:

```bash
V13_EPOCHS=60 V13_SEEDS=42,73 V13_DEVICE=cpu \
python rapid_yield_forecast/v13_crop_response_final/scripts/run_v13_final.py

python rapid_yield_forecast/v13_crop_response_final/scripts/finalize_v13.py
python rapid_yield_forecast/v13_crop_response_final/scripts/validate_v13.py
```

The final historical district predictions, probabilities, targets, and intervals
are in `artifacts/final_predictions.parquet`.

