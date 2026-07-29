# V15 runbook

## Forecast date

The promoted V15 model is a **March 5** forecast. January and February remain
valid earlier V5/V14 outlook dates, but the new V15 crop correction has only
been strictly promoted at March 5.

## Inputs needed for one district

### Always required

- district ID and state;
- the latest district yield history;
- the V5 physical/economic feature row available by March 5;
- current-season Sentinel crop-index sequence;
- observed weather up to March 5;
- future-weather forecast sequence available on March 5;
- soil and phenology fields used by V5.

### Satellite sequence

Provide January 15, February 15 and March 5 Sentinel observations. The encoder
expects six crop indices over the stored 21-position representation. Missing
values may be masked, but the feature order must match the training manifest.

### Future weather

The model accepts the same ten-step, sixteen-variable future sequence used by
the V12/V15 dataset. Forecasts must be the issue available at forecast time;
realized later weather must never be inserted.

## Output

For every district, return:

- `v15_point_prediction`;
- `q05` through `q95`;
- `probability_rise`;
- `probability_severe_drop`;
- V5 conservative anchor;
- V14 future-weather shadow;
- V15 crop correction;
- forecast issue date and data-completeness flags.

Showing the three point components is important. A government user can see
whether a change came from history, future weather or current crop condition.

## Operational calculation

1. Run the existing V5 point model.
2. Run the V14 future-weather shadow bundle.
3. Create the V15 crop representation with both deployment encoders and average
   the two representations.
4. Run the base physical XGBoost bundle.
5. Run the crop-aware XGBoost bundle.
6. Calculate:

```text
V15 = V14_shadow + 1.25 × (crop_aware − base_physical)
```

7. Move the empirical quantiles to the V15 centre and multiply their distance
   from the centre by 0.95.
8. Interpolate the quantile curve to obtain rise and severe-drop probabilities.

## Deployment files

- `models/encoder_modis_pretrained_seed42_through2022_deployment.pt`
- `models/encoder_modis_pretrained_seed73_through2022_deployment.pt`
- `models/v15_xgb_base_physical_d2_through2022.joblib`
- `models/v15_xgb_current_physical_d2_through2022.joblib`
- `models/v15_deployment_recipe.json`

## Safety rules

- Never use a yield label newer than the model cutoff during evaluation.
- Never replace a forecast with realized future weather.
- Never use a later satellite observation for an earlier issue date.
- If Sentinel is missing or fails quality checks, fall back to V14/V5 rather
  than filling the crop correction with an arbitrary value.
- Display V5 beside V15 until additional unseen seasons validate V15.

## Revalidation

When a new season becomes available:

1. Freeze the model before reading the new yield.
2. Produce predictions for all 119 districts.
3. Add the season to the year and state-year audit.
4. Recompute grouped bootstrap intervals and probability coverage.
5. Promote V15 over V5 only after multiple new years retain the gain.
