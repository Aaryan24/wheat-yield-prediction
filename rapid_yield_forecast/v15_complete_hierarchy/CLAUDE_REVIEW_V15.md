# V15 review handoff

Please audit this as a serious district wheat-yield forecasting result. Try to
falsify it rather than merely suggest more model complexity.

## Objective

Predict wheat yield for 119 districts in Haryana, Punjab and Uttar Pradesh using
only information available by March 5: recent yield history, weather observed
so far, future-weather forecasts, soil/phenology, lagged economic context and
current-season satellite crop condition.

## Result to audit

```text
V15 = V14 future-weather shadow
      + 1.25 × (MODIS-pretrained current-crop XGBoost
                − matched physical-only XGBoost)
```

Strict metrics:

- V5 all-four-year RMSE: 273.2618 kg/ha.
- V14 all-four-year RMSE: 271.6517 kg/ha.
- V15 all-four-year RMSE: 269.5239 kg/ha.
- V5 2021–2022 RMSE: 288.6100 kg/ha.
- V15 2021–2022 RMSE: 287.2004 kg/ha.
- V15 improves three of four years and eight of twelve state-years.
- Four-year state-year bootstrap 95% gain range: 0.18 to 8.06 kg/ha.
- Four-year year-bootstrap 95% gain range: −0.57 to 6.67 kg/ha.

Selection is on 2019–2020. Confirmation is 2021–2022. Both 2021 and 2022 models
are trained only through 2020.

## Seven implemented stages

1. 1990–2022 history and learned 10–20 year district normal.
2. Log percentage anomaly models.
3. MODIS 2000–2022 temporal pretraining.
4. Explicit state seasonal shock prediction.
5. Shrunk district exposure to state shock.
6. Sentinel crop/weather cross-attention fine-tuning.
7. District q05–q95 distribution and rise/severe-drop probabilities.

Only MODIS/Sentinel transfer and the distribution are promoted. The learned
normal and explicit hierarchy fail to transfer safely and remain shadow
evidence.

## Specific questions

1. Is the district cross-fitting and held-year timing genuinely leak-free?
2. Is subtracting the matched physical-only XGBoost a valid isolation of the
   crop representation?
3. Is 1.25 a defensible near-tie correction weight, or should the 0.125
   conservative shadow be operationally preferred?
4. Does the grouped bootstrap treat the true independent units correctly?
5. Is the 0.95 distribution scaling rule defensible?
6. Which single new experiment has the best chance of turning this small gain
   into a reliable gain across new years?
7. Can the failure of the state hierarchy be repaired without using the four
   test years for model selection?

Read these files:

- `README.md`
- `RESULT.md`
- `METHODOLOGY.md`
- `AUDIT.md`
- `artifacts/validation.json`
- `artifacts/point_model_ablation_metrics.csv`
- `artifacts/point_year_state_audit.csv`
- `artifacts/point_grouped_bootstrap.csv`
- `artifacts/seven_stage_evidence.csv`
- `artifacts/release_manifest.json`

Please return:

- any leakage or evaluation flaw;
- any arithmetic mismatch;
- whether V15 should remain a challenger or be promoted;
- the strongest repair experiment, with an exact protocol;
- a short final verdict in plain language.
