# V15 methodology

## 1. First-principles design

District wheat yield has three main pieces:

```text
ordinary district level
+ common seasonal shock
+ local response to that shock
```

In log form, the explicit hierarchy is:

```text
log(yield[d,t])
= log(normal[d,t])
+ intercept[d]
+ exposure[d] × state_shock[state,t]
+ local_residual[d,t]
```

The final promoted model uses a safer additive correction form:

```text
V15 point
= V14 future-weather shadow
+ 1.25 × (crop-aware XGBoost − matched crop-blind XGBoost)
```

Subtracting the matched model isolates what the new crop representation added.
It prevents unrelated XGBoost differences from being mislabelled as satellite
signal.

## 2. Data

### Yield

- Official DES district wheat yield: 2010–2022.
- ICRISAT district wheat yield: 1990–2019.
- Overlap check on 1,177 district-years: correlation 0.9952, MAE 16.25 kg/ha,
  mean difference 1.64 kg/ha.
- Combined long table: 3,658 rows, 119 present-day districts, 1990–2022.
- Six post-split districts use documented parent-district history proxies before
  their own records begin.

### Satellite

- MODIS history: 2000–2022, January 15, February 15 and March 5 clocks.
- Sentinel crop sequence: 2017–2022.
- Sentinel inputs contain six crop indices over 21 spatial/temporal positions.
- Extreme PSRI values outside the physically valid audit range are masked.

### Weather, soil and economic context

The V5 physical table supplies:

- observed seasonal weather;
- strict future-weather/reforecast features;
- soil and phenology;
- recent district and state yield history;
- available lagged economic variables.

Only values available by the March 5 forecast date are used in the promoted
V15 experiment.

## 3. Strict evaluation

The final comparison uses:

- 2019 model trained through 2018;
- 2020 model trained through 2019;
- 2021 model trained through 2020;
- 2022 model also trained through 2020.

The 2022 model does not learn from the 2021 yield. This is the strict
reforecast-baseline setup used throughout the project.

Hyperparameters and the correction weight are selected on 2019–2020.
2021–2022 are untouched confirmation years.

For encoder features used to train a downstream yield model, districts are
split into three stable groups. Each training district is represented by an
encoder that excluded its group. Test-year districts use the full encoder
trained only through the allowed cutoff.

## 4. Stage 1: 10–20 year district normal

Ten history-only candidates were trained or calculated:

- three-year weighted history;
- ten-year mean;
- twenty-year trend;
- twenty-year exponential mean;
- Ridge at three strengths;
- Extra Trees;
- depth-1 and depth-2 XGBoost.

Features include twenty district lags, availability masks, 3/5/10/20-year
means, spread, min, max and slope, five state lags and state identity.

A second meta-regression learns the remaining error after the robust
three-year normal:

```text
normal_meta = weighted3 + learned_correction
```

This is a real regression setup rather than a hand-selected average. It was not
promoted because it failed to transfer to 2021–2022.

## 5. Stages 2, 4 and 5: anomaly hierarchy

For each normal candidate:

1. Convert district yield to a log percentage anomaly.
2. Average anomalies within state-year to form the state shock target.
3. Predict the state shock from lagged shocks and physical/encoder context.
4. Estimate how sensitive each district is to state shocks.
5. Predict the district residual left after the state shock.

District exposure is shrunk toward 1:

```text
exposure = (n × raw_exposure + 8 × 1) / (n + 8)
```

This stops a district with only a few years from receiving an extreme response.

Five state models, six residual models, three encoder settings and several
feature groups were crossed. Across the stable-normal rerun, 774 hierarchy
candidates were evaluated.

## 6. Stages 3 and 6: MODIS-to-Sentinel encoder

### MODIS pretraining

The model sees the MODIS sequence up to one clock and predicts the next clock.
This produces thousands of learning examples without requiring yield labels.

### Sentinel fine-tuning

The same temporal encoder is adapted to Sentinel and trained to predict the next
crop-index movement. It contains:

- a crop sequence Transformer;
- a current/observed weather Transformer;
- a future-weather Transformer;
- cross-attention from crop condition to observed weather;
- cross-attention from crop condition to future weather;
- a next-crop-change head;
- a sign head.

Thirty percent future-branch dropout teaches one network both:

- current crop trajectory without future weather;
- current crop trajectory with future weather.

Two random seeds are averaged.

### Representation tests

Forty-eight independent XGBoost combinations were tested:

- Sentinel-only versus MODIS-pretrained;
- physical versus physical-plus-MODIS base inputs;
- depth 1 versus depth 2;
- base, current, full-future, future-effect, transition and pooled
  representation groups.

Sixty-four isolated correction definitions were then tested beside V5 and V14.

## 7. Final integration

The selected correction is:

```text
crop_correction
= XGB(physical + MODIS-pretrained current crop representation)
 − XGB(the same physical features only)
```

Both models have depth 2 and two seeds. The correction multiplier is 1.25, the
smallest near-tie weight under the development selection rule.

The V14 anchor supplies a separate future-weather correction. Therefore the
promoted V15 combines:

- present yield level;
- future weather;
- current satellite crop condition.

## 8. Probability distribution

V14's strict empirical district residual distribution is moved so that its
centre equals the V15 point.

The distance of each quantile from the centre is multiplied by 0.95:

```text
V15_quantile[q]
= V15_point + 0.95 × (V14_quantile[q] − V5_point)
```

The scale is selected only on 2019–2020. A candidate must cover between 78% and
82% for its stated 80% range and at least 88% for its stated 90% range before
pinball loss can select it. This prevents narrow, overconfident ranges.

Rise and severe-drop probabilities are obtained by interpolating the final
district quantile curve.

## 9. Deployment refit

After evaluation is frozen:

- two MODIS-pretrained encoders are refit through 2022 on the M4 GPU;
- the matched physical and crop-aware XGBoost models are refit through 2022;
- their bundles are saved in `models/`.

These models are for future inference only. They are marked
`score_claimed_for_refit=false` and are never used to calculate the reported
2019–2022 result.
