# V13 methodology

## 1. Data

The yield panel contains 119 districts in Haryana, Punjab, and Uttar Pradesh,
six seasons from 2017 through 2022, and three forecast clocks. This gives 714
district-season yield labels and 2,142 clock-specific rows.

The crop-response task contains 1,428 label-free transitions:

- 714 January 15 -> February 15 transitions;
- 714 February 15 -> March 5 transitions.

Each clock-specific row contains:

- six wheat-mask Sentinel-2 index tokens;
- six experienced-weather/crop-stage tokens;
- ten future-weather tokens;
- 108 issue-safe history, global-transfer, economic, soil, phenology, and risk
  features.

The future-weather sequence is selected from the latest issue dated no later than
two days before the forecast clock. Realised future weather and later satellite
observations are never inputs.

## 2. Cleaning

The first five satellite indices have no values outside +/-2. PSRI has 96 ratio
singularities outside that range, including values above 60 and below -100.
Only those PSRI cells are set to missing. Missing values are replaced by the
training-fold centre after scaling.

## 3. Crop-response model

The model has 39,454 trainable parameters.

For each input type, a one-layer Transformer turns the dated/tokenised values
into a short internal representation. Cross-attention lets every satellite index
ask which experienced and future weather tokens matter.

It predicts changes in 15 dynamic satellite summaries:

- recent 40-day state;
- recent 20-day state;
- seasonal peak;
- growth;
- 20-day change;

for the generic crop, active-crop, and hybrid wheat-mask views.

If current crop state is \(c_t\), the model predicts:

\[
\widehat{\Delta c}_{t+1}=f(c_t,w_{\leq t},\widehat{w}_{>t})
\]

and:

\[
\widehat{c}_{t+1}=c_t+\widehat{\Delta c}_{t+1}.
\]

The main training error is robust error on the size of satellite change. A
smaller binary error asks whether each summary rises or falls.

Before this, the weather encoders learn forecast-to-realised-weather mapping from
8,330, 12,376, or 16,541 examples depending on the historical cutoff. The final
through-2022 representation uses 24,871 such examples.

## 4. Controls

Three otherwise matched response models are trained:

1. Crop only.
2. Crop plus weather already experienced.
3. Crop plus experienced and future weather.

Persistence—predicting no change—is also scored.

Every neural result averages seeds 42 and 73. The small model was faster on CPU
than MPS because each training batch is small; using the Mac GPU added transfer
and launch overhead.

## 5. Downstream heads

The response model is frozen before learning yield. Its representation contains:

- pooled current-crop state;
- pooled experienced weather;
- pooled future weather;
- cross-attended crop/weather state;
- predicted crop change and its sign probability.

Two feature sets are tested:

- frozen response representation only;
- frozen response representation plus the existing 108 issue-safe features.

Regularised linear heads then test:

- direct yield;
- percentage change from last yield;
- percentage residual left by V5/V7;
- probability yield rises.

The point heads use Ridge penalties 100, 1,000, and 10,000. Direction heads use
logistic strengths 0.005, 0.02, 0.1, and 0.5. Point corrections are blended with
the anchor at 0%, 10%, 25%, 50%, 75%, or 100%. Direction heads are blended with
the incumbent in 5% steps from 0% to 50%.

Across all clocks and folds, 936 downstream heads are fitted. Late years do not
choose among them.

## 6. Strict historical simulation

| Test year | Latest crop/yield pretraining year |
|---|---:|
| 2019 | 2018 |
| 2020 | 2019 |
| 2021 | 2020 |
| 2022 | 2020 |

2019-2020 selects the recipe. The same recipe is reused without tuning on
2021-2022. State-year cluster bootstraps resample the six state-year shocks
rather than pretending all 238 district rows are independent.

## 7. Promotion

Point yield must improve both development years and at least four of six
development state-year cells. It is promoted only if it also improves both late
years and the late grouped 95% RMSE-gain range stays above zero.

Direction uses development `AUC - 0.20 × Brier`. Promotion requires higher late
AUC, no worse late Brier, and a grouped 95% AUC-gain range above zero.

Future weather is strictly promoted for crop trajectory only if the full model
beats the no-future control and that isolated late gain has a grouped lower bound
above zero. It did not pass this last requirement.

## 8. Deployment refit

After evaluation is frozen, two-seed crop-response representations are refit
through 2022. The promoted March logistic head is also refit through 2022 and
saved. No score is claimed for this deployment refit; all reported scores come
from the historical simulations above.

