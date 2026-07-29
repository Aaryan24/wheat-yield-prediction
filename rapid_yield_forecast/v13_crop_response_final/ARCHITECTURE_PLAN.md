# V13 final consolidation: forecast weather -> crop response -> yield

## Question

Can the future-weather forecast improve yield prediction after it is first forced to
learn a physical intermediate task: how the satellite-observed wheat crop changes
between January, February, and March?

This is deliberately different from V12. V12 trained the weather representation to
reconstruct future weather and then asked a small yield dataset to learn the
weather-to-crop-to-yield relationship. V13 first trains that relationship without
using yield labels.

## Strict information contract

- Forecast clocks: 15 January, 15 February, and 5 March.
- A weather forecast must have been issued at least two days before its forecast
  clock.
- The input satellite record stops at the forecast clock.
- January-to-February and February-to-March satellite transitions are training
  targets. They are never inputs to an earlier forecast.
- A historical yield test year is never used in either representation pretraining
  or the downstream yield/sign model that predicts that year.
- Model selection uses 2019 and 2020. The already reused 2021 and 2022 period is a
  confirmation period, not a model-selection period.
- No post-2022 yield label is read.

## Architecture

There are three short sequences:

1. Current crop state: six Sentinel-2 index tokens (NDVI, EVI, NDRE, NDMI, NIRv,
   and PSRI), each containing 21 wheat-mask summaries.
2. Weather already experienced: six crop-stage weather tokens.
3. Forecast weather: ten dated future-weather tokens.

Each sequence gets a small one-layer Transformer. Cross-attention then asks two
questions:

- For every crop-index token, which future-weather periods matter?
- For every crop-index token, which parts of the season already experienced matter?

The model predicts the change in 15 dynamic satellite summaries at the next
forecast clock. Its main loss is robust error on the size of the change. A smaller
loss also asks whether each satellite summary will rise or fall.

The representation is then frozen. Small regularised regression and classification
heads test whether it adds to:

- the current point-yield anchor;
- the current V11/V12 probability that yield will increase.

## Required controls

- Persistence: assume the next satellite state equals the current state.
- Crop only: current satellite state, without weather.
- No future: current crop plus weather already experienced.
- Full response model: current crop, experienced weather, and forecast weather.
- Downstream tabular control: frozen representation with and without the existing
  issue-safe tabular features.
- Point targets: yield from zero, percentage change from last yield, and residual
  percentage error left by the anchor.

## Promotion rules

### Crop trajectory

Future weather is called useful only if the full model beats both persistence and
the no-future model in development and does not reverse in the late period.

### Point yield

A candidate is selected only on 2019-2020 and must beat the anchor in both years
and at least four of six state-year cells. It is promoted only if it also beats the
anchor overall in 2021-2022, beats it in both late years, and has a positive
cluster-bootstrap lower bound.

### Increase/decrease probability

A blend weight is selected only on 2019-2020 using:

`AUC - 0.20 * Brier score`

It is promoted only if, in 2021-2022, AUC rises, Brier score does not worsen, and
the state-year cluster-bootstrap 95% interval for AUC gain is above zero.

If a V13 candidate fails, the final system retains the earlier component. A failed
experiment cannot make the deployed policy worse.

