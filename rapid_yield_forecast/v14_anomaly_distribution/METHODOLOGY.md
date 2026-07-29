# V14 methodology

## Questions tested

V14 answers four questions.

1. Can we predict seasonal percentage change instead of raw district yield?
2. Does selecting with more historical forecast years change that conclusion?
3. Does the next-month crop outlook add useful information to XGBoost?
4. Can we return an honest probability distribution for every district?

## Data and timing

The yield panel contains 119 districts in Haryana, Punjab, and Uttar Pradesh.
Yield labels end in season-start year 2022.

Inputs include:

- up to ten earlier district yields;
- weather and crop-stage stress summaries;
- strictly lagged economic variables;
- static soil, drainage, crop area, and crop calendar fields;
- issue-safe March MODIS crop-condition history;
- current Sentinel-2 wheat-only crop state;
- weather experienced before the forecast date; and
- dated future-weather summaries from forecasts issued before the decision
  date.

March 5 is the clock in this experiment. Later observed weather and later
satellite crop state are not input features.

## Part A: independent anomaly prediction

### Target

For district \(d\) and year \(t\):

\[
a_{d,t}=\log\left(\frac{y_{d,t}}{b_{d,t}}\right).
\]

The model learns \(a\), not \(y\). The final answer is:

\[
\widehat y_{d,t}=b_{d,t}\exp(\widehat a_{d,t}).
\]

Five normal-yield values \(b\) were tried:

- 60/25/15 weighted last-three yield;
- five-year mean;
- five-year median;
- five-year linear trend;
- five-year exponentially weighted mean.

The anomaly models were:

- zero anomaly;
- Ridge regression with two regularisation levels;
- robust Huber regression;
- shallow Extra Trees;
- depth-one XGBoost;
- depth-two XGBoost; and
- a state-shock regression with district-specific exposure.

Feature families were:

- yield history;
- physical, soil, and lagged economic inputs;
- strict MODIS crop condition;
- physical plus MODIS;
- future-crop outlook only;
- physical plus MODIS plus compact outlook; and
- physical plus MODIS plus the broad outlook.

This produced 250 candidates and 170,765 district predictions. Every yield test
fit used only earlier seasons.

### Shrinkage and multi-year selection

The learned anomaly was also reduced by:

\[
\widehat a_{\mathrm{used}}=c+w\widehat a,
\]

where \(w\) ranged from 0.0 to 1.2. The offset \(c\) was calculated only on the
selection years and capped at ±0.10.

Four honest selection views were run:

| Selection years | Candidates | Parameter combinations |
|---|---:|---:|
| 2016–2018 | 145 | 1,885 |
| 2017–2020 | 145 | 1,885 |
| 2018–2020 | 145 | 1,885 |
| 2019–2020 | 250 | 3,250 |

This directly tests whether the model only looks weak because 2021–2022 is
unusual. It does not: recent-history selection improves stability, but the late
RMSE remains around 329–331 kg/ha.

## Part B: future crop trajectory

The crop-response network has about 39,000 parameters. It predicts how six
wheat-only satellite measurements change by the next observation:

- NDVI;
- EVI;
- NDRE;
- NDMI;
- NIRv; and
- PSRI.

It has two versions:

- `no_future`: current crop state plus weather already experienced;
- `full`: the same inputs plus dated future weather.

The full model lets current crop state ask, through attention, which future
weather periods matter. Date labels tell it whether a weather item is near the
current date, during grain filling, or later.

This representation is trained with satellite change as its target. It uses no
yield labels.

The strict feature table contains:

- 4,641 district-date rows;
- 62 compact outlook fields;
- two neural-network seeds;
- 60 epochs per fit;
- three held-district groups for cross-fitted training features; and
- representation cutoffs of 2018, 2019, and 2020.

A training district never receives an outlook representation fitted using its
own later satellite target.

## Part C: matched XGBoost test

Depth-one and depth-two XGBoost models were fitted with identical target,
training years, seeds, and base features. The only difference was the outlook
view:

- no outlook;
- no-future crop response;
- compact full response;
- isolated future effect; and
- broad full response.

Directly blending a whole outlook XGBoost into V5 was almost tied and slightly
worse late. That test mixed two things: the future signal and the second
model's unrelated errors.

The improved test isolates the difference. It constructs 18 possible
corrections such as:

\[
\Delta=
\widehat y_{\mathrm{with\ future}}
-\widehat y_{\mathrm{without\ future}}.
\]

Each correction was added to V5 with 29 weights from -2.0 to 5.0, giving 522
combinations. The recipe was selected on 2019–2020.

The exact best development weight was 2.25. A stability rule treats scores
within 0.1% as tied and chooses the smaller correction. That locked weight is
1.75. It performs better late than the exact optimum and avoids a needlessly
large adjustment, without using late labels to choose the rule.

## Part D: probability distribution

The distribution is empirical. It does not assume that yield error follows a
perfect bell curve.

For every earlier out-of-sample prediction:

\[
r_{d,t}=
\frac{y_{d,t}-\widehat y_{d,t}}
{\max(\text{recent yield SD},0.07b_{d,t},150)}.
\]

All years receive equal total weight. A district's state errors are mixed with
the global pool and shrunk toward global errors when the state sample is
small.

For a new district:

1. calculate its local yield scale;
2. take weighted 5th–95th percentiles of historical normalised errors;
3. multiply them by the local scale;
4. centre the shape on V5; and
5. read rise and severe-fall probabilities from the resulting cumulative
   distribution.

Twenty history-shape choices were tested: four error pools times five width
scales. Three direct V5-error distributions, one quantile-shift calibration,
and the outlook-centred shadow were also audited.

The primary method was chosen on 2019–2020 for low percentile error while
avoiding badly under-covered 80% and 90% ranges. It was then reused unchanged
on 2021–2022.

## Evaluation rules

- No yield label after 2022 is read.
- Every rolling yield fit trains only on earlier years.
- 2019–2020 selects outlook recipes.
- 2021–2022 is reused confirmation.
- Results are shown for every forecast year and every state-year.
- State-year grouped bootstraps preserve shared regional shocks.
- A candidate within 10 kg/ha or 3% of V5 is retained as a shadow when it has
  coherent evidence elsewhere.
- Retaining a shadow is not the same as declaring production superiority.

