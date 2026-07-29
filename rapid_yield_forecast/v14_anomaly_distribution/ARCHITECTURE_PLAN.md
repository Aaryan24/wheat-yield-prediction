# V14: district normal yield + seasonal anomaly + probability distribution

## Questions

1. Can a simpler model beat V5 by separating stable district productivity from
   the current season's positive or negative shock?
2. Does the V13 forecast of next-month crop condition improve a matched V5-style
   XGBoost when added as an issue-safe input?
3. Can the model return a useful probability distribution for every district,
   rather than only a point estimate?

## Core equation

For district \(d\) and season \(t\):

\[
\widehat y_{d,t}=b_{d,t}\exp(\widehat a_{d,t})
\]

where:

- \(b_{d,t}\) is normal district yield calculated only from earlier yields;
- \(a_{d,t}=\log(y_{d,t}/b_{d,t})\) is the seasonal anomaly;
- weather, satellite crop condition, soil, and forecast crop trajectory predict
  the anomaly, not the raw yield level.

This is a standalone model. It does not use V5 as its base.

## Data

- District yield: 119 districts, 2010-2022.
- Earlier history fields: up to ten lagged yields.
- MODIS crop state: January, February, and March summaries, 2000-2022.
- V5 physical/economic fields: issue-safe March 5 research feature table.
- V13 Sentinel crop response: 2017-2022.
- Future-weather forecast issue must precede the clock by at least two days.
- No post-2022 yield label may be opened.

## Strict V13 outlook feature construction

The next-crop-state model predicts satellite change using current crop,
experienced weather, and forecast weather. Downstream yield training rows use
district-group cross-fitted outlooks: a district's later satellite target is
excluded when producing its own training feature.

Historical yield test years use:

| Yield test year | Latest representation-pretraining season |
|---|---:|
| 2019 | 2018 |
| 2020 | 2019 |
| 2021 | 2020 |
| 2022 | 2020 |

## Track A: standalone anomaly model

Normal-yield controls:

- recent weighted history;
- five-year mean;
- five-year median;
- five-year linear trend;
- five-year exponentially weighted mean.

Seasonal-anomaly controls:

- zero anomaly;
- history only;
- physical weather/soil;
- strict MODIS crop anomalies;
- physical + MODIS;
- physical + MODIS + future crop outlook.

Regularised Ridge, Huber, Extra Trees, shallow XGBoost, and a state-shock model
are compared. Prediction is always reconstructed through the equation above.

## Track B: V5-style XGBoost outlook increment

Matched shallow XGBoost models use the same training years, target, seed, and
base features:

- V5 compact features;
- V5 compact + no-future crop outlook;
- V5 compact + future-conditioned crop outlook;
- V5 compact + isolated future-weather increment.

The selected challenger can then receive a development-selected bounded weight
over the frozen V5 point prediction.

## Probability distribution

For every district, prequential historical errors are turned into weighted
empirical distributions. Candidate calibration pools are:

- global residuals;
- equal-weight historical years;
- state/global shrinkage;
- state/global shrinkage after scaling by local historical variability.

The output stores 5th through 95th percentiles, mean, standard deviation,
probability of rising, and probability of a severe decline.

The first-pass distribution selection uses 2019-2020 mean pinball loss. The
final extension also penalises serious under-coverage of the intended 80% and
90% ranges, because a sharp but dishonest interval is not useful to a
government user. Width, coverage, and pinball loss are all reported.

## Promotion

2019-2020 selects recipes. The selected recipe is reused unchanged on
2021-2022.

A point model receives production promotion only if it:

- lowers pooled late RMSE;
- improves both late years;
- improves at least four of six late state-year cells;
- has a positive state-year cluster-bootstrap 95% lower bound.

The V13 outlook is credited only if the matched XGBoost with outlook beats the
matched no-outlook XGBoost under the same grouped test.

Probability distributions are promoted only if late mean pinball loss improves
without materially damaging 80% coverage.

Production promotion is not the same as scientific rejection. A candidate is
retained as a named shadow/research model when its 2021-2022 RMSE is within
10 kg/ha or 3% of the incumbent and it has stronger evidence elsewhere, such as:

- better pooled rolling 2016-2022 performance;
- more individual years or state-year cells improved;
- better increase/decrease performance;
- better probability-distribution score or calibration;
- a physically coherent future-crop contribution.

Every forecast year is reported separately. A small two-year RMSE difference
cannot erase useful multi-year or mechanistic evidence.

## Final outcome

- The standalone anomaly model remains a retained research fallback and does
  not replace V5.
- The isolated future-crop correction improves RMSE in all four forecast years
  and is retained as the V14 frontier shadow.
- Frozen V5 remains the safe point output until sealed-year confirmation.
- A full 5th-95th percentile distribution is now available for every district.
