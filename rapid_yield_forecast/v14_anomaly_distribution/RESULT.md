# V14 final result

## What we learned

There are three distinct conclusions.

First, predicting only the yield anomaly is scientifically sensible, but the
standalone version is not yet as accurate as V5. It should remain available as
a simpler fallback and research model.

Second, the predicted future crop condition contains a small amount of useful
yield information. The useful part is not the full output of another XGBoost
model. It is the *difference* between otherwise matched models with and without
the future-crop outlook.

Third, a district probability distribution can be added around V5 without
making an arbitrary normal-bell-curve assumption. The resulting 50%, 80%, and
90% ranges are close to their intended coverage and are narrower than the
earlier conservative V5 range.

## Best point result

The locked V14 shadow correction is:

\[
\widehat y_{\mathrm{V14}}
=
\widehat y_{\mathrm{V5}}
+1.75\left[
\frac{
\widehat y_{\mathrm{full}}
+\widehat y_{\mathrm{effect}}
+\widehat y_{\mathrm{broad}}
}{3}
-\widehat y_{\mathrm{no\ future}}
\right].
\]

In plain language:

1. predict yield with a crop outlook that has seen future weather;
2. predict it again with the future-weather part hidden;
3. repeat this through three views of the outlook;
4. average the three future-aware answers;
5. keep only their difference from the no-future answer; and
6. add a small version of that difference to V5.

This isolates the proposed future-crop signal. It does not replace V5 with a
weaker raw XGBoost.

| Season | V5 RMSE | V14 shadow RMSE | Gain |
|---|---:|---:|---:|
| 2019 | 277.97 | **275.38** | +2.58 |
| 2020 | 234.16 | **230.88** | +3.28 |
| 2021 | 286.55 | **285.79** | +0.76 |
| 2022 | 290.66 | **290.45** | +0.20 |
| **2019–2022 together** | 273.26 | **271.65** | **+1.61** |
| **2021–2022 together** | 288.61 | **288.13** | **+0.48** |

It improves all four years and 7 of 12 state-year cells. A grouped bootstrap
assigns a 95.7% probability that the four-year pooled RMSE gain is positive,
but the 95% range is -0.18 to +4.45 kg/ha. The lower end is still slightly
negative.

The correct decision is therefore:

- keep frozen V5 as the safe point forecast;
- keep V14 as the frontier shadow forecast;
- show both during the next live season; and
- decide promotion only after the sealed 2023–2024 test.

## Probability distribution

For every district, V14 returns 19 yield percentiles from the 5th to the 95th
percentile. This supports:

- a central yield estimate;
- 50%, 80%, and 90% likely ranges;
- probability that yield rises from last season; and
- probability of a fall of at least 10%.

The primary range uses frozen V5 as its centre. Its shape comes from strictly
historical forecast errors, with years weighted equally and a state-specific
part shrunk toward the national error pool.

### Range quality on 2021–2022

| Range | Intended coverage | Actual coverage | Mean total width |
|---|---:|---:|---:|
| Central 50% | 50% | **52.1%** | 329 kg/ha |
| Central 80% | 80% | **79.0%** | 661 kg/ha |
| Central 90% | 90% | **89.1%** | 942 kg/ha |

The earlier conservative V5 80% range covered 80.25% with a width of
711 kg/ha. V14 covers 78.99% with a width of 661 kg/ha. It gives up about
1.3 percentage points of coverage while becoming about 7.1% narrower.

For risk-averse use, V14 also stores a wider range. Its late 80% range covers
83.6% and has a width of 793 kg/ha.

### Rise and severe-fall probabilities

| Output, 2021–2022 | V13 AUC | V5 AUC | V14 shadow AUC |
|---|---:|---:|---:|
| Yield rises | 0.796 | 0.833 | **0.843** |
| Yield falls at least 10% | 0.564 | 0.742 | **0.786** |

AUC can be read as a ranking score. An AUC of 0.843 means that if one rising
district and one non-rising district are chosen at random, the model gives the
rising district the higher probability about 84% of the time.

V14 also has lower probability error than V5:

| Event | V5 Brier error | V14 Brier error |
|---|---:|---:|
| Rise | 0.1645 | **0.1599** |
| Severe fall | 0.0909 | **0.0877** |

The gain over V5 is promising but not statistically settled. The gain over the
V13 severe-fall probability is much clearer. The full grouped comparisons are
in `artifacts/probability_group_bootstrap.json`.

## Standalone anomaly model

The independent model predicts:

\[
\widehat y=b\exp(\widehat a),
\]

where \(b\) is normal district yield from earlier seasons and \(\widehat a\) is
the predicted positive or negative shock.

It was tested with five definitions of normal yield, seven model settings, and
weather, soil, economics, MODIS crop condition, and future-crop features. The
development-selected model is a shallow XGBoost using weighted recent yield as
the normal level.

| Independent anomaly model | RMSE |
|---|---:|
| 2017–2022 diagnostic | 301.87 |
| 2019–2022 | 306.48 |
| 2021–2022 | 328.47 |
| V5, 2021–2022 | **288.61** |

Shrinking its correction, changing its normal-yield formula, using a robust
linear model, trees, state shocks, and selecting with three or four earlier
years did not close the gap. The best recent-four-year selection scored
330.79 kg/ha late.

The important failure mode is a shared year shift. The district anomaly model
can rank many districts correctly, but it cannot reliably know how the entire
region moves in a new unusual year. V5's agreement gate handles that problem
better.

This model is not deleted. It is packaged as a fallback and remains useful for
longer rolling research. Its gap from V5 is large enough that it should not
replace V5 today.

## What a live district result looks like

The final district output contains:

- `production_point_prediction`: frozen V5;
- `shadow_point_prediction`: V14 future-crop challenger;
- `q05` through `q95`: the full yield distribution;
- `probability_rise`;
- `probability_severe_drop`;
- `conservative_q10` and `conservative_q90`; and
- the raw future-crop increment, so the reason for any V14 adjustment is
  inspectable.

The complete table is
`artifacts/final_predictions.parquet`.

## Honest limitation

The gain is not a large breakthrough in point RMSE. It is a small, physically
coherent signal that repeats in four years and was hidden when the full second
model was blended directly.

All 2021–2022 results are reused confirmation, not a never-seen test. No
post-2022 yield label was read. The decisive next step is an exactly frozen
2023–2024 evaluation.

