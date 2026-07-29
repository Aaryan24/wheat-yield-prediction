# V13 final result

## Bottom line

V13 produced one strict, useful improvement and one promising research signal.

The strict improvement is the **March 5 probability that district yield will rise
or fall**. On the untouched 2021-2022 confirmation period:

| March 5 trend output | Before V13 | Final V13 |
|---|---:|---:|
| AUC | 0.7682 | **0.7956** |
| Brier error (lower is better) | 0.2057 | **0.1970** |
| Correct at a plain 50% cutoff | — | **73.5%** |

The grouped 95% range for the AUC gain is **+0.0030 to +0.0540**. It stays above
zero, so this is not just a pooled win caused by treating every district as an
independent year.

No V13 point-yield correction survived the same confirmation test. The final
system therefore keeps the stronger existing point forecast:

- January 15: V7 point forecast;
- February 15: V7 point forecast;
- March 5: V5 point forecast.

That is the correct outcome. The final model improves only the component for
which there is repeatable evidence and cannot make the point forecast worse.

## Final model by forecast date

| Output | January 15 | February 15 | March 5 |
|---|---|---|---|
| Yield in kg/ha | locked V7 | locked V7 | locked V5 |
| Chance yield rises | V11 | V12 blend | **V13 blend** |
| Chance of severe decline | V11 severe sidecar | V11 severe sidecar | V11 severe sidecar |
| Yield ranges | V7 calibrated ranges | V7 calibrated ranges | V5 calibrated ranges |
| Future crop trajectory | V13 research output | V13 research output | not enough later Sentinel data |

The March V13 trend probability is:

`0.85 × previous V12 probability + 0.15 × new V13 probability`

The 15% weight was chosen only on 2019-2020. It was then left unchanged for
2021-2022.

## Point-yield accuracy

The figures below are RMSE in kg/ha. Lower is better.

| Clock | 2019-2020 | 2021-2022 |
|---|---:|---:|
| January 15 | 296.75 | **316.85** |
| February 15 | 296.75 | **316.85** |
| March 5 | 257.00 | **288.61** |

January and March V13 corrections initially looked useful on 2019-2020 but did
not transfer:

- January: 296.75 -> 276.27 in development, but 316.85 -> 321.58 late.
- March: 257.00 -> 252.85 in development, but 288.61 -> 301.55 late.

They are rejected. February selected zero correction.

## Yield ranges

On 2021-2022:

| Clock | Intended range | Actual coverage | Mean total width |
|---|---:|---:|---:|
| January 15 | 80% | 79.0% | 912 kg/ha |
| February 15 | 80% | 79.0% | 912 kg/ha |
| March 5 | 80% | 78.2% | 650 kg/ha |

In plain language, the typical 80% range is about point prediction plus or minus
456 kg/ha in January/February and plus or minus 325 kg/ha on March 5.

## What happened with future weather?

The new Transformer was first trained to predict how the wheat-only satellite
state changes from January to February and from February to March. This uses no
yield labels.

Late-period satellite-transition RMSE was:

| Crop-trajectory model | RMSE |
|---|---:|
| Assume no crop change | 0.12830 |
| Current crop only | 0.07011 |
| Current crop + experienced weather | 0.06498 |
| Current crop + experienced and forecast weather | **0.06445** |

This is real progress over simply carrying the current crop state forward. The
full model also beats crop-only with a grouped 95% gain range of +0.00125 to
+0.01121.

But the isolated gain from adding *future* weather over experienced weather is
only +0.00074, with a 95% range of -0.00241 to +0.00372. Therefore:

- future weather contains plausible crop-trajectory information;
- the extra contribution is not yet stable enough to force into district yield;
- the final March direction gain actually comes from current crop state, past
  weather, and the existing issue-safe tabular information—not future weather.

## What the model needs for a live prediction

For one district and one forecast date:

1. District identity and season.
2. At least the previous three district yields.
3. Sentinel-2 wheat-mask summaries available by that date: NDVI, EVI, NDRE,
   NDMI, NIRv, and PSRI.
4. Weather experienced so far, arranged into six crop-stage summaries.
5. Ten dated future-weather forecast summaries from an issue at least two days
   old. These drive the research trajectory, not the promoted March trend head.
6. Only lagged economic information that would really be known at prediction
   time.
7. Static soil, drainage, crop-area, and phenology information.

The live outputs are:

- expected yield in kg/ha;
- 50%, 80%, and 90% yield ranges;
- probability yield rises versus last season;
- probability of a severe decline;
- a research outlook for the next satellite-observed crop state.

## Important cleaning discovery

PSRI is a ratio. When its denominator was almost zero, 96 cells became impossible
values such as -102 or +63. All other normal PSRI data were close to their
physical range. V13 treats only `abs(PSRI) > 2` as missing. This rule uses no
yield, year, district, or prediction result, and the entire final experiment was
rerun after applying it.

## Final interpretation

V13 does not solve district point-yield forecasting. It does produce the
strongest March increase/decrease output in the project so far, while preserving
the best point model and honest uncertainty ranges. It also shows that a
weather-conditioned crop-trajectory model can learn something genuine, but more
seasons or denser satellite targets are needed before its future-weather branch
should change the official yield number.

