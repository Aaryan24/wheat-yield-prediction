# V15 results in simple language

## What improved

V5 was already strong because most district yields are close to their recent
history. V15 does not ask a new deep model to relearn that fact. It asks two
smaller questions:

1. Does the future-weather outlook imply a change from the no-future outlook?
2. Does the crop currently look better or worse than the ordinary physical
   XGBoost expects?

Those two corrections are added to the stable V5 anchor.

The final four-year RMSE is **269.52 kg/ha**, compared with **273.26 kg/ha** for
V5. That is a **3.74 kg/ha** improvement. On the untouched 2021–2022 years, the
improvement is **1.41 kg/ha**.

This is not a dramatic leap. It is important because:

- the correction was chosen on 2019–2020;
- it still helped when moved to 2021–2022;
- it improves three of four individual years;
- it improves eight of twelve state-years;
- the state-year grouped bootstrap gives a 98.0% chance that the four-year gain
  is positive.

The year-level bootstrap is weaker because there are only four years. Its 95%
range still crosses zero. Therefore V15 is called a **frontier challenger**, not
a proven production replacement.

## Year-by-year result

| Season start | V5 RMSE | V14 RMSE | V15 RMSE | V15 gain vs V5 |
|---|---:|---:|---:|---:|
| 2019 | 277.97 | 275.38 | **270.87** | **+7.10** |
| 2020 | 234.16 | 230.88 | **228.55** | **+5.61** |
| 2021 | 286.55 | 285.79 | **281.09** | **+5.45** |
| 2022 | **290.66** | 290.45 | 293.18 | **−2.52** |

The 2022 miss is why the crop correction is not presented as universally safe.
A very conservative V15 shadow, using only one tenth as much crop correction,
improves all four years but has a smaller pooled gain.

## What the satellite Transformer learned

The encoder predicts how the crop's satellite measurements will move from one
observation date to the next.

| Encoder | Future weather used? | Transition RMSE | “No crop change” RMSE |
|---|---|---:|---:|
| Sentinel only | Yes | 0.04179 | 0.08762 |
| MODIS-pretrained | Yes | **0.04163** | 0.08762 |
| MODIS-pretrained | No | 0.04268 | 0.08762 |

Three things are visible:

1. It learns a real crop-trajectory signal: the error is less than half the
   “assume no change” error.
2. Future weather makes the trajectory forecast slightly better.
3. Pretraining on MODIS 2000–2022 makes it slightly better again.

The improvement is small because Sentinel itself has only six seasons. The
useful V15 yield correction came from the **current/no-future crop
representation**, while V14 already supplied the separate future-weather
correction.

## What did not work

### Learned 10–20 year district normal

The pooled regression normal reached 292.06 RMSE on 2019–2020, slightly better
than the fixed three-year normal at 296.90. It then deteriorated to 404.25 on
2021–2022. The fixed three-year normal was 365.68 there.

Reason: long-term trend fitting can mistake an old growth path for the present
district level. Recent history adapts faster to reporting changes, management
changes and local breaks.

The learned normal is implemented and saved, but it is not promoted.

### State shock followed by district exposure

The best independent hierarchy on the selection years reached 272.99 RMSE and
showed useful direction information. Its blend with V5 improved 2019–2020 by
about 2 kg/ha but became worse in 2021–2022.

The decomposition is sensible:

```text
district anomaly
= district intercept
+ district sensitivity × predicted state shock
+ district-specific residual
```

The failure is data quantity, not arithmetic. There are too few independent
state-seasons for a flexible state shock model to remain stable. It remains a
well-documented shadow experiment.

### Independent deep/XGBoost yield model

The encoder-augmented XGBoost alone is much worse than V5. The deep
representation is useful as a **small correction**, not as a replacement for
the recent-history anchor.

## Probability output

Every district receives:

- one point prediction;
- q05, q10, ..., q95;
- probability that yield rises from last year;
- probability of a fall greater than 10%.

The released range is deliberately wider than the mathematically sharpest
candidate because the sharp candidate was overconfident.

| Period | 80% range coverage | 90% range coverage | Rise AUC | Severe-drop AUC |
|---|---:|---:|---:|---:|
| 2019–2020 | 78.6% | 91.6% | 0.851 | 0.824 |
| 2021–2022 | 78.2% | 87.0% | 0.837 | 0.799 |
| 2019–2022 | 78.4% | 89.3% | 0.845 | 0.809 |

An AUC near 0.84 means that if one rising district-season and one non-rising
district-season are chosen at random, the model ranks the rising one as more
likely about 84% of the time.

## Bottom line

The strongest current forecast is V15 combined:

- V5 for the stable yield level;
- V14 for future-weather tendency;
- V15 MODIS-to-Sentinel encoder for current crop condition;
- a calibrated district probability distribution around the result.

Use it as the research/frontier output. Keep V5 beside it as the conservative
official anchor until more unseen seasons confirm the gain.
