# V14 audit and evidence limits

## Validation

`scripts/validate_v14.py` passes 41 checks.

The checks confirm:

- exactly 119 districts in each of 2019, 2020, 2021, and 2022;
- unique district-year rows;
- no yield label after 2022;
- exact recomputation of V5 and V14 RMSE;
- improvement in all four forecast years;
- strictly increasing district quantiles;
- probabilities inside zero and one;
- every standalone fit trains only on earlier yield years;
- the crop-response representation uses no yield labels;
- later satellite condition is a training target, never an input;
- all three held-district cross-fit groups exist;
- all four XGBoost deployment components and both seeds load;
- the error-calibration pool spans 2016–2022; and
- recorded SHA-256 hashes match the release files.

The machine-readable result is `artifacts/validation.json`.

## Point-result uncertainty

The V14 correction improves pooled RMSE:

- +1.61 kg/ha over 2019–2022;
- +0.48 kg/ha over 2021–2022.

It improves all four individual years. Under a simple independent-year sign
test, four wins out of four has a one-sided probability of 6.25% under a
50/50 null.

The state-year grouped bootstrap is more conservative:

| Period | Mean gain | 95% range | Probability gain is positive |
|---|---:|---:|---:|
| 2019–2022 | +1.65 kg/ha | -0.18 to +4.45 | 95.7% |
| 2021–2022 | +0.39 kg/ha | -2.06 to +2.02 | 69.6% |

The four-year evidence is promising. The late-only evidence is not strong
enough for an unconditional production claim.

## Probability uncertainty

Against V5 on 2021–2022, V14 has better point AUC and Brier error for both
events. Grouped uncertainty still crosses zero:

| Comparison | Mean AUC gain | 95% range |
|---|---:|---:|
| Rise: V14 vs V5 | +0.010 | -0.006 to +0.024 |
| Severe fall: V14 vs V5 | +0.028 | -0.042 to +0.078 |

Against V13, V14's Brier-error gain is positive in all 5,000 grouped bootstrap
draws for both events. The severe-fall AUC gain also has a positive 95% lower
bound.

This supports retaining V14 probabilities as the leading research output, not
pretending their smaller gain over V5 is already certain.

## Data limitations

The effective sample is not 1,547 independent observations. Districts within
the same year share weather, policy, measurement changes, and regional shocks.
The number of independent unusual seasons is much smaller.

Important limitations are:

- only four fully aligned forecast years for the strict future-crop feature;
- only two reused late-confirmation years;
- state coverage is limited to Haryana, Punjab, and Uttar Pradesh;
- future-weather contribution to crop-trajectory accuracy is small;
- crop masks and satellite composites still contain measurement noise;
- district yield statistics may be revised or measured inconsistently; and
- the deployment refit through 2022 has no claimed score of its own.

## Leakage controls

The outlook representation obeys two separate timing rules:

1. the weather forecast issue precedes the prediction clock; and
2. a training district's later satellite target is excluded from the
   representation model used to create that district's yield feature.

Historical yield models train only through the year before their test season.
The 2021 and 2022 outlook test features come from a representation cutoff at
2020.

## Promotion decision

| Component | Status | Reason |
|---|---|---|
| V5 point | Safe incumbent | Strongest confirmed point base |
| V14 future-crop point | Frontier shadow | Four of four years improve; CI narrowly crosses zero |
| V14 primary ranges | Preferred research distribution | Good calibration and sharper than old V5 envelope |
| V14 event probabilities | Leading research probabilities | Better point metrics; small V5 gain remains uncertain |
| Standalone anomaly model | Retained fallback | Simpler and useful, but late RMSE gap is large |

The frozen 2023–2024 evaluation is the next promotion gate.

