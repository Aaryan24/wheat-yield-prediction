# V13 audit

| Check | Result |
|---|---|
| Maximum opened yield season | 2022 |
| Districts | 119 |
| Yield seasons | 6 |
| Forecast clocks | January 15, February 15, March 5 |
| Crop transitions | 1,428 |
| Later satellite used as input | No |
| Realised future weather used as input | No |
| Forecast latency | 2 days |
| Neural seeds | 42 and 73 |
| Crop-response parameters | 39,454 |
| Impossible PSRI cells removed | 96 |
| Development selection | 2019-2020 only |
| Late confirmation | 2021-2022 |
| Point correction promoted | None |
| Direction correction promoted | March 5 only |
| Future-weather trajectory strictly promoted | No |

## Leakage checks

- For every historical test year, weather/crop representation pretraining ends
  before that year.
- For 2021 and 2022, all fitting ends in 2020.
- Satellite observations from the target date are response targets only, never
  earlier-clock inputs.
- Forecast issue selection is mechanical and uses a two-day operational delay.
- The PSRI cleaning rule is independent of yield and prediction error.
- The deployment refit through 2022 is separated from scored historical models.

## Instability found and handled

The first complete run exposed PSRI singularities that made late crop-trajectory
RMSE appear near 1.45 even though normal sample errors were around 0.03. The
largest transition was an impossible 102-unit PSRI change. The ratio failure was
removed with a physical rule and the entire experiment was rerun.

After cleaning, late trajectory RMSE is 0.06445 for the full model. The full
model beats crop-only with stable grouped uncertainty, but its gain over the
no-future model is small and its interval crosses zero. The policy marks the
future branch research-only.

## Rejected apparent wins

- January point correction: strong development result, late RMSE worse by
  4.73 kg/ha.
- March point correction: small development win, late RMSE worse by
  12.94 kg/ha.
- January trend addition: late pooled AUC improves, but grouped interval crosses
  zero.
- February trend addition: reverses strongly late.
- Future-weather-only increment: positive point estimate, but late grouped
  interval crosses zero.

## Accepted result

March increase/decrease:

- development AUC 0.8064 -> 0.8222;
- late AUC 0.7682 -> 0.7956;
- late Brier 0.2057 -> 0.1970;
- grouped late AUC-gain interval +0.0030 to +0.0540.

