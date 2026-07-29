# V15 complete hierarchy

## Completed outcome

All seven stages below were implemented. Strict testing promoted the
MODIS-to-Sentinel crop correction and the calibrated distribution. The learned
normal and explicit state hierarchy remain shadow components because they did
not transfer safely to 2021–2022.

The final challenger is:

```text
V15 = V14 future-weather shadow
      + 1.25 × (crop-aware XGBoost − matched physical XGBoost)
```

Its strict RMSE is 269.52 kg/ha over 2019–2022 and 287.20 kg/ha over the
untouched 2021–2022 years.

V15 implements the complete seven-part architecture:

1. learn normal district yield from up to 20 earlier years;
2. predict the current season's log-percentage anomaly;
3. pretrain a crop encoder on MODIS seasons beginning in 2000;
4. predict the common state-season shock first;
5. estimate how strongly every district responds to that shock;
6. fine-tune the MODIS crop encoder on recent Sentinel crop transitions; and
7. return a complete probability distribution.

## Point equation

For district \(d\), state \(s\), and season \(t\):

\[
\log y_{d,t}
=
\log b_{d,t}
+\alpha_d
+\beta_d g_{s,t}
+r_{d,t}.
\]

- \(b\): learned normal district yield from earlier history;
- \(g\): predicted shared state-season shock;
- \(\alpha_d,\beta_d\): shrunken district exposure;
- \(r\): district-specific residual predicted from weather, economics, soil,
  MODIS/Sentinel crop condition, and future weather.

## Strict timing

- A yield test year uses yield labels only through the preceding year.
- MODIS pretraining stops at the representation cutoff.
- Sentinel fine-tuning stops at the representation cutoff.
- Yield-training representations are district-group cross-fitted.
- Future weather must come from an issue available before March 5.
- No yield label after 2022 is opened.

## Evaluation

- 2019-2020 selects recipes.
- 2021-2022 confirms the selected recipe.
- All rolling years, individual years, states, and state-years are reported.
- Close candidates are retained as shadows.
- The hierarchy is tested independently and as a bounded correction to V5.
