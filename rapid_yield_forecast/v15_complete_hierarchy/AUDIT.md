# V15 audit

## Validation status

`artifacts/validation.json` reports **pass**.

Validated facts:

- 3,658 long-history rows, 1990–2022, 119 districts;
- 476 final predictions, exactly 119 districts × four years;
- no duplicate district-year output;
- no post-2022 yield label read;
- every held-year encoder has a training cutoff earlier than its target year;
- no yield labels are used in encoder pretraining or fine-tuning;
- no later satellite observation is inserted into an earlier clock;
- V15 formula reproduces every final point exactly;
- q05–q95 are finite and monotonic;
- four-year RMSE reproduces at 269.523911 kg/ha;
- both deployment encoders and both deployment XGBoost bundles load;
- deployment refits are not used for score claims.

## Data reconciliation

ICRISAT and official DES overlap on 1,177 district-years:

- correlation: 0.995206;
- MAE: 16.2515 kg/ha;
- mean ICRISAT minus DES difference: 1.6415 kg/ha.

DES is used from 2010 onward. ICRISAT supplies the earlier history. Six
post-split districts use parent proxies before their own records begin. This is
documented in `data/data_manifest.json`.

## Instabilities found

### Long-history regression drift

The best learned normal on 2019–2020 became substantially worse in 2021–2022.
It is saved but not promoted.

### State hierarchy transfer failure

The state-shock/district-exposure model improved selection years but degraded
the untouched years when blended with V5. Its coefficient estimates and
predictions are saved as shadow evidence.

### Small number of independent seasons

There are 476 district-year rows but only four independent weather seasons and
twelve state-seasons. District rows must not be treated as 476 independent
proofs.

For V15 versus V5 over all four years:

- state-year grouped bootstrap mean gain: 3.78 kg/ha;
- 95% range: 0.18 to 8.06;
- probability of positive gain: 98.0%;
- year grouped 95% range: −0.57 to 6.67.

The state-year result is encouraging. The year result explains why V15 is still
called a challenger.

### 2022

V15 is 2.52 kg/ha worse than V5 in 2022. This result is retained. A conservative
0.125 crop-correction shadow improves all four years, but it is not promoted
because choosing it after seeing all four years would be dishonest.

### Probability width

The pure pinball optimum used a 0.80 scale and covered only 72.7% on development
years for its labelled 80% range. It is explicitly rejected. The released 0.95
scale covers 78.6% on development and 78.2% on untouched years.

## Seven-stage completion

All seven requested stages were implemented. “Completed” does not mean every
stage was promoted. Stages 1, 2, 4 and 5 are honest negative/unstable results.
Stages 3, 6 and 7 contribute directly to the V15 release.

See `artifacts/seven_stage_evidence.csv` for the exact mapping.

## Remaining gaps

- Only three states are in the current district panel.
- Only four strict test seasons are available.
- The promoted crop correction is March 5 only.
- Sentinel history begins in 2017.
- Future weather helps crop trajectory only slightly.
- Crop masks and crop-specific high-resolution satellite extraction can still
  be improved.
- GAT/spatial models remain less convincing than the simpler correction under
  the current sample size; no GAT component is promoted here.

These are limits on confidence, not hidden implementation tasks.
