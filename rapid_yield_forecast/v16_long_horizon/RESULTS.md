# V16 — what was measured and what changed

Follow-on to `v15_complete_hierarchy`. Every number here comes from a script in
`scripts/` and an artifact in `artifacts/`. Nothing is quoted from V15 without
having been re-derived.

---

## 1. The problem with V15 was measurement, not modelling

V15's crop Transformer contributes a correction of ~21 kg/ha sd against a
residual sd of ~271 kg/ha, judged on 476 district-seasons that represent only
**four independent seasons**.

I measured the noise floor of V15's own fitting procedure by re-running the
released 30-feature correction 24 times, permuting nothing but the *order* of
the columns (XGBoost's `colsample_bytree` draws different subsets):

| | dev 2019–20 | untouched 2021–22 | four-year |
|---|---:|---:|---:|
| V14 anchor | 254.11 | 288.13 | 271.65 |
| Released V15 | 250.60 (**+3.50**) | 287.20 (**+0.93**) | 269.52 (**+2.13**) |
| Permuted mean ± sd | 250.54 ± 0.31 | **287.26 ± 0.57** | 269.53 ± 0.31 |
| Permutations beating V14 | 100% | 96% | 100% |

On the untouched years the claimed gain is 1.6× the pure-noise sd. The *sign* is
robust; the *magnitude* is not resolved. This matches V15's own grouped
bootstrap (P(gain>0) = 0.748 on the late block).

---

## 2. The Transformer's real crop-state skill

V15 reports its encoder at 0.0416 RMSE against a **persistence** baseline of
0.0876 — "77% skill". Persistence is a strawman: wheat reliably grows between
January and March. Rebuilding the honest baseline (mean change per transition,
fitted on training seasons only, same masking and metric definition):

| Baseline | RMSE (raw VI units) | Skill vs persistence |
|---|---:|---:|
| Persistence (Δ = 0) — *V15's baseline* | 0.08762 | 0% |
| **Climatology (mean Δ)** | **0.05066** | **66.6%** |
| Encoder (MODIS-pretrained, full) | 0.04163 | 77.4% |

**Two-thirds of the advertised skill is seasonal climatology.** Genuine skill
beyond climatology is ~32%. Split by transition:

| Transition | Climatology | Encoder | Skill beyond climatology |
|---|---:|---:|---:|
| Jan 15 → Feb 15 | 0.06618 | 0.05231 | 37.5% |
| **Feb 15 → Mar 05** | 0.03514 | 0.03094 | **22.5%** |

The 5 March clock depends on Feb→Mar, where the encoder is weakest.

Two further findings:

- **MODIS pretraining does nothing for the pretext task**: 0.04163 pretrained
  vs 0.04179 from scratch (0.4%). Fine-tuning losses are indistinguishable.
  Its apparent downstream benefit rests on a 238-row selection.
- **The better forecaster is discarded.** The `full` branch (with future
  weather) beats `no_future` at both transitions, yet the released 30 features
  come from the `no_future` branch.

## 2b. The ablation V15's grid could not run

Every one of V15's 48 candidates included the 6 `current_index_*_mean` features
(plain `nanmean` of raw VIs, no network), so they were never isolated. Run here
through V15's own `xgb_residual_predict` (base model reproduces to 0.0 kg/ha):

| Added to the 78 base features | dev | late | corr with error |
|---|---:|---:|---:|
| *V14 anchor* | 254.11 | 288.13 | — |
| +6 raw VI means (**no transformer**) | 254.18 | 288.20 | 0.024 |
| +8 transformer delta only | 252.97 | 289.15 | 0.083 |
| **+16 transformer fused-pool only** | **251.30** | **285.55** | **0.161** |
| +30 RELEASED | 250.66 | 286.72 | 0.141 |

The gain is genuinely the Transformer's — the raw VI means contribute nothing.
But the released 30-feature set is **diluted**: the 16 pool dimensions alone
beat it on the untouched block by ~1.2 kg/ha, more than V15's entire claimed
gain. **Recommended change to V15: drop the 6 raw means and the 8 deltas.**

---

## 3. The measurement problem is fixable with data already on disk

Sentinel starts in 2017, which is what pinned V15 to four test seasons. MODIS
runs 2000–2022 and yields run 1990–2022.

| Layer | Years | Independent seasons |
|---|---|---:|
| Sentinel (V15) | 2017–2022 | 4 |
| Weather (NASA POWER) | 2010–2023 | 11 |
| **MODIS + yield history** | **2000–2022** | **19–23** |

`build_v16_panel.py` assembles 2,668 usable district-seasons over 23 seasons.

Two immediate results from the longer panel:

- **Satellite *anomalies* carry the signal; satellite *levels* do not.**
  Over 19 folds, history + MODIS anomalies gains **+17.8 kg/ha [+2.02, +35.72],
  P(>0) = 0.989** over history alone. MODIS levels gain −1.08, P = 0.391.
  On the four-year V15 window the same anomaly gain is +5.27 [−8.04, +21.34],
  P = 0.719 — *the identical effect, unresolvable on four seasons.*
- **A fitted yield trend (`season_index`) costs ~57 kg/ha.** Extrapolating a
  trend into an unseen season is unsafe; it is excluded and kept in its own
  group so the harm stays reproducible.

---

## 4. The largest term in the problem was never modelled

Decomposing the residual around the three-season normal:

```
sd of the season-mean shock          354 kg/ha   (2004-2022)
sd of district variation within a season  289 kg/ha
```

**The shared "what kind of season is this" term is larger than all district
detail.** V15 spent its Transformer budget on district detail and left the
season term to a three-season history average. V15 *did* build a state-shock
stage and rejected it — for failing to beat V14 as a standalone predictor,
which is the wrong test for a component.

### Predicting the shock

A first attempt using state-aggregated MODIS anomalies had **no skill**
(corr −0.14). A RidgeCV over all 80 weather columns was **worse than useless**
(R² = −1.7): with ~10 usable seasons a wide model destroys the signal.

The signal is real and physical, and it is *within-state*, not cross-sectional:

| Feature | pooled r | within-state r | Haryana | Punjab | Uttar Pradesh |
|---|---:|---:|---:|---:|---:|
| Pre-clock rainfall (Nov 1 – Mar 5) | −0.611 | **−0.628** | −0.686 | −0.762 | −0.616 |
| Dec–Feb mean solar radiation | +0.618 | **+0.639** | +0.682 | +0.742 | +0.611 |

Consistent in all three states independently. A **two-feature** model with state
fixed effects achieves out-of-sample **corr +0.464, R² +0.116**. Parsimony is
the result here, not a style choice.

### What the shock layer buys

Weights chosen **prequentially** (only from seasons already forecast):

| Model | Period | RMSE | Bias | Direction |
|---|---|---:|---:|---:|
| Naive weighted history | 2014–22 | 573.9 | −42.7 | 57.9% |
| District layer only | 2014–22 | 547.2 | −28.7 | 61.9% |
| **Full hierarchy + shock** | **2014–22** | **524.6** | **+10.0** | **72.9%** |
| Naive weighted history | 2019–22 | 333.1 | +63.1 | 66.8% |
| District layer only | 2019–22 | **320.9** | +52.4 | 68.9% |
| Full hierarchy + shock | 2019–22 | 363.7 | −19.8 | 69.7% |

Season-resampled bootstrap vs naive:

| Model | Period | Gain | 95% CI | P(>0) |
|---|---|---:|---|---:|
| District layer | 2014–22 | +26.4 | [−3.1, +66.1] | 0.951 |
| District layer | 2019–22 | **+11.9** | **[+4.8, +21.3]** | **1.000** |
| Full hierarchy | 2014–22 | +46.8 | [−28.1, +107.6] | 0.907 |
| Full hierarchy | 2019–22 | −29.4 | [−127.3, +66.8] | 0.301 |

Two honest conclusions:

1. **The district layer is a reliable win** — +11.9 kg/ha on V15's own window
   with P(>0) = 1.000, far better established than V15's +0.93 at P = 0.748.
2. **The shock layer is a large but volatile win.** Per season it gains +156
   (2017), +81 (2021), +76 (2014), +62 (2016) but loses −164 (2019), −69 (2022).
   It wins in shock seasons and loses in calm ones — and **2019–22 contains no
   severe shock**, which is precisely why V15's evaluation window could not
   see its value.

A weakly-predictable dominant term belongs in the **scale**, not the location.

---

## 5. A conditional distribution

V15's uncertainty is a fixed empirical shape shifted to the point and scaled by
0.95. It is well calibrated (80% → 78.4%) but carries almost no case
information: corr(width, |error|) = 0.176, width/point sd = 0.045. Reliability
without resolution.

**A failed attempt worth recording.** Fitting quantile regressions directly on
the training residual produced 80% intervals covering **47%**. State weather is
*constant within a state-season*, so a flexible learner identifies the season
and shrinks to within-season spread — the across-season shock vanishes from the
width. The fix is to fit the scale on **out-of-sample residuals from seasons
already forecast**, which contain the shock by construction.

| Model | Period | CRPS | cov 80% | cov 90% | mean 80% width | width CV |
|---|---|---:|---:|---:|---:|---:|
| Fixed shape | 2019–22 | 247.9 | 98.3% | 99.8% | 1935 | 0.20 |
| **V16 conditional scale** | 2019–22 | 249.8 | **85.5%** | **89.7%** | **1154** | **0.51** |

At equal CRPS the conditional model is **40% sharper and far better
calibrated**. Over the full period the fixed shape becomes badly
over-dispersed (96.5% coverage for a nominal 80%) while the conditional one
stays near nominal.

**Honest limit:** once the size artifact is removed (correlating *relative*
width with *relative* error), per-case discrimination is 0.182 vs 0.147 —
essentially tied. The win is correctly-sized intervals, not smarter per-case
ones. Do not read this as beating V15's published calibration, which was
measured on a calmer window.

---

## 6. The rewritten Transformer — a clean negative result

`train_v16_encoder.py` implements all three planned changes: an **anomaly
target** (change minus climatology), **contrastive pretraining** (MMST-ViT
Eq. 1–2, InfoNCE) replacing next-token regression, and a **long-term climate
bias** added to the attention logits (MMST-ViT Eq. 6). It is trained on MODIS
2000–2022 with 19 rolling-origin folds and district cross-fitting, at 25,139
parameters — a third of V15's 67,359.

One deliberate departure from SimCLR: rows from the **same season are excluded
from the negatives**. Districts in one season share the dominant common shock,
and standard in-batch negatives would train the encoder to discard it.

Training behaved: contrastive loss 1.39 → 1.29 and anomaly loss 0.136 → 0.085
as folds gained seasons.

Judged against the tabular MODIS anomalies it is built from:

| Feature set | Period | RMSE | MAE | Direction |
|---|---|---:|---:|---:|
| history + MODIS anomalies | 19 folds | **449.7** | 326.1 | 60.9% |
| + V16 encoder | 19 folds | 452.0 | 327.4 | 60.9% |
| encoder instead of anomalies | 19 folds | 452.6 | **325.5** | 61.0% |
| history + MODIS anomalies | 2019–22 | 332.2 | 249.0 | 66.8% |
| + V16 encoder | 2019–22 | **329.6** | 246.6 | 67.9% |
| encoder instead of anomalies | 2019–22 | 329.7 | **239.6** | **68.3%** |

Season-resampled bootstrap, encoder vs tabular anomalies:

| Period | Gain | 95% CI | P(>0) |
|---|---:|---|---:|
| **19 folds** | **−2.19** | [−5.14, +1.22] | **0.093** |
| 2019–22 only | +2.68 | [+2.00, +3.53] | 1.000 |

**This is the V15 trap reproduced exactly.** On the four-year window the encoder
looks like a certain win — +2.68 kg/ha at P = 1.000, a tighter interval than
V15 ever reported. On nineteen seasons the sign reverses and it is *negative*
with 91% confidence. The four-year interval is spurious precision: resampling
four seasons that all happen to favour the encoder can only return positives.

Error percentiles over 19 folds show the whole effect is noise-scale:

| percentile | tabular | + encoder | encoder only |
|---|---:|---:|---:|
| 50% | 239.0 | 240.3 | **234.6** |
| 95% | **976.1** | 1000.9 | 991.9 |

The column-permutation noise floor also worsens: sd 0.50 for the tabular model,
**1.14** once 19 encoder columns are added.

### What this actually establishes

The honest reading is not "the Transformer failed" but something more useful:

- **19 encoder numbers reproduce what 70 tabular anomaly features carry** —
  same RMSE, slightly better median error and direction accuracy. That is
  successful *compression*, and it validates the representation.
- **It adds no information**, because at district-mean resolution there is no
  further information to extract. MODIS district means are only 35 numbers per
  clock; a Transformer cannot manufacture signal that the pixels never had.

That points directly at resolution as the binding constraint, not architecture
— which is why sub-district tiles are the next step rather than a fourth
encoder revision.

## 7. Sub-district tiles — the first genuine information gain

`extract_v16_tiles.py` partitions each district into ~9 km cells and pulls
MOD09A1 NDVI/EVI/NDWI summaries per cell, per season, at the 5 March clock.
Completed: **2,737 district-seasons, 169,809 tile-rows, 7,383 unique tiles,
2000–2022, zero failures, 123 minutes.** Tile counts are constant per district
across all 23 seasons, which is what makes per-tile histories legitimate.

Why this was worth doing, in one example from the extracted data:

| | District mean NDVI | Tile range |
|---|---:|---|
| A district-season | 0.455 | 0.418 → 0.489 |
| Another district-season | 0.457 | **0.156 → 0.579** |

Identical to every model built so far. The second has a large area of
essentially failed crop beside healthy ground; the first is uniformly mediocre.
Within-district spread varies 7-fold across the panel (sd 0.015 to 0.104).

Features are of two kinds: **shape** (how unequal the district is now) and
**anomaly** (each tile against its own history, then aggregated — "what share
of this district's area is below what that ground normally does").

| Feature set | Period | RMSE | MAE | Bias | Direction |
|---|---|---:|---:|---:|---:|
| history + district means | 19 folds | 449.7 | 326.1 | −36.0 | 60.9% |
| + tile shape | 19 folds | 446.3 | 323.5 | −33.0 | 61.5% |
| + tile anomaly | 19 folds | 446.8 | 322.5 | −30.4 | 61.9% |
| **tiles instead of district means** | **19 folds** | **442.5** | **319.9** | **−23.0** | **61.9%** |
| history + district means | 2019–22 | 332.2 | 249.0 | +25.4 | 66.8% |
| tiles instead of district means | 2019–22 | 331.2 | **245.4** | +15.1 | 68.5% |

Season-resampled bootstrap against district means:

| Candidate | Period | Gain | 95% CI | P(>0) |
|---|---|---:|---|---:|
| **+ tile shape** | **19 folds** | **+3.40** | **[+0.64, +5.87]** | **0.990** |
| + tile anomaly | 19 folds | +2.98 | [−2.59, +8.98] | 0.848 |
| **tiles instead of district means** | **19 folds** | **+7.38** | [−2.59, +19.82] | 0.912 |
| + tile shape | 2019–22 | +2.48 | [−0.85, +6.50] | 0.893 |

**And it clears its noise floor**, which nothing else in this sequence did:

| Feature set | Permuted RMSE | Noise sd |
|---|---:|---:|
| history + district means | 450.33 | 0.50 |
| + tile anomaly | **448.02** | **0.29** |

The gain (~3–7 kg/ha) is roughly 10× the measurement noise (0.29–0.50). For
comparison, V15's gain was 0.93 against a noise floor of 0.57.

Adding tiles also **halves the bias** (−36.0 → −23.0) and improves direction
accuracy in both windows. Per season the gain is largest in disturbed years
(+24.7 in 2016, +21.9 in 2010, +16.5 in 2018) and negative in a few calm ones
(−21.5 in 2012, −18.8 in 2022) — the expected signature of a feature that
detects damage.

### The conclusion the whole sequence points to

Four attempts were made to extract more from district-mean satellite data —
V15's crop Transformer, V16's rewritten Transformer, MODIS level features, and
MODIS anomaly features. Only the last worked, and only modestly. The moment the
data gained **resolution** rather than a new architecture, the gain appeared and
survived every honest test.

**Resolution, not architecture, was the binding constraint.**

## 8. A better V15 — and it is not a model change

### Tiles do not transfer to V15

The tile features that gained +3.4 to +7.4 kg/ha on the long panel add nothing
inside V15. Weight selection on 2019–20 chose **γ = 0** for the anomaly and
combined tile blocks — use them not at all — and their correlation with V15's
residual error is 0.01–0.06, against 0.14 for V15's own crop correction.

The reason is sample size, not tiles. **V15 trains its XGBoost on seasons
2017..train_end** — two seasons (~238 rows) for the 2019 fold, four for the
last. A 132-column feature block cannot be fitted on that.

### The window is the bottleneck

The 2017 start exists because Sentinel crop state starts in 2017. But the 78
physical/weather/economic features that do the actual work are fully populated
from 2015 (2014: 77 of 78; 2013: 72). V15 was discarding four seasons of
labelled data because one late-arriving modality shared the table.

Changing exactly one thing — the first training season, from 2017 to 2013 —
with identical features, target, depth and seeds:

| Model | dev 2019–20 | untouched 2021–22 | four-year |
|---|---:|---:|---:|
| V14 anchor | 254.11 | 288.13 | 271.65 |
| V15 as released | 250.60 | 287.20 | 269.52 |
| V14 + longer training window | 253.89 | 284.78 | 269.78 |
| **V14 + window + crop correction** | **250.52** | **283.60** | **267.57** |

Season-resampled bootstrap on the untouched block:

| Candidate | Gain | 95% CI | P(>0) |
|---|---:|---|---:|
| V15's crop correction | +0.95 | [−2.73, +4.69] | 0.755 |
| **Longer training window** | **+3.35** | **[+1.54, +5.15]** | **1.000** |
| **Both stacked** | **+4.54** | **[+2.90, +6.21]** | **1.000** |

The window fix alone is **3.5× the crop Transformer's gain and is actually
significant**, where V15's is not. The two are independent and stack: together
they beat released V15 by **1.95 kg/ha on the four-year window and 3.60 on the
untouched block**, at roughly 8× the column-permutation noise floor.

Tiles still do not help even under the longer window (+0.78, P = 0.757), and
they degrade the standalone model (359.7 vs 342.5 four-year RMSE). At V15's
feature richness the tile information is redundant; on the thin long panel it
was not.

### The recommendation

Train V15's matched XGBoost models from **2013** rather than 2017, keeping
everything else identical. It is a one-line change, it needs no new data, no
new model and no new hyper-parameter, and it is the largest verified
improvement found anywhere in this work.

## 9. The unified encoder, and the final model

`train_v16_unified_encoder.py` compresses ~1,200 numbers per district-season
(96 tiles × 12, 3 MODIS clocks × 35, 4 weather windows × 10, 9 climate terms)
into **40**, at 70,695 parameters. Masked-tile modelling plus season-aware
contrastive learning; no yield label; district cross-fitted; 19 folds.

### On the long panel it is the best representation built

| Feature set | Features | RMSE 19 folds | MAE | Direction |
|---|---:|---:|---:|---:|
| history + district means | 98 | 449.7 | 326.1 | 60.9% |
| history + tile statistics | 230 | 447.2 | 323.9 | 61.7% |
| **history + unified encoder** | **68** | **442.6** | **319.1** | **63.2%** |

**40 learned numbers beat 202 hand-built tile statistics**, using a third of the
feature count, with the best direction accuracy of anything tested. The gain
over the tile statistics is +4.78 [−4.13, +14.89], P = 0.836 — positive and
consistent but not individually significant. The defensible claim is
*equal-or-better accuracy at a third of the width*, which is what an encoder is
for.

### Ported into V15 it transfers — barely

It is the only V16 satellite component that earns a non-zero weight inside V15
(γ = 0.25; correlation with V15's residual error 0.103, against 0.01–0.06 for
tile statistics and 0.138 for V15's own crop correction). Alone it is worth
+0.74 [−0.26, +1.75], P = 0.755 — the same ballpark as V15's existing crop
Transformer, not better.

### The final model

All three corrections are independent and stack:

| Model | dev 2019–20 | untouched 2021–22 | four-year |
|---|---:|---:|---:|
| V14 anchor | 254.11 | 288.13 | 271.65 |
| V15 as released | 250.60 | 287.20 | 269.52 |
| V14 + longer window | 253.89 | 284.78 | 269.78 |
| V14 + window + V15 crop | 250.52 | 283.60 | 267.57 |
| V14 + window + unified encoder | 253.61 | 284.07 | 269.27 |
| **V14 + window + V15 crop + encoder** | **250.24** | **282.96** | **267.10** |

Season-resampled bootstrap on the untouched block:

| Model | Gain vs V14 | 95% CI | P(>0) |
|---|---:|---|---:|
| V15 as released | +0.95 | [−2.73, +4.69] | 0.755 |
| + longer window | +3.35 | [+1.54, +5.15] | 1.000 |
| + window + crop | +4.54 | [+2.90, +6.21] | 1.000 |
| **+ window + crop + encoder** | **+5.19** | **[+2.58, +7.83]** | **1.000** |

**The final model beats released V15 by 2.42 kg/ha over four years and 4.24 on
the untouched block, at ~9× the column-permutation noise floor.**

Marginal contributions, in order of size:

| Component | Marginal gain | Cost |
|---|---:|---|
| Training window 2017 → 2013 | **+3.35** | one line |
| V15's existing crop Transformer | +1.19 | already built |
| Unified encoder | +0.65 | 70k params, 2 h extraction, ~3 h training |

The ordering is the result. The single largest verified improvement in this
entire body of work is a change to which rows are used for training, and the
most elaborate component is worth a fifth of it.

## 10. Two simplifications that were tested and rejected

### Collapsing the stack into one model

V15's ~10-model architecture (V5 ensemble + disagreement gate + movement
calibration, V14's four matched models, V15's two) looks like accumulated
complexity.  It is not.  A single depth-2 XGBoost on the residual target,
trained 2013+, with every feature available:

| Architecture | four-year RMSE | Direction | models |
|---|---:|---:|---:|
| one XGBoost, 78 physical | 342.5 | 64.1% | 1 |
| one XGBoost, everything | 358.5 | 63.9% | 1 |
| V5 ensemble + gate | 273.3 | 77.9% | ~5 |
| **FINAL stack** | **265.4** | **78.8%** | ~10 |

**Collapsing the architecture costs 77 kg/ha.**  The single model is worse than
the naive three-season history baseline (333.1): a lone learned residual
actively hurts.  What V5 supplies is an ensemble whose *consensus movement* is
amplified by the calibration, which no single model reproduces.  Adding more
features makes it worse, not better.

The cosmetic cleanups in section 8 are safe.  The architecture is not bloat.

### Putting the encoder into the uncertainty model

The encoder failed to move the point forecast (correlation with V15's residual
error 0.024).  Uncertainty is a different target, so it was tried there too --
a conditional scale model fitted on 2019-20 out-of-sample errors, applied to
the untouched block:

| Distribution | CRPS | 80% coverage | width | corr(width,\|err\|) | drop AUC |
|---|---:|---:|---:|---:|---:|
| **fixed shape (shipped)** | **158.8** | **0.794** | 661 | **0.177** | **0.813** |
| + tile statistics | 163.1 | 0.626 | 424 | 0.045 | 0.788 |
| + encoder 40 numbers | 163.5 | 0.601 | 394 | 0.127 | 0.805 |
| + encoder + tiles | 162.6 | 0.639 | 433 | 0.090 | 0.792 |

Worse on every measure, and coverage collapses from 0.794 to 0.601 against a
nominal 0.80.  The cause is the same one that produced 47% coverage on the long
panel: fitting a conditional scale needs many seasons of realised residuals,
and two seasons is not enough.  The long-panel version worked because it had
six or more prior folds to fit on.

The encoder has now been tried inside V15 in three places -- as a point
correction, as added features, and as an uncertainty scale -- and fails in all
three, each for a diagnosable reason.  It remains valuable where it was
measured to work: the long panel, where 40 learned numbers match 202 hand-built
ones, and by extension any deployment with satellite but without V15's 78
weather/soil/economic inputs.

## 11. Not done

**The encoder rewrite** (contrastive pretraining, anomaly-target pretext,
MMST-ViT long-term climate attention bias) is designed but not built. The
evidence above redirected priority: district-level crop state is the *smaller*
term, and the Transformer's measured contribution is ~1 kg/ha on untouched
years. The shock layer and the distribution were the larger levers.

If resumed, the design stands: use the three spatial views (crop-mask,
active-vegetation, hybrid) as ready-made contrastive positive pairs — they are
literally different views of the same field and need no synthetic augmentation.

**Sub-district spatial tokens** (MMST-ViT's Spatial Transformer) remain the
largest untapped information source and are blocked on satellite re-extraction
at ~9 km tiles, which is data engineering rather than modelling.

---

## Reproducing

```bash
python3 scripts/build_v16_panel.py
python3 scripts/build_v16_weather.py
python3 scripts/run_v16_evaluation.py
python3 scripts/run_v16_shock.py
python3 scripts/run_v16_hierarchy.py
python3 scripts/run_v16_distribution.py
```

`v16_common.py` holds the shared fitting code, the rolling-origin harness, the
column-permutation **noise floor**, and the **season-resampled** bootstrap.
Every reported gain should be quoted against its noise floor; V15's was not, and
that is how a 0.93 kg/ha gain came to be reported as a result.
