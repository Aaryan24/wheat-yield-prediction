# V15 exact architecture and mathematics

## Ground-up technical specification of the released research model

**Model:** V15 complete hierarchy / crop-transfer frontier  
**Forecast unit:** one wheat-growing district and one harvest season  
**Primary forecast clock:** information available by 5 March  
**Target:** district wheat yield in kilograms per hectare  
**Strict evaluation years:** 2019, 2020, 2021, and 2022  
**Development years used to choose model variants:** 2019 and 2020  
**Untouched confirmation years:** 2021 and 2022  
**Document date:** 26 July 2026

---

## 1. What V15 really is

V15 is not one giant neural network trained from raw weather directly to yield.
It is a deliberately layered model:

1. V5 produces a strong, conservative district-yield estimate from yield history,
   weather, satellite, and lagged economic information.
2. V14 adds a small correction based on the difference between a model that sees
   future-weather/crop-outlook information and an otherwise matched model that
   does not see it.
3. V15 trains a small crop-state Transformer. It first learns crop development
   from 2000–2022 MODIS satellite history, then learns January-to-February and
   February-to-March Sentinel crop changes.
4. Two identically configured XGBoost models are trained:

   - a base model using the ordinary 78 V14 physical/history/economic inputs;
   - a crop-aware model using those same 78 inputs plus 30 V15 crop-state
     representation inputs.

5. The difference between those two XGBoost predictions is treated as the
   isolated information added by the crop encoder.
6. That difference is added to V14 with a conservative weight of 1.25.
7. An empirical residual distribution is shifted around the V15 point prediction
   to output 19 quantiles and event probabilities.

The exact released point equation is:

$$
\boxed{
\widehat y^{V15}_{d,t}
=
\widehat y^{V14}_{d,t}
+1.25
\left(
\widehat y^{crop}_{d,t}
-
\widehat y^{base}_{d,t}
\right)
}
$$

where:

- $d$ is the district;
- $t$ is the season start year;
- $\widehat y^{V14}$ is the frozen V14 shadow prediction;
- $\widehat y^{crop}$ is the V15 XGBoost with current crop-state features;
- $\widehat y^{base}$ is the matched V15 XGBoost without those crop-state
  features.

This equation is the most important description of V15.

---

## 2. What is in the final model and what is only a shadow experiment

This distinction matters because V15 tested more systems than it finally used.

### 2.1 Systems that actually produce the released V15 prediction

| Part | Used in final point? | Role |
|---|---:|---|
| V5 point model | Yes | Strong historical and multimodal anchor |
| V14 future-outlook correction | Yes | Earlier future-weather/crop-response correction |
| MODIS-pretrained V15 crop Transformer | Yes | Converts crop and weather sequences into crop-state features |
| V15 base XGBoost | Yes | Reference prediction without V15 crop features |
| V15 crop-aware XGBoost | Yes | Prediction with 30 V15 crop-state features |
| V15 correction weight 1.25 | Yes | Controls the size of the crop-state correction |
| V14 empirical quantile shape | Yes | Supplies the uncertainty shape |
| V15 distribution scale 0.95 | Yes | Slightly contracts that uncertainty shape |

### 2.2 Systems evaluated but not used in the final point

| Part | Final status | Reason |
|---|---|---|
| Learned 10–20 year district normal | Shadow only | Did not beat the simpler frozen point anchor |
| State-shock/district-exposure hierarchy | Shadow only | Scientifically useful, but not strong enough to replace V14 |
| Direct crop-Transformer-to-yield prediction | Rejected | Too few independent yield years; it learned unstable shortcuts |
| Full future branch of the V15 encoder | Not in selected 30 features | Future weather remained useful in V14, but this V15 branch did not add a stable extra correction |
| MODIS features directly appended to XGBoost | Rejected for final V15 | Larger feature set was less stable than transfer pretraining |
| Graph/GAT system | Evaluated before V15, not selected | No strict gain large or stable enough for the final model |

Therefore, “V15 complete hierarchy” means the entire hierarchy was built and
audited. It does **not** mean every experimental stage is multiplied into the
released prediction.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $y_{d,t}$ | Observed yield for district $d$ in season $t$, kg/ha |
| $\widehat y_{d,t}$ | Predicted yield |
| $y_{d,t-k}$ | Yield $k$ seasons before the target season |
| $b_{d,t}$ | Three-season weighted history baseline |
| $r_{d,t}$ | Residual yield target, $y_{d,t}-b_{d,t}$ |
| $x_{d,t}$ | Ordinary tabular features |
| $z_{d,t}$ | V15 crop representation features |
| $F(\cdot)$ | XGBoost residual model |
| $d_h$ | Transformer hidden size, fixed at 32 |
| $Q_\alpha$ | Predicted yield quantile at probability $\alpha$ |
| $\mathbb 1(\cdot)$ | Indicator: 1 when the statement is true, otherwise 0 |

All yields and yield errors in this document are in kg/ha unless stated
otherwise.

---

## 4. Exact data inventory

### 4.1 Yield labels and long history

The sealed V15 yield table contains:

- 3,658 district-season rows;
- 119 districts;
- seasons from 1990 through 2022;
- no yield labels after 2022.

Sources:

- ICRISAT district yield history for the older period;
- official DES yield data for the recent period;
- the final combined series uses the older source through 2009 and DES from
  2010 through 2022.

The two sources overlap for 1,177 rows. On that overlap:

$$
\operatorname{corr}(y^{ICRISAT},y^{DES})=0.995206
$$

$$
\operatorname{MAE}(y^{ICRISAT},y^{DES})=16.252
$$

$$
\operatorname{mean}(y^{ICRISAT}-y^{DES})=1.642
$$

This high agreement is why the older source was considered safe for learning
long-run district behaviour.

For six districts created by administrative splits, pre-2010 history is
backfilled from the parent district. A level ratio is estimated on overlapping
years:

$$
\rho_d
=
\operatorname{median}_{t\in overlap}
\left(\frac{y_{d,t}}{y_{parent(d),t}}\right)
$$

and clipped to:

$$
\rho_d\in[0.70,1.30].
$$

The old parent history is then multiplied by $\rho_d$. These reconstructed
values support historical features; they are not treated as newly observed
post-2022 labels.

### 4.2 MODIS satellite sequence

The long satellite pretraining panel contains:

- 2,737 district-seasons;
- 2000–2022;
- three forecast clocks: 15 January, 15 February, and 5 March;
- a tensor of shape:

$$
2737\times3\times35.
$$

The three clocks are time tokens. Each token has 35 summaries.

The raw satellite families are:

- MOD09Q1 NDVI;
- MOD09Q1 near-infrared reflectance;
- MOD09Q1 red reflectance;
- MOD13Q1 EVI;
- MOD13Q1 NDVI.

Each family has seven summaries:

1. last valid mean;
2. November–December mean;
3. most recent 48-day mean;
4. season maximum;
5. season mean;
6. season temporal standard deviation;
7. slope per day.

Thus:

$$
5\text{ satellite families}\times7\text{ summaries}=35\text{ features}.
$$

MODIS is used to teach the encoder how crop vegetation normally develops
through the season. It is **not** directly used as a raw 2000–2022 yield target.

### 4.3 Sentinel crop-state sequence

The recent high-resolution panel contains:

- 2,142 district-clock rows;
- 119 districts;
- seasons 2017–2022;
- clocks 15 January, 15 February, and 5 March.

For every district-clock row, the crop-state array has shape:

$$
6\times21.
$$

The six vegetation indices are:

1. NDVI;
2. EVI;
3. NDRE;
4. NDMI;
5. NIRv;
6. PSRI.

For each index there are 21 values:

$$
3\text{ spatial views}\times7\text{ time summaries}=21.
$$

The three views are:

- crop-mask view;
- active-vegetation view;
- hybrid view.

The seven summaries are:

- early;
- season;
- recent 40 days;
- recent 20 days;
- peak;
- growth;
- change over 20 days.

Therefore one crop token is flattened to:

$$
6\times21=126\text{ numbers}.
$$

At a clock such as 15 February, only the January and February crop tokens are
visible. The March token is masked. This prevents future satellite leakage.

PSRI values with absolute value above 2 were treated as physically invalid and
masked. This affected 96 cells.

### 4.4 Experienced weather sequence

Each district-clock row has a six-token sequence with 16 features per token:

$$
6\times16.
$$

The features are:

1. past maximum-temperature mean;
2. past maximum-temperature maximum;
3. past minimum-temperature mean;
4. past precipitation sum;
5. past solar-radiation mean;
6. past relative-humidity mean;
7. past wind-speed mean;
8. past root-zone soil moisture mean;
9. past top-layer soil moisture mean;
10. past profile soil moisture mean;
11. weather available fraction;
12. solar available fraction;
13. soil available fraction;
14. lag-window midpoint;
15. sine of day of year;
16. cosine of day of year.

The last two values encode the circular calendar:

$$
\operatorname{doy\_sin}
=
\sin\left(\frac{2\pi\,DOY}{365.25}\right)
$$

$$
\operatorname{doy\_cos}
=
\cos\left(\frac{2\pi\,DOY}{365.25}\right).
$$

This lets the network distinguish, for example, a hot period in January from the
same temperature in March.

### 4.5 Future-weather sequence

Each district-clock row has ten five-day future tokens, with 16 features per
token:

$$
10\times16.
$$

The features are:

1. forecast maximum-temperature mean;
2. forecast maximum-temperature maximum;
3. forecast minimum-temperature mean;
4. forecast precipitation sum;
5. forecast solar-radiation mean;
6. forecast wind-speed mean;
7. spatial standard deviation of maximum temperature;
8. spatial standard deviation of minimum temperature;
9. spatial standard deviation of precipitation;
10. spatial standard deviation of solar radiation;
11. spatial standard deviation of wind;
12. number of forecast days above 32 °C;
13. number of forecast days above 34 °C;
14. forecast-lead midpoint;
15. sine of day of year;
16. cosine of day of year.

For every historical reforecast, the latest issue date must satisfy:

$$
\text{issue date}\leq\text{forecast clock}-2\text{ days}.
$$

No realized weather after the forecast clock is used as a forecast input.

### 4.6 Ordinary V15 XGBoost panel

The selected V15 tabular model uses 78 numeric inputs. They cover:

- recent district and state yield history;
- extended 5- and 10-year history summaries;
- observed weather to the forecast date;
- strict future-weather features;
- heat, rainfall, radiation, soil, and water-balance stress;
- state and regional stress summaries;
- wheat MSP level and lagged changes.

State is also one-hot encoded. There are three states, so the design matrix has:

$$
78+3=81
$$

columns before the median imputer adds any missingness indicators.

The crop-aware model adds 30 encoder features:

$$
81+30=111
$$

columns before missingness indicators.

The exact 78-column list is in Appendix A.

---

## 5. Stage A: V5 frozen anchor

V15 retains V5 because it remains the strongest conservative anchor in this
small-year setting.

### 5.1 Three-year weighted history component

The simple history estimate is:

$$
H_{d,t}
=
0.60y_{d,t-1}
+0.25y_{d,t-2}
+0.15y_{d,t-3}.
$$

This component is not the entire V5. It is one member of the V5 ensemble.

### 5.2 V5 component predictions

For each district-season, V5 has five component outputs:

- $H$: recent-history prediction;
- $R$: strict weather/satellite XGBoost prediction;
- $P$: corrected physics XGBoost prediction;
- $E$: lagged economic/full XGBoost prediction;
- $C$: locked CY-Bench transfer prediction.

The CY-Bench component is an Extra Trees model with:

- maximum depth 2;
- minimum leaf size 20;
- weather-all-views feature set;
- shrinkage 0.75.

The physics component has zero direct point weight in the selected V5 average,
but its sign still participates in the conflict gate.

### 5.3 V5 disagreement gate

Let the previous season yield be $L=y_{d,t-1}$. For the four non-history models,
compute their proposed movement signs:

$$
s_R=\operatorname{sign}(R-L),\quad
s_P=\operatorname{sign}(P-L),\quad
s_E=\operatorname{sign}(E-L),\quad
s_C=\operatorname{sign}(C-L).
$$

Their summed vote is:

$$
S=s_R+s_P+s_E+s_C.
$$

The four models are considered to have a majority direction when:

$$
|S|\geq2.
$$

The majority direction is $\operatorname{sign}(S)$. A conflict is declared when
that direction disagrees with the direction proposed by history:

$$
G_{d,t}
=
\mathbb 1\left[
|S|\geq2
\;\land\;
\operatorname{sign}(H-L)\neq\operatorname{sign}(S)
\;\land\;
\operatorname{sign}(H-L)\neq0
\right].
$$

### 5.4 V5 normal and conflict weights

When there is no conflict, the raw V5 estimate is:

$$
V^{raw}
=
0.50H+0.15R+0.00P+0.15E+0.20C.
$$

When the non-history models strongly disagree with history:

$$
V^{raw}
=
0.20H+0.30R+0.00P+0.30E+0.20C.
$$

The implementation can be written in one equation:

$$
V^{raw}_{d,t}
=
(1-G_{d,t})
\left(0.50H+0.15R+0.15E+0.20C\right)
+G_{d,t}
\left(0.20H+0.30R+0.30E+0.20C\right).
$$

### 5.5 V5 movement calibration

The raw ensemble was too conservative in how far it moved away from last
season. V5 therefore uses:

$$
\widehat y^{V5}_{d,t}
=
L
+1.5001443110
\left(V^{raw}_{d,t}-L\right).
$$

The movement scale was fitted only on 2019–2020. If:

$$
x_i=V^{raw}_i-L_i,\qquad z_i=y_i-L_i,
$$

the least-squares scale is:

$$
\widehat s=\frac{\sum_i x_i z_i}{\sum_i x_i^2},
$$

then constrained to the allowed search range. The locked value is
$1.5001443110$.

---

## 6. Stage B: V14 future-outlook correction inherited by V15

V14 asks a useful causal-style question:

> What changes in the prediction when future crop/weather outlook features are
> added to otherwise matched models?

### 6.1 V14 residual target

All matched V14 XGBoost models predict the residual around the simple
three-season history baseline:

$$
b_{d,t}
=
0.60y_{d,t-1}
+0.25y_{d,t-2}
+0.15y_{d,t-3},
$$

$$
r_{d,t}=y_{d,t}-b_{d,t}.
$$

For feature vector $x$, the point prediction is:

$$
\widehat y=b+\frac{F_{42}(x)+F_{73}(x)}{2}.
$$

The two models use random seeds 42 and 73.

### 6.2 V14 matched model variants

At XGBoost depth 2, V14 obtains:

- $\widehat y^{NF}$: ordinary features plus no-future outlook features;
- $\widehat y^{FULL}$: ordinary features plus full future-aware features;
- $\widehat y^{EFFECT}$: ordinary features plus explicit future-effect features;
- $\widehat y^{BROAD}$: ordinary features plus the broad future feature set.

The isolated V14 future increment is:

$$
c^{V14}
=
\frac{
\widehat y^{FULL}
+\widehat y^{EFFECT}
+\widehat y^{BROAD}
}{3}
-
\widehat y^{NF}.
$$

The released V14 shadow point is:

$$
\widehat y^{V14}
=
\operatorname{clip}
\left(
\widehat y^{V5}+1.75c^{V14},
500,
7000
\right).
$$

This is the prediction that becomes the anchor for V15.

### 6.3 V14 XGBoost settings

The V14 matched XGBoost models use:

- 350 boosting trees;
- maximum tree depth 2;
- learning rate 0.025;
- minimum child weight 25;
- row subsampling 0.85;
- column subsampling 0.65;
- L2 penalty 50;
- L1 penalty 5;
- squared-error objective;
- histogram tree builder;
- median imputation with missing-value indicators;
- state one-hot encoding;
- seeds 42 and 73, averaged.

---

## 7. Stage C: V15 crop-transfer Transformer

The V15 neural model does not predict yield. It predicts how the observed crop
state is likely to change from one forecast clock to the next. Its hidden
representation is later supplied to XGBoost.

This separation is intentional:

- there are thousands of crop-transition examples;
- there are only a few genuinely independent recent yield seasons;
- learning crop development first is statistically safer than asking a neural
  network to memorize district yield.

### 7.1 Input tensors

For one district-season at one forecast clock:

| Tensor | Shape | Meaning |
|---|---:|---|
| MODIS crop sequence | $3\times35$ | Long-history vegetation sequence used in pretraining |
| Sentinel crop sequence | $3\times126$ | January, February, March crop-state tokens; future clocks masked |
| Experienced weather | $6\times16$ | Weather and soil observed before the clock |
| Future weather | $10\times16$ | Ten dated five-day forecast tokens |
| Crop token mask | $3$ | Which crop clocks are legally visible |
| Experienced-weather mask | $6$ | Which past-weather tokens exist |
| Future-weather mask | $10$ | Which forecast tokens exist |

### 7.2 Normalization

For an ordinary feature $j$, normalization is:

$$
\widetilde x_j=\frac{x_j-\mu_j}{\sigma_j}.
$$

The mean and standard deviation are fitted only on the training portion of the
fold.

Separate scales are learned for:

- MODIS features;
- flattened Sentinel features;
- experienced-weather variables;
- future-weather variables.

Missing normalized values are replaced with zero, but the corresponding token
mask remains false. Zero therefore means “neutral after normalization,” while
the mask tells attention not to treat a missing token as observed.

The crop-change target uses a robust scale:

$$
m_j=\operatorname{median}(\Delta x_j),
$$

$$
s_j=1.4826
\operatorname{median}\left(|\Delta x_j-m_j|\right).
$$

If this scale is unusable, the standard deviation is used. The normalized
change is:

$$
\widetilde{\Delta x}_j
=
\frac{\Delta x_j-m_j}{s_j}.
$$

### 7.3 Input adapters

All modalities are mapped to a common 32-dimensional space.

MODIS adapter:

$$
e^{MODIS}_k=W_Mx^{MODIS}_k+b_M,\qquad W_M\in\mathbb R^{32\times35}.
$$

Sentinel adapter:

$$
e^{S2}_k=W_Sx^{S2}_k+b_S,\qquad W_S\in\mathbb R^{32\times126}.
$$

Experienced-weather adapter:

$$
e^{past}_k=W_Px^{past}_k+b_P,\qquad W_P\in\mathbb R^{32\times16}.
$$

Future-weather adapter:

$$
e^{future}_k=W_Fx^{future}_k+b_F,\qquad W_F\in\mathbb R^{32\times16}.
$$

Learned position embeddings are added:

- crop position tensor: $1\times3\times32$;
- past-weather position tensor: $1\times6\times32$;
- future-weather position tensor: $1\times10\times32$.

These position vectors tell the model which token is January, February, March,
or which five-day forecast window it represents.

### 7.4 Transformer blocks

The crop branch uses two Transformer encoder layers.

The experienced-weather branch uses one Transformer encoder layer.

The future-weather branch uses one Transformer encoder layer.

Every layer has:

- hidden width 32;
- 4 attention heads;
- head width $32/4=8$;
- feed-forward width 64;
- GELU activation;
- dropout 0.08;
- pre-layer normalization.

For one attention head:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{8}}+M
\right)V,
$$

where $M$ is the mask that makes unavailable tokens impossible to attend to.

Four heads are concatenated:

$$
\operatorname{MHA}(X)
=
\operatorname{Concat}(head_1,\ldots,head_4)W_O.
$$

The feed-forward block is:

$$
\operatorname{FFN}(h)
=
W_2\operatorname{GELU}(W_1h+b_1)+b_2,
$$

where $W_1$ expands 32 to 64 dimensions and $W_2$ contracts 64 back to 32.

### 7.5 Current crop query

After the crop sequence passes through the two crop Transformer layers, V15
selects the token corresponding to the current legal forecast clock:

$$
q=h^{crop}_{current}.
$$

For example:

- at 15 January, $q$ is the January token;
- at 15 February, $q$ is the February token after attending to January and
  February;
- at 5 March, $q$ is the March token after attending to all three legal crop
  observations.

### 7.6 Cross-attention to experienced and future weather

The current crop token asks two separate questions:

1. Which experienced-weather periods explain the current crop state?
2. Which forecast weather periods are relevant to its next change?

Experienced-weather context:

$$
c^{past}
=
\operatorname{MHA}
\left(
Q=q,\,
K=H^{past},\,
V=H^{past}
\right).
$$

Future-weather context:

$$
c^{future}
=
\operatorname{MHA}
\left(
Q=q,\,
K=H^{future},\,
V=H^{future}
\right).
$$

These are separate 4-head cross-attention layers with dropout 0.06.

The fused crop state is:

$$
h^{fused}
=
\operatorname{LayerNorm}
\left(
q+c^{past}+c^{future}
\right).
$$

In the no-future branch:

$$
c^{future}=0.
$$

That branch is important because it gives an explicit counterfactual:

> What representation would the model form from crop condition and experienced
> weather alone?

### 7.7 Masked sequence pooling

For a sequence $H=(h_1,\ldots,h_K)$ and availability mask $m_k$:

$$
\operatorname{pool}(H)
=
\frac{\sum_{k=1}^{K}m_kh_k}
{\max(1,\sum_{k=1}^{K}m_k)}.
$$

V15 produces:

- crop pool;
- experienced-weather pool;
- future-weather pool;
- fused current-crop vector;
- experienced-weather cross-attention context;
- future-weather cross-attention context.

Each is 32-dimensional.

### 7.8 Prediction heads

#### MODIS next-token head

$$
\widehat x^{MODIS}_{next}
=
W_{M2}
\operatorname{GELU}
\left(W_{M1}h+b_{M1}\right)
+b_{M2},
$$

with:

- $32\rightarrow48$;
- GELU;
- $48\rightarrow35$.

#### Sentinel crop-change head

$$
\widehat{\Delta x}
=
W_{S2}
\operatorname{GELU}
\left(W_{S1}h^{fused}+b_{S1}\right)
+b_{S2},
$$

with:

- $32\rightarrow64$;
- GELU;
- $64\rightarrow126$.

#### Sentinel change-sign head

$$
\widehat \ell^{sign}=W_{sign}h^{fused}+b_{sign},
$$

with:

- $32\rightarrow126$ logits.

The full encoder has **67,359 trainable parameters**.

Exact parameter count by module:

| Module | Trainable parameters |
|---|---:|
| MODIS adapter, $35\rightarrow32$ | 1,152 |
| Sentinel adapter, $126\rightarrow32$ | 4,064 |
| Two-layer crop Transformer | 17,088 |
| Experienced-weather adapter, $16\rightarrow32$ | 544 |
| Future-weather adapter, $16\rightarrow32$ | 544 |
| One-layer experienced-weather Transformer | 8,544 |
| One-layer future-weather Transformer | 8,544 |
| Crop-to-experienced-weather cross-attention | 4,224 |
| Crop-to-future-weather cross-attention | 4,224 |
| Final LayerNorm | 64 |
| MODIS head | 3,299 |
| Sentinel crop-change head | 10,302 |
| Sentinel sign head | 4,158 |
| Three learned position-embedding tensors | 608 |
| **Total** | **67,359** |

---

## 8. How the crop encoder is trained

### 8.1 MODIS pretraining task

The model learns two transitions:

- January $\rightarrow$ February;
- February $\rightarrow$ 5 March.

For each source position, later crop tokens are hidden. The model predicts the
next 35-dimensional MODIS token.

For error $e=\widehat x-x$, Smooth L1 with threshold $\beta=0.5$ is:

$$
L_{\beta}(e)
=
\begin{cases}
\frac{e^2}{2\beta}, & |e|<\beta,\\
|e|-\frac{\beta}{2}, & |e|\geq\beta.
\end{cases}
$$

Only finite target cells contribute to the loss.

Pretraining settings:

- AdamW optimizer;
- learning rate $8\times10^{-4}$;
- weight decay $7\times10^{-4}$;
- batch size 256;
- 24 epochs;
- gradient norm clipped at 2.

This task uses no yield label. It teaches seasonal crop dynamics.

### 8.2 Sentinel fine-tuning task

For each recent district-season:

$$
\Delta x=x_{next\ clock}-x_{current\ clock}.
$$

Transitions:

- 15 January $\rightarrow$ 15 February;
- 15 February $\rightarrow$ 5 March.

The main target is the robust-normalized 126-dimensional change. There is also
a binary sign target:

$$
s_j=\mathbb 1(\Delta x_j>0).
$$

The total loss is:

$$
L
=
\operatorname{SmoothL1}
\left(
\widehat{\widetilde{\Delta x}},
\widetilde{\Delta x};
\beta=0.5
\right)
+0.08
\operatorname{BCEWithLogits}
\left(
\widehat \ell^{sign},
s
\right).
$$

Fine-tuning settings:

- AdamW optimizer;
- learning rate $6\times10^{-4}$;
- weight decay $9\times10^{-4}$;
- batch size 128;
- 45 epochs;
- gradient norm clipped at 2;
- future branch randomly removed for 30% of training examples;
- seeds 42 and 73, averaged.

Future-branch removal teaches the same encoder to operate both with and without
future forecasts. It also makes the future contribution measurable.

### 8.3 Two encoder variants tested

V15 trained:

- `scratch`: Sentinel transition training from random initialization;
- `modis_pretrained`: MODIS pretraining followed by Sentinel transition
  fine-tuning.

The released correction uses `modis_pretrained`.

---

## 9. Exact encoder representation

The encoder emits 230 candidate features per district-season. They are used for
model comparison, but only 30 enter the selected V15 correction.

### 9.1 Full 230-feature representation

For both the no-future and full branches, V15 saves the first 16 dimensions of
six 32-dimensional vectors:

$$
6\times16=96.
$$

The six vectors are:

- crop pool;
- state/past-weather pool;
- future-weather pool;
- fused pool;
- past-weather context;
- future-weather context.

It also reduces the predicted $6\times21$ crop change to:

- six vegetation-index mean changes;
- overall mean absolute change;
- overall fraction of positive changes.

That is:

$$
6+2=8.
$$

Therefore each branch contributes:

$$
96+8=104.
$$

Both branches contribute:

$$
2\times104=208.
$$

V15 then adds:

- 16 dimensions of
  $h^{fused}_{full}-h^{fused}_{no\ future}$;
- six current raw vegetation-index means.

Total:

$$
208+16+6=230.
$$

### 9.2 The selected 30 V15 crop features

The winning feature set is called `current`. It contains:

1. the first 16 dimensions of the **no-future fused crop state**;
2. six predicted no-future vegetation-index mean changes;
3. the predicted no-future overall mean absolute crop change;
4. the predicted no-future fraction of positive crop changes;
5. six current raw vegetation-index means.

The exact count is:

$$
16+6+1+1+6=30.
$$

This is a subtle but important result:

- V15’s selected extra signal is mainly the crop’s current condition and the
  experienced weather that led to it;
- V15 did not find a stable additional gain from its own future branch;
- future-weather information is still present through V14 and through the
  ordinary physical forecast features.

---

## 10. Leakage-safe representation generation

If the encoder created representations for its own training districts, its
features could look better simply because it remembered them. V15 prevents this
with district cross-fitting.

District group:

$$
g(d)
=
\left(
\sum_{c\in district\_id}\operatorname{Unicode}(c)
\right)\bmod3.
$$

For training representations:

- districts are divided into groups 0, 1, and 2;
- the encoder generating a group’s representations is trained without that
  group.

For a held-out forecast year:

- the encoder is trained through the allowed cutoff;
- it is then applied to every district in the forecast year.

Strict folds:

| Test year | Latest allowed training season |
|---:|---:|
| 2019 | 2018 |
| 2020 | 2019 |
| 2021 | 2020 |
| 2022 | 2020 |

The 2022 model deliberately remains trained through 2020 because 2021 and 2022
form the untouched late confirmation block.

MODIS pretraining, Sentinel fine-tuning, normalization, and district
cross-fitting all obey the same fold cutoff.

---

## 11. Stage D: matched V15 XGBoost models

### 11.1 Residual target

The XGBoost models do not directly relearn full district yield. They learn what
is missing from the recent-history baseline:

$$
b_{d,t}
=
0.60y_{d,t-1}
+0.25y_{d,t-2}
+0.15y_{d,t-3},
$$

$$
r_{d,t}=y_{d,t}-b_{d,t}.
$$

For seed $s$:

$$
\widehat r^{(s)}=F_s(x).
$$

The two-seed point estimate is:

$$
\widehat y
=
b+\frac{\widehat r^{(42)}+\widehat r^{(73)}}{2}.
$$

The individual XGBoost outputs are clipped to the physically broad range
$[500,7000]$ in the strict experiment.

### 11.2 Base model

The base model uses:

$$
x^{base}\in\mathbb R^{78}
$$

plus the three state one-hot columns.

Its prediction is:

$$
\widehat y^{base}
=
b
+\frac{
F^{base}_{42}(x^{base})
+F^{base}_{73}(x^{base})
}{2}.
$$

### 11.3 Crop-aware model

The crop-aware model uses the exact same base inputs plus the 30 selected
encoder features:

$$
x^{crop}=[x^{base};z^{current}].
$$

Its prediction is:

$$
\widehat y^{crop}
=
b
+\frac{
F^{crop}_{42}(x^{crop})
+F^{crop}_{73}(x^{crop})
}{2}.
$$

### 11.4 Why the difference is used

The isolated crop contribution is:

$$
c^{V15}
=
\widehat y^{crop}-\widehat y^{base}.
$$

Because the two models:

- use the same training rows;
- have the same target;
- use the same base features;
- use the same tree settings;
- use the same seeds;

their difference is a controlled estimate of what the V15 crop representation
adds.

### 11.5 Exact V15 XGBoost settings

Both models use:

- 350 trees;
- maximum depth 2;
- learning rate 0.025;
- minimum child weight 25;
- row subsample 0.85;
- column subsample 0.65;
- L2 penalty 50;
- L1 penalty 5;
- objective `reg:squarederror`;
- histogram tree builder;
- median imputation;
- missingness indicators;
- state one-hot encoding;
- random seeds 42 and 73;
- average of the two seed predictions.

The chosen model identifier is:

`modis_pretrained__physical__d2__current_minus_base`

Its name literally means:

- MODIS-pretrained encoder;
- 78-column physical/history/economic panel;
- depth-2 XGBoost;
- current-crop feature set;
- subtract matched base model.

---

## 12. Stage E: final V15 point equation and weight selection

### 12.1 Candidate correction grid

V15 tested:

- scratch and MODIS-pretrained encoders;
- physical and physical-plus-raw-MODIS base groups;
- tree depths 1 and 2;
- base, current, full, effect, transition, and pool representation subsets;
- multiple difference definitions;
- both V5 and V14 anchors;
- correction weights:

$$
\gamma\in\{-3.00,-2.75,\ldots,2.75,3.00\}.
$$

This produced 48 independent XGBoost candidates and 64 isolated correction
candidates.

### 12.2 Development selection score

For every candidate on 2019–2020:

$$
S
=
0.50\,RMSE_{all\ districts}
+0.25\,RMSE_{equal\ state}
+0.25\,RMSE_{mean\ year}.
$$

This avoids choosing a model that looks good only because one large state or one
easy year dominates the row count.

The mathematical optimum on development was $\gamma=1.75$. V15 uses a
regularized near-tie rule:

$$
\tau=\max(0.35,\;0.0015S_{best}).
$$

All candidates with:

$$
S\leq S_{best}+\tau
$$

are treated as practically tied, and the smallest absolute correction weight is
preferred. That selected:

$$
\gamma=1.25.
$$

### 12.3 Released point formula

The exact evaluated formula is:

$$
\boxed{
\widehat y^{V15}
=
\widehat y^{V14}
+1.25
\left(
\widehat y^{crop}
-
\widehat y^{base}
\right)
}.
$$

The component XGBoost predictions and V14 are already constrained to broad
physical bounds. The final strict scoring artifact applies the formula above
without an additional last-step clip. The deployment layer may safely enforce
$[500,7000]$ after this equation.

---

## 13. Stage F: probability distribution

V15 outputs more than one number. It outputs:

- point prediction;
- quantiles q05, q10, q15, ..., q95;
- probability of yield increasing versus last season;
- probability of a severe fall of more than 10%.

### 13.1 Source uncertainty shape

V14 built an empirical uncertainty shape from only earlier residuals.

For calibration row $i$:

$$
e_i=y_i-\widehat y_i.
$$

Years are weighted equally. If year $t$ has $n_t$ rows:

$$
w_i^{year}=\frac{1}{n_t}.
$$

Residuals are normalized by:

$$
a_i
=
\max
\left(
\operatorname{recentSD}_i,\,
0.07\,normal_i,\,
150
\right).
$$

Scaled residual:

$$
u_i=\frac{e_i}{a_i}.
$$

State residuals receive an additional mixture weight:

$$
\lambda_{state}
=
\frac{n_{state}}{n_{state}+50}.
$$

The remaining weight stays on the global residual pool. Weighted residual
quantiles are then rescaled to the target district. For sorted residuals
$u_{(1)}\leq\cdots\leq u_{(N)}$, the interpolation coordinate used by the code
is:

$$
c_j
=
\frac{
\sum_{k=1}^{j}w_{(k)}-\frac12w_{(j)}
}{
\sum_{k=1}^{N}w_{(k)}
}.
$$

The weighted quantile is obtained by linearly interpolating $u_{(j)}$ against
$c_j$.

For forecast year $t$, only calibration seasons 2016 through $t-1$ are allowed.

The resulting q05–q95 shape was originally centered on the frozen V5 production
point. The exact inherited method is:

`history_shape__scaled_state_equal_year__w1.00`

### 13.2 Shift to V15

Let:

- $Q^{old}_\alpha$ be the inherited V14 quantile;
- $p^{V5}$ be the old distribution centre;
- $p^{V15}$ be the new centre.

The released V15 quantile is:

$$
Q^{V15}_\alpha
=
p^{V15}
+0.95
\left(
Q^{old}_\alpha-p^{V5}
\right),
$$

for:

$$
\alpha\in
\{0.05,0.10,0.15,\ldots,0.95\}.
$$

The 0.95 scale slightly narrows the inherited distribution. Quantiles are made
monotonic with a cumulative maximum and clipped to $[500,7000]$.

### 13.3 Why scale 0.95 was selected

Candidate centres:

- V5;
- V14;
- V15.

Candidate scales:

$$
0.75,0.80,\ldots,1.50.
$$

The development score is:

$$
S_{dist}
=
\operatorname{meanPinball}
+0.05|\text{scale}-1|.
$$

The released range also had to satisfy:

$$
0.78\leq coverage_{80}\leq0.82,
$$

$$
coverage_{90}\geq0.88.
$$

These coverage constraints stopped an overly narrow but superficially sharp
distribution from winning.

### 13.4 Event probabilities

The 19 quantiles define an approximate cumulative distribution function $F(y)$.
Linear interpolation is used between quantiles. Two artificial endpoints are
added:

$$
(Q_{0.05}-500,\;0)
\quad\text{and}\quad
(Q_{0.95}+500,\;1).
$$

Probability of an increase:

$$
P(y_t>y_{t-1})
=
1-F(y_{t-1}).
$$

Probability of a severe drop:

$$
P(y_t<0.90y_{t-1})
=
F(0.90y_{t-1}).
$$

---

## 14. The seven-stage hierarchy that was built and audited

This subsystem matters scientifically but is not part of the released V15
point. It is documented here so that “complete hierarchy” is not misunderstood.

### 14.1 Learned district normal

Candidate district-normal features include:

- yield lags 1–20;
- rolling means and standard deviations over 3, 5, 10, and 20 years;
- rolling slopes;
- state yield lags 1–5;
- calendar year.

Candidate models included:

- weighted three-year history;
- ten-year mean;
- 20-year trend;
- 20-year exponentially weighted mean;
- Ridge regression with penalties 10, 100, and 1000;
- Extra Trees;
- two XGBoost variants.

Model choice used rolling prior-year selection, never the target or future year.
These learned normals did not reliably beat the frozen V5/V14 system.

### 14.2 State shock

For a candidate district normal $n_{d,t}$, log anomaly is:

$$
a_{d,t}
=
\log
\left(
\frac{y_{d,t}}{n_{d,t}}
\right).
$$

The shared state shock is:

$$
g_{s,t}
=
\frac{1}{N_{s,t}}
\sum_{d\in s}a_{d,t}.
$$

### 14.3 District exposure

Raw district sensitivity to state shock is:

$$
\beta_d^{raw}
=
\frac{
\operatorname{Cov}(a_{d,t},g_{s(d),t})
}{
\operatorname{Var}(g_{s(d),t})
}.
$$

It is shrunk toward 1:

$$
\beta_d
=
\frac{
n_d\beta_d^{raw}+8
}{
n_d+8
},
$$

then clipped:

$$
\beta_d\in[0.25,1.75].
$$

The district intercept is shrunk toward zero:

$$
\alpha_d
=
\frac{n_d}{n_d+12}\alpha_d^{raw},
$$

then clipped to:

$$
\alpha_d\in[-0.15,0.15].
$$

Residual local anomaly:

$$
\epsilon_{d,t}
=
a_{d,t}-\alpha_d-\beta_dg_{s,t}.
$$

Reconstructed yield:

$$
\widehat y_{d,t}
=
n_{d,t}
\exp
\left(
\operatorname{clip}
\left(
\widehat\alpha_d
+\widehat\beta_d\widehat g_{s,t}
+\widehat\epsilon_{d,t},
-0.60,
0.60
\right)
\right).
$$

Final hierarchy outputs are clipped to $[500,7000]$.

This design remains useful for interpretation—“normal yield, shared state
shock, district sensitivity”—but its strict predictions were not selected over
the V14 anchor.

---

## 15. Strict evaluation design

### 15.1 Why this is called strict

For test year $t$:

- no yield label from $t$ or later is used to fit the model;
- no weather occurring after the forecast clock is used as an observed input;
- forecast issue date is at least two days before the clock;
- every historical feature is lagged;
- the training encoder representation is district-cross-fitted;
- all normalizers are fitted inside the allowed fold;
- 2019–2020 choose architectures and weights;
- 2021–2022 confirm the locked choices;
- no post-2022 yield label is read.

### 15.2 Metrics

RMSE:

$$
RMSE
=
\sqrt{
\frac{1}{N}\sum_i(\widehat y_i-y_i)^2
}.
$$

MAE:

$$
MAE
=
\frac{1}{N}\sum_i|\widehat y_i-y_i|.
$$

Bias:

$$
Bias
=
\frac{1}{N}\sum_i(\widehat y_i-y_i).
$$

Direction accuracy:

$$
\frac{1}{N}
\sum_i
\mathbb 1
\left[
(\widehat y_i>y_{i,t-1})
=
(y_i>y_{i,t-1})
\right].
$$

Thus the exact implementation asks whether the model correctly classified an
increase versus “not an increase.”

Equal-state RMSE first computes error within each state and then gives every
state equal importance.

Mean-year RMSE computes RMSE separately for each test year and averages those
year scores.

### 15.3 Point results

| Model | 2019–2022 RMSE | 2021–2022 RMSE | 2019–2022 MAE | 2019–2022 direction accuracy |
|---|---:|---:|---:|---:|
| V5 frozen anchor | 273.262 | 288.610 | 198.212 | 77.94% |
| V14 shadow anchor | 271.652 | 288.129 | 196.135 | 78.78% |
| **V15 released frontier** | **269.524** | **287.200** | **195.677** | **78.78%** |

Gain:

$$
273.262-269.524=3.738\text{ kg/ha versus V5 over all four years}.
$$

$$
288.610-287.200=1.410\text{ kg/ha versus V5 on untouched 2021–2022}.
$$

The gain is real in the strict artifacts, but small. V15 should be described as
the current research frontier, not as a universally proven production victory.

The best crop-encoder XGBoost used **independently**, without V5/V14 anchoring,
had development RMSE 376.544. This is why the encoder is used as an isolated
correction rather than as a standalone yield model.

### 15.4 Distribution results

| Period | 80% coverage | 90% coverage | Rise AUC | Severe-drop AUC |
|---|---:|---:|---:|---:|
| 2019–2020 development | 78.57% | 91.60% | 0.851 | 0.824 |
| 2021–2022 confirmation | 78.15% | 86.97% | 0.837 | 0.799 |
| 2019–2022 combined | 78.36% | 89.29% | 0.845 | 0.809 |

The intended 80% interval covers 78.36% over all four years, which is close to
target. The intended 90% interval covers 89.29%.

---

## 16. Training-time model versus deployment refit

These are different objects and should never be confused.

### Strict scoring models

- trained using only the fold-allowed past;
- produce the reported 2019–2022 scores;
- representations for training districts are cross-fitted;
- 2021 and 2022 are never used to choose the architecture or weights.

### Deployment refit

- uses the locked architecture and weights;
- refits the encoder and matched XGBoost models through 2022;
- is intended to create later-season forecasts;
- makes no scored claim for years after 2022 because their labels were not read.

The deployment model is a refit of a previously selected recipe, not a new
model chosen after seeing later labels.

---

## 17. What must be supplied to produce a new prediction

For district $d$, current season $t$, and forecast clock $c$:

1. district and state identifiers;
2. at least the previous three yields for V5 and the residual baseline;
3. preferably 5–10 years of earlier yield history for extended features;
4. observed weather up to clock $c$;
5. soil-moisture observations up to clock $c$ when available;
6. a valid future-weather forecast issued no later than $c-2$ days;
7. current Sentinel crop-index summaries available by $c$;
8. the wheat MSP known by clock $c$ and only lagged MSP changes;
9. saved preprocessing scales;
10. saved V5, V14, V15 encoder, base-XGBoost, and crop-XGBoost weights.

The output record contains:

- V5 point;
- V14 shadow point;
- V15 point;
- raw V15 crop correction;
- q05 through q95;
- probability of yield increasing;
- probability of a fall greater than 10%.

---

## 18. End-to-end algorithm

For a new district-season:

1. Compute all lagged yield-history features.
2. Compute V5 component predictions $H,R,P,E,C$.
3. Apply the V5 conflict gate and movement calibration to obtain
   $\widehat y^{V5}$.
4. Generate V14 no-future, full, effect, and broad outlook predictions.
5. Compute $c^{V14}$ and then $\widehat y^{V14}$.
6. Build the January/February/March Sentinel token sequence, masking clocks not
   yet observed.
7. Build six past-weather and ten future-weather tokens.
8. Normalize each modality with stored training scales.
9. Run the MODIS-pretrained crop encoder in no-future mode.
10. Extract the selected 30 crop-state features.
11. Build the ordinary 78-column tabular vector.
12. Run the base depth-2 two-seed XGBoost.
13. Run the crop-aware depth-2 two-seed XGBoost.
14. Compute:

    $$
    c^{V15}=\widehat y^{crop}-\widehat y^{base}.
    $$

15. Compute:

    $$
    \widehat y^{V15}=\widehat y^{V14}+1.25c^{V15}.
    $$

16. Shift the inherited uncertainty shape:

    $$
    Q^{V15}_\alpha
    =
    \widehat y^{V15}
    +0.95(Q^{old}_\alpha-\widehat y^{V5}).
    $$

17. Interpolate the quantile CDF to obtain rise and severe-drop probabilities.

---

## Appendix A. Exact 78 ordinary V15 XGBoost inputs

### A.1 District and state yield-history inputs

1. `lag_1_yield`
2. `lag_2_yield`
3. `lag_3_yield`
4. `lag_4_yield`
5. `lag_5_yield`
6. `yield_recent_mean`
7. `yield_recent_std`
8. `yield_recent_slope`
9. `state_lag_1_mean_yield`
10. `lag_1_minus_state`
11. `ext_recent_5_mean`
12. `ext_recent_5_std`
13. `ext_recent_5_slope`
14. `ext_recent_10_mean`
15. `ext_recent_10_std`
16. `ext_recent_10_slope`
17. `ext_lag1_minus_state`
18. `ext_prior_10_proxy_share`

### A.2 Observed weather and soil inputs

19. `obs_early_tmax_mean_anomaly`
20. `obs_early_precip_sum_anomaly`
21. `obs_early_rain_days`
22. `obs_mid_tmax_max`
23. `obs_mid_solar_mean`
24. `obs_mid_precip_sum_anomaly`
25. `obs_mid_dry_spell_max`
26. `obs_late_tmax_mean_anomaly`
27. `obs_late_precip_sum_anomaly`
28. `obs_late_heavy_rain_days`
29. `obs_late_vpd_mean_anomaly`
30. `obs_all_gdd_sum_anomaly`
31. `obs_all_heavy_rain_days`
32. `v2_soil_mid_root_mean_past_z`
33. `v2_soil_late_root_last_past_z`
34. `v2_soil_all_top_change`

### A.3 Future-weather inputs

35. `fcst_early_tmax_mean_anomaly`
36. `fcst_early_precip_sum_anomaly`
37. `fcst_early_solar_mean_anomaly`
38. `fcst_late_tmax_mean_anomaly`
39. `fcst_late_precip_sum_anomaly`
40. `fcst_late_solar_mean_anomaly`
41. `v2_bcc_fcst_early_tmax_mean_anomaly`
42. `v2_bcc_fcst_early_precip_sum_anomaly`
43. `v2_bcc_fcst_late_tmax_mean_anomaly`
44. `v2_bcc_fcst_late_precip_sum_anomaly`

### A.4 Multi-window physical and stress inputs

45. `v3_mv_w3_tmax_mean_z`
46. `v3_mv_w4_tmax_mean_z`
47. `v3_mv_w5_tmax_mean_z`
48. `v3_mv_w3_precip_sum_z`
49. `v3_mv_w4_precip_sum_z`
50. `v3_mv_w5_precip_sum_z`
51. `v3_mv_w4_solar_mean_z`
52. `v3_mv_w5_solar_mean_z`
53. `v3_mv_late_edd32_sum_z`
54. `v3_mv_late_edd34_sum_z`
55. `v3_mv_late_precip_sum_z`
56. `v3_mv_late_solar_mean_z`
57. `v3_mv_all_tmax_vintage_spread_mean`
58. `v3_terminal_heat_delayed`
59. `v3_april_heat_delayed`
60. `v3_stress_hot_dry`
61. `v3_stress_wet_cloud`
62. `v3_stress_worst`
63. `v3_bucket_start_fraction`
64. `v3_bucket_end_fraction`
65. `v3_bucket_unmet_demand_sum_mm`

### A.5 Shared state and regional stress inputs

66. `v3_state_mv_late_edd32_sum_z`
67. `v3_state_mv_late_precip_sum_z`
68. `v3_state_mv_late_solar_mean_z`
69. `v3_state_stress_hot_dry`
70. `v3_state_stress_wet_cloud`
71. `v3_region_mv_late_edd32_sum_z`
72. `v3_region_mv_late_precip_sum_z`
73. `v3_region_mv_late_solar_mean_z`
74. `v3_region_stress_hot_dry`
75. `v3_region_stress_wet_cloud`

### A.6 Economic inputs

76. `econ_msp_wheat_rs_quintal`
77. `econ_msp_yoy_change_pct`
78. `econ_msp_three_year_change_pct`

In addition to these 78 numeric inputs, state name is converted to three binary
one-hot columns.

---

## Appendix B. Exact 30 selected crop-encoder inputs

1. `enc__no_future_fused_pool_00`
2. `enc__no_future_fused_pool_01`
3. `enc__no_future_fused_pool_02`
4. `enc__no_future_fused_pool_03`
5. `enc__no_future_fused_pool_04`
6. `enc__no_future_fused_pool_05`
7. `enc__no_future_fused_pool_06`
8. `enc__no_future_fused_pool_07`
9. `enc__no_future_fused_pool_08`
10. `enc__no_future_fused_pool_09`
11. `enc__no_future_fused_pool_10`
12. `enc__no_future_fused_pool_11`
13. `enc__no_future_fused_pool_12`
14. `enc__no_future_fused_pool_13`
15. `enc__no_future_fused_pool_14`
16. `enc__no_future_fused_pool_15`
17. `enc__no_future_delta_index_0_mean`
18. `enc__no_future_delta_index_1_mean`
19. `enc__no_future_delta_index_2_mean`
20. `enc__no_future_delta_index_3_mean`
21. `enc__no_future_delta_index_4_mean`
22. `enc__no_future_delta_index_5_mean`
23. `enc__no_future_delta_abs_mean`
24. `enc__no_future_delta_positive_fraction`
25. `enc__current_index_0_mean`
26. `enc__current_index_1_mean`
27. `enc__current_index_2_mean`
28. `enc__current_index_3_mean`
29. `enc__current_index_4_mean`
30. `enc__current_index_5_mean`

Index order is NDVI, EVI, NDRE, NDMI, NIRv, PSRI.

---

## Appendix C. Authoritative implementation and audit files

The exact claims in this document are grounded in:

- `scripts/prepare_v15_data.py`
- `scripts/train_v15_encoder.py`
- `scripts/run_v15_integration.py`
- `scripts/run_v15_distribution.py`
- `scripts/run_v15_hierarchy.py`
- `scripts/train_v15_learned_normal.py`
- `scripts/finalize_v15.py`
- `scripts/validate_v15.py`
- `data/data_manifest.json`
- `artifacts/encoder_training_audit.csv`
- `artifacts/encoder_xgb_training_audit.csv`
- `artifacts/encoder_correction_selected_metrics.csv`
- `artifacts/integration_summary.json`
- `artifacts/distribution_summary.json`
- `artifacts/final_predictions.parquet`
- `artifacts/validation.json`

Inherited frozen definitions are grounded in:

- `../v5/root_cybench_lab/run_v5_integration.py`
- `../v5/root_cybench_lab/artifacts/v5_integration/model_lock.json`
- `../v14_anomaly_distribution/scripts/run_v14_lab.py`
- `../v14_anomaly_distribution/scripts/run_v14_extensions.py`
- `../v14_anomaly_distribution/scripts/finalize_v14.py`
- `../v14_anomaly_distribution/artifacts/final_predictions.parquet`

---

## Final one-paragraph summary

V15 begins with the V14 prediction, which is itself V5 plus an isolated
future-outlook correction. It trains a 67,359-parameter, width-32 crop
Transformer on long MODIS crop sequences and recent Sentinel crop transitions.
At inference, the Transformer compresses the current crop condition and
experienced weather into 30 crop-state values. A depth-2 XGBoost using those
values is compared with an otherwise identical XGBoost without them. V15 adds
1.25 times that difference to V14. It then moves a historically calibrated
q05–q95 residual distribution to the new point and scales its width by 0.95.
This gives one district-yield estimate, 19 yield quantiles, a probability of
yield increasing, and a probability of a fall greater than 10%.
