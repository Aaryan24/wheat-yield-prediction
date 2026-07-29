// Architecture deck — a component-by-component specification.
//
// Written for a reader with no prior exposure to the system, who needs enough
// detail to write a methods section. One idea per slide, a plain-English lead
// sentence before every specification, and no experimental narrative.
const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "figures");

const INK = "1B2430", NAVY = "1D4E89", ORANGE = "C2410C", GREEN = "15803D";
const MUTED = "6B7A8C", PALE = "F4F6F8", LINE = "E1E7ED", WHITE = "FFFFFF";
const SKY = "8FB4E3", MIST = "C5D3E5";
const HEAD = "Cambria", BODY = "Calibri", MONO = "Consolas";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "District wheat yield forecasting — system architecture";

const W = 13.3, M = 0.65;

function dark() {
  const s = pres.addSlide();
  s.background = { color: INK };
  return s;
}
// Every content slide: kicker, title, then one plain sentence saying what the
// component does before any specification appears.
function slide(kicker, title, lead) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  if (kicker) s.addText(kicker.toUpperCase(), {
    x: M, y: 0.32, w: 9, h: 0.26, fontFace: BODY, fontSize: 11,
    color: ORANGE, bold: true, charSpacing: 2, margin: 0 });
  s.addText(title, { x: M, y: 0.60, w: W - 2 * M, h: 0.55,
    fontFace: HEAD, fontSize: 28, bold: true, color: INK, margin: 0 });
  if (lead) s.addText(lead, { x: M, y: 1.22, w: W - 2 * M, h: 0.5,
    fontFace: BODY, fontSize: 15, color: MUTED, margin: 0 });
  return s;
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.05,
    fill: { color: fill || PALE }, line: { color: fill === INK ? INK : LINE } });
}
function bullets(s, x, y, w, h, items, fs, colour) {
  s.addText(items.map((t, i) => ({ text: t,
    options: { bullet: true, breakLine: i < items.length - 1 } })), {
    x, y, w, h, fontFace: BODY, fontSize: fs || 14,
    color: colour || INK, paraSpaceAfter: 8, margin: 0 });
}
function mono(s, x, y, w, text, fs, colour) {
  s.addText(text, { x, y, w, h: 0.4, fontFace: MONO, fontSize: fs || 13,
    color: colour || NAVY, bold: true, margin: 0 });
}
function note(s, text, y) {
  s.addText(text, { x: M, y: y, w: W - 2 * M, h: 0.45,
    fontFace: BODY, fontSize: 12.5, color: MUTED, italic: true, margin: 0 });
}
// label / value table with generous row height
function table(s, x, y, cols, rows, opts) {
  const o = opts || {};
  const lh = o.lh || 0.55, fs = o.fs || 13;
  let cy = y;
  if (o.head) {
    cols.forEach((c, i) => s.addText(o.head[i], {
      x: x + c[0], y: cy, w: c[1], h: 0.3, align: c[2] || "left",
      fontFace: BODY, fontSize: 11, color: MUTED, margin: 0 }));
    cy += 0.36;
  }
  rows.forEach((r, j) => {
    if (o.zebra !== false) card(s, x, cy, o.w, lh - 0.08, o.fill);
    r.forEach((v, i) => s.addText(v, {
      x: x + cols[i][0] + (i === 0 ? 0.28 : 0), y: cy, w: cols[i][1],
      h: lh - 0.08, valign: "middle", align: cols[i][2] || "left",
      fontFace: (o.monoCols || []).includes(i) ? MONO : BODY,
      fontSize: i === 0 ? fs : fs - 0.5, bold: i === 0 || (o.boldCols || []).includes(i),
      color: (o.colours || {})[i] || INK, margin: 0 }));
    cy += lh;
  });
  return cy;
}

// ═══════════════════════════════════════════════════ 1  title
{
  const s = dark();
  s.addText("Forecasting district wheat yield", {
    x: M, y: 2.0, w: 11.5, h: 0.85, fontFace: HEAD, fontSize: 40, bold: true,
    color: WHITE, margin: 0 });
  s.addText("System architecture and component specification", {
    x: M, y: 2.95, w: 11.5, h: 0.6, fontFace: HEAD, fontSize: 23,
    color: SKY, margin: 0 });
  s.addText("A probabilistic forecast for 119 Indian districts, issued on 5 March —\n" +
            "eight weeks before the harvest it predicts.", {
    x: M, y: 4.0, w: 11.5, h: 0.95, fontFace: BODY, fontSize: 16,
    color: MIST, lineSpacing: 26, margin: 0 });
}

// ═══════════════════════════════════════════════════ 2  the task
{
  const s = slide("The task", "Predict the harvest before it happens",
    "One number is not enough. The system outputs a full range of possible harvests and how likely each is.");
  const cols = [[0, 3.4], [3.4, 4.2], [7.6, 4.1]];
  table(s, M, 2.05, cols, [
    ["Predict", "wheat yield in kg/ha", "for each district, each season"],
    ["When", "5 March", "eight weeks before harvest"],
    ["Where", "119 districts", "Haryana, Punjab, Uttar Pradesh"],
    ["Typical yield", "≈ 4,500 kg/ha", "median district area 3,405 km²"],
  ], { w: 11.7, lh: 0.78, fs: 14 });

  card(s, M, 5.2, 11.7, 1.5, "FDF1EA");
  s.addText("The hard constraint", { x: M + 0.4, y: 5.4, w: 5, h: 0.3,
    fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("Nothing observed after 5 March may enter the forecast — no weather, no satellite image. Weather forecasts must have been published at least two days earlier. This is enforced in code, not by convention.", {
    x: M + 0.4, y: 5.75, w: 11.0, h: 0.8, fontFace: BODY, fontSize: 13.5,
    color: MUTED, margin: 0 });
}

// ═══════════════════════════════════════════════════ 3  what comes out
{
  const s = slide("The task", "What the system produces",
    "For every district and every season, a single row containing all of this.");
  const items = [
    ["Point forecast", "4,162 kg/ha", "the single best estimate"],
    ["19 quantiles", "q05 … q95", "the full range of plausible harvests"],
    ["P(yield rises)", "8%", "chance of beating last season"],
    ["P(fall > 5%)", "75%", "chance of a mild decline"],
    ["P(fall > 10%)", "46%", "chance of a serious decline"],
    ["P(fall > 20%)", "4%", "chance of a severe decline"],
  ];
  const cols = [[0, 3.2], [3.2, 2.6, "right"], [6.4, 5.3]];
  table(s, M, 2.05, cols, items, {
    w: 11.7, lh: 0.68, fs: 14, monoCols: [1], boldCols: [1],
    colours: { 1: NAVY, 2: MUTED } });
  note(s, "Values shown are the actual forecast for Rewari district, Haryana, in 2022 — used as the worked example later. Its harvest came in at 4,150 kg/ha.", 6.35);
}

// ═══════════════════════════════════════════════════ 4  system diagram
{
  const s = slide("Architecture", "The complete system");
  s.addImage({ path: path.join(FIG, "A1_system_architecture.png"),
    x: 0.32, y: 1.25, w: 12.65, h: 6.04 });
}

// ═══════════════════════════════════════════════════ 5  shape of the model
{
  const s = slide("Architecture", "How the pieces fit together",
    "The system is a conservative estimate, adjusted four times. Nothing replaces the estimate; each stage only nudges it.");
  card(s, M, 2.0, 11.7, 1.05, "EAF1FB");
  mono(s, M + 0.45, 2.32, 11.0,
    "ŷ  =  Blend   +   1.75 · c(weather ahead)   +   0.25 · c(training data)   +   2.25 · c(crop vision)", 14);

  const rows = [
    ["Stage 1", "Blend", "five simple estimates averaged, with a gate and a movement scale"],
    ["Stage 2", "Weather ahead", "what changes when the model may see the 10-day forecast"],
    ["Stage 3", "Training data", "what changes when the model is trained on four more seasons"],
    ["Stage 4", "Crop vision", "what changes when a network summarises satellite crop condition"],
    ["Stage 5", "Uncertainty", "turns the single number into 19 quantiles and every probability"],
  ];
  const cols = [[0, 1.5], [1.5, 2.6], [4.1, 7.4]];
  table(s, M, 3.3, cols, rows, { w: 11.7, lh: 0.66, fs: 13.5,
    colours: { 0: NAVY, 2: MUTED } });
  note(s, "Each “c” is a correction. The next slides define each one precisely.", 6.75);
}

// ═══════════════════════════════════════════════════ 6  data
{
  const s = slide("Inputs", "Where the information comes from");
  const cols = [[0, 3.3], [3.3, 2.0], [5.3, 3.1], [8.4, 3.3]];
  table(s, M, 1.5, cols, [
    ["Harvest records", "1990–2022", "3,658 seasons", "ground truth and history"],
    ["MODIS satellite", "2000–2022", "2,737 × 3 × 35", "crop condition, coarse"],
    ["Sentinel-2 satellite", "2017–2022", "2,142 × 126", "crop condition, fine"],
    ["Daily weather", "2010–2023", "546,805 days", "temperature, rain, sunlight"],
    ["Weather forecast", "per season", "10 × 5-day windows", "conditions after the clock"],
    ["Support price", "annual", "3 features", "economic signal"],
  ], { w: 11.7, lh: 0.72, fs: 13.5, monoCols: [2],
       head: ["Source", "Coverage", "Volume", "What it contributes"],
       colours: { 3: MUTED } });
  note(s, "Satellite indices measure how the crop reflects light: healthy plants reflect infrared strongly and red weakly, so the ratio tracks living plant material.", 6.15);
}

// ═══════════════════════════════════════════════════ 7  features
{
  const s = slide("Inputs", "The 78 numbers the models actually see",
    "Raw data is compressed into a fixed panel of 78 columns, computed identically for every district and season.");
  const cols = [[0, 0.9, "center"], [0.9, 3.5], [4.4, 7.1]];
  table(s, M, 2.05, cols, [
    ["18", "Yield history", "lags 1–5, five- and ten-year summaries, state mean, district minus state"],
    ["16", "Observed weather and soil", "early, mid and late season windows; soil moisture at three depths"],
    ["10", "Forecast weather", "temperature, rainfall and solar anomalies, plus bias-corrected variants"],
    ["31", "Stress indices", "heat degree-days above 32 °C and 34 °C, water balance, district and regional"],
    ["3", "Economics", "support price level, one-year and three-year changes"],
  ], { w: 11.7, lh: 0.82, fs: 13.5, monoCols: [0],
       colours: { 0: NAVY, 2: MUTED } });
  note(s, "Plus three binary state indicators, and a flag marking any value that was missing and had to be filled in.", 6.4);
}

// ═══════════════════════════════════════════════════ 8  baseline
{
  const s = slide("Stage 0", "The starting estimate",
    "Before any model runs, take a weighted average of the district's last three harvests.");
  card(s, M, 1.95, 11.7, 0.95, "FDF1EA");
  mono(s, M + 0.45, 2.22, 11.0, "b   =   0.60 · y(t−1)   +   0.25 · y(t−2)   +   0.15 · y(t−3)", 15, INK);

  s.addText("Why this works", { x: M, y: 3.15, w: 5.5, h: 0.35,
    fontFace: BODY, fontSize: 14, bold: true, color: INK, margin: 0 });
  bullets(s, M, 3.55, 5.4, 1.7, [
    "Yield is persistent — irrigation, soil and farming practice change slowly",
    "A district that produced 4,500 kg/ha for three years will probably do so again",
    "Weights decay because recent seasons are more informative",
  ], 13.5);

  s.addText("What every model predicts instead", { x: M + 6.3, y: 3.15, w: 5.5,
    h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: INK, margin: 0 });
  mono(s, M + 6.3, 3.58, 5.4, "r   =   y   −   b", 15, INK);
  s.addText("No model ever predicts yield directly. Each predicts the residual — how far this season departs from business as usual.", {
    x: M + 6.3, y: 4.1, w: 5.4, h: 0.9, fontFace: BODY, fontSize: 13.5,
    color: MUTED, margin: 0 });

  card(s, M, 5.35, 11.7, 1.35, INK);
  s.addText("Why that matters more than it looks", { x: M + 0.4, y: 5.55,
    w: 6, h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: SKY, margin: 0 });
  s.addText("No model has to learn that one district out-yields another — the baseline absorbs it. Every bit of learning capacity is spent on what is unusual about this season, which is the part that actually varies.", {
    x: M + 0.4, y: 5.92, w: 11.0, h: 0.65, fontFace: BODY, fontSize: 13.5,
    color: MIST, margin: 0 });
}

// ═══════════════════════════════════════════════════ 9  blend members
{
  const s = slide("Stage 1", "The Blend — five estimates, averaged",
    "Five separate models look at the season differently. Averaging cancels their independent mistakes.");
  const cols = [[0, 0.75, "center"], [0.75, 3.0], [3.75, 6.2], [9.95, 1.5, "right"]];
  table(s, M, 2.1, cols, [
    ["H", "Weighted history", "the baseline from the previous slide", "0.50"],
    ["R", "Weather + satellite", "trees on observed weather and satellite to 5 March", "0.15"],
    ["E", "Economic", "trees on lagged prices and support-price changes", "0.15"],
    ["C", "Global transfer", "a model trained on wheat elsewhere in the world", "0.20"],
    ["P", "Physics", "soil water balance and heat stress", "0.00"],
  ], { w: 11.7, lh: 0.76, fs: 14, monoCols: [0],
       colours: { 0: NAVY, 2: MUTED, 3: NAVY }, boldCols: [3] });

  card(s, M, 6.0, 11.7, 1.05, "FDF1EA");
  s.addText("The physics member has weight zero — it did not improve the average. It is kept because its direction still votes in the gate on the next slide, and a direction can be informative even when a level is not.", {
    x: M + 0.4, y: 6.22, w: 11.0, h: 0.65, fontFace: BODY, fontSize: 13,
    color: INK, margin: 0 });
}

// ═══════════════════════════════════════════════════ 10 gate
{
  const s = slide("Stage 1", "The disagreement gate",
    "Normally trust history. But when the other models agree that this season is different, stop leaning on last year.");
  card(s, M, 2.0, 11.7, 1.15, "EAF1FB");
  mono(s, M + 0.45, 2.22, 11.0, "s = sign( member − last season )       for R, P, E and C", 13);
  mono(s, M + 0.45, 2.66, 11.0, "S = s(R) + s(P) + s(E) + s(C)", 13);

  s.addText("The gate fires when both are true:", { x: M, y: 3.35, w: 6, h: 0.35,
    fontFace: BODY, fontSize: 14, bold: true, color: INK, margin: 0 });
  bullets(s, M, 3.75, 5.5, 1.1, [
    "| S | ≥ 2  — a majority agree on direction",
    "that direction contradicts what history implies",
  ], 13.5);

  card(s, M + 6.2, 3.35, 5.5, 1.6);
  s.addText("Weights switch", { x: M + 6.55, y: 3.5, w: 5, h: 0.3,
    fontFace: BODY, fontSize: 12.5, bold: true, color: INK, margin: 0 });
  mono(s, M + 6.55, 3.85, 5.0, "normal   0.50 H · 0.15 R · 0.15 E · 0.20 C", 11);
  mono(s, M + 6.55, 4.25, 5.0, "fires    0.20 H · 0.30 R · 0.30 E · 0.20 C", 11, ORANGE);

  card(s, M, 5.3, 11.7, 1.35, INK);
  s.addText("In practice", { x: M + 0.4, y: 5.5, w: 5, h: 0.3, fontFace: BODY,
    fontSize: 13, bold: true, color: SKY, margin: 0 });
  s.addText("The gate fires on 20.6% of district-seasons. On the other 79.4% the blend stays anchored to history, which is the safe default when nothing unusual is being signalled.", {
    x: M + 0.4, y: 5.86, w: 11.0, h: 0.6, fontFace: BODY, fontSize: 13.5,
    color: MIST, margin: 0 });
}

// ═══════════════════════════════════════════════════ 11 calibration
{
  const s = slide("Stage 1", "Movement calibration",
    "Averaging five models pulls the result toward the middle. The blend under-moves, so its movement is scaled back up.");
  card(s, M, 2.05, 11.7, 0.95, "EAF1FB");
  mono(s, M + 0.45, 2.32, 11.0, "Blend   =   L   +   1.50 · ( Blend_raw  −  L )        where L = last season", 14);

  s.addText("How the 1.50 was obtained", { x: M, y: 3.3, w: 6, h: 0.35,
    fontFace: BODY, fontSize: 14, bold: true, color: INK, margin: 0 });
  s.addText("Take historical cases. Let x be how far the raw blend moved from last season, and z how far the truth actually moved. Fit the single number that best rescales one into the other, by least squares through the origin:", {
    x: M, y: 3.7, w: 5.5, h: 1.2, fontFace: BODY, fontSize: 13.5,
    color: MUTED, margin: 0 });
  mono(s, M, 4.85, 5.5, "s  =  Σ xᵢ zᵢ  /  Σ xᵢ²", 14);

  card(s, M + 6.2, 3.3, 5.5, 2.35, "FDF1EA");
  s.addText("What it means", { x: M + 6.55, y: 3.5, w: 5, h: 0.3,
    fontFace: BODY, fontSize: 12.5, bold: true, color: INK, margin: 0 });
  s.addText("The answer came out near 1.5.\n\nWhen the blend says “up 100 kg/ha”, the truth has historically moved about 150. The scale undoes the shrinkage that averaging introduced.", {
    x: M + 6.55, y: 3.88, w: 5.0, h: 1.6, fontFace: BODY, fontSize: 13.5,
    color: INK, lineSpacing: 21, margin: 0 });

  card(s, M, 5.95, 11.7, 1.0, INK);
  s.addText("Stage 1 produces one conservative yield estimate per district-season. Stages 2 to 4 adjust it; none of them replaces it.", {
    x: M + 0.4, y: 6.22, w: 11.0, h: 0.5, fontFace: BODY, fontSize: 14,
    color: WHITE, margin: 0 });
}

// ═══════════════════════════════════════════════════ 12 matched design
{
  const s = slide("Stages 2–4", "How each correction is measured");
  card(s, M, 1.45, 11.7, 1.85, "EAF1FB");
  s.addText("Train two models that are identical in every respect — same rows, same target, same settings, same random seeds — except that one is given an extra piece of information.\n\nThe difference between their predictions is what that information contributes, and nothing else.", {
    x: M + 0.45, y: 1.68, w: 11.0, h: 1.5, fontFace: BODY, fontSize: 15,
    color: INK, lineSpacing: 24, margin: 0 });

  s.addText("This is the same logic as a drug trial: two groups treated identically apart from one thing, so any difference is attributable to that thing.", {
    x: M, y: 3.5, w: 11.7, h: 0.4, fontFace: BODY, fontSize: 13.5,
    color: MUTED, italic: true, margin: 0 });

  const cols = [[0, 1.5], [1.5, 3.6], [5.1, 5.0], [10.1, 1.4, "right"]];
  table(s, M, 4.05, cols, [
    ["Stage 2", "the 10-day weather forecast", "mean(FULL, EFFECT, BROAD) − NO-FUTURE", "×1.75"],
    ["Stage 3", "four more seasons of training data", "trained-from-2013 − trained-from-2017", "×0.25"],
    ["Stage 4", "16 values from the crop network", "with-those-values − without", "×2.25"],
  ], { w: 11.7, lh: 0.82, fs: 13.5, monoCols: [2],
       colours: { 0: NAVY, 2: MUTED, 3: ORANGE }, boldCols: [3] });
  note(s, "Each weight is deliberately less than one: a correction that cannot be measured precisely is applied at a fraction of its face value, so noise cannot dominate.", 6.75);
}

// ═══════════════════════════════════════════════════ 13 tree spec
{
  const s = slide("Stages 2–4", "The prediction models underneath",
    "Every correction is produced by gradient-boosted decision trees — hundreds of small yes/no question trees, averaged.");
  card(s, M, 2.0, 5.6, 3.65);
  s.addText("Settings", { x: M + 0.35, y: 2.18, w: 5, h: 0.3, fontFace: BODY,
    fontSize: 13, bold: true, color: INK, margin: 0 });
  const specs = [["Trees", "350"], ["Maximum depth", "2"], ["Learning rate", "0.025"],
    ["Minimum child weight", "25"], ["Row subsample", "0.85"], ["Column subsample", "0.65"],
    ["L2 penalty", "50"], ["L1 penalty", "5"], ["Random seeds", "42 and 73, averaged"]];
  specs.forEach(([k, v], i) => {
    const y = 2.58 + i * 0.33;
    s.addText(k, { x: M + 0.35, y, w: 3.1, h: 0.3, fontFace: BODY, fontSize: 12,
      color: MUTED, margin: 0 });
    s.addText(v, { x: M + 3.45, y, w: 1.9, h: 0.3, fontFace: MONO, fontSize: 12,
      bold: true, color: INK, margin: 0 });
  });

  card(s, M + 6.1, 2.0, 5.6, 1.75, "FDF1EA");
  s.addText("Why depth 2", { x: M + 6.45, y: 2.18, w: 5, h: 0.3, fontFace: BODY,
    fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("Each tree may ask only two yes/no questions. With few independent harvests to learn from, a deeper tree would memorise districts instead of learning how they respond.", {
    x: M + 6.45, y: 2.55, w: 5.0, h: 1.05, fontFace: BODY, fontSize: 13,
    color: INK, margin: 0 });

  card(s, M + 6.1, 3.95, 5.6, 1.7);
  s.addText("Preparation", { x: M + 6.45, y: 4.12, w: 5, h: 0.3, fontFace: BODY,
    fontSize: 13, bold: true, color: INK, margin: 0 });
  bullets(s, M + 6.45, 4.48, 5.0, 1.05, [
    "Missing values filled with the column median, plus a flag",
    "State encoded as three binary columns",
    "Predictions clipped to 500–7,000 kg/ha",
  ], 12.5);
  note(s, "Identical settings across every matched pair, so a measured difference can only come from the extra information — never from a different configuration.", 5.9);
}

// ═══════════════════════════════════════════════════ 14 why a network
{
  const s = slide("Stage 4", "Why a neural network at all",
    "The satellite sees the crop. The question is how to turn a large, messy pile of measurements into a few useful numbers.");
  card(s, M, 2.0, 5.6, 2.2);
  s.addText("The obvious approach", { x: M + 0.35, y: 2.2, w: 5, h: 0.3,
    fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("Compute summary statistics — average greenness, its spread, its 10th percentile. This works, but the recipe is fixed by the analyst in advance, and it treats every part of a district as interchangeable.", {
    x: M + 0.35, y: 2.58, w: 5.0, h: 1.4, fontFace: BODY, fontSize: 13,
    color: MUTED, margin: 0 });

  card(s, M + 6.1, 2.0, 5.6, 2.2, "FDF1EA");
  s.addText("What a network does instead", { x: M + 6.45, y: 2.2, w: 5, h: 0.3,
    fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("It learns the recipe. Formally it is a function that maps the whole pile to 16 numbers — and the entire design question is which 16.", {
    x: M + 6.45, y: 2.58, w: 5.0, h: 1.4, fontFace: BODY, fontSize: 13,
    color: INK, margin: 0 });

  card(s, M, 4.5, 11.7, 2.2, INK);
  s.addText("The decision that shapes everything: it is never shown a harvest figure", {
    x: M + 0.45, y: 4.75, w: 11.0, h: 0.35, fontFace: BODY, fontSize: 14,
    bold: true, color: SKY, margin: 0 });
  s.addText("There are thousands of examples of how a crop changes between January and March, and only a handful of harvests. So the network is trained on the data-rich question — how does wheat develop — and its internal summary is handed to a simple model for the data-poor question of what the yield will be.", {
    x: M + 0.45, y: 5.2, w: 11.0, h: 1.2, fontFace: BODY, fontSize: 14,
    color: MIST, lineSpacing: 23, margin: 0 });
}

// ═══════════════════════════════════════════════════ 15 network diagram
{
  const s = slide("Stage 4", "The crop-vision network");
  s.addImage({ path: path.join(FIG, "A2_network_architecture.png"),
    x: 0.32, y: 1.25, w: 12.65, h: 6.04 });
}

// ═══════════════════════════════════════════════════ 16 network inputs
{
  const s = slide("Stage 4", "What the network is shown",
    "Three streams of numbers. Each row of a stream is called a token — the unit the network reasons over.");
  const cols = [[0, 3.0], [3.0, 1.7, "right"], [4.9, 6.6]];
  table(s, M, 2.05, cols, [
    ["Crop state", "3 × 126", "6 vegetation indices × 3 spatial views × 7 time summaries,\nat 15 January, 15 February and 5 March"],
    ["Past weather", "6 × 16", "six windows before the clock: temperature, rainfall, solar\nradiation, humidity, wind, soil moisture at three depths"],
    ["Forecast weather", "10 × 16", "ten dated five-day windows ahead, with spatial spread and\ncounts of days above 32 °C and 34 °C"],
  ], { w: 11.7, lh: 1.15, fs: 14, monoCols: [1],
       colours: { 1: NAVY, 2: MUTED }, boldCols: [1] });

  card(s, M, 5.75, 11.7, 1.2, "FDF1EA");
  s.addText("Masking — how the leakage rule is enforced inside the network", {
    x: M + 0.4, y: 5.95, w: 8, h: 0.3, fontFace: BODY, fontSize: 13,
    bold: true, color: INK, margin: 0 });
  s.addText("At the 15 February clock the March token does not yet exist. Its attention score is set to minus infinity, so it receives exactly zero weight and cannot influence anything.", {
    x: M + 0.4, y: 6.3, w: 11.0, h: 0.55, fontFace: BODY, fontSize: 13,
    color: MUTED, margin: 0 });
}

// ═══════════════════════════════════════════════════ 17 attention
{
  const s = slide("Stage 4", "Attention — the one operation to understand",
    "Each token looks at all the others, decides how much each one matters, and takes a weighted average.");
  card(s, M, 2.0, 11.7, 0.95, "EAF1FB");
  mono(s, M + 0.45, 2.28, 11.0, "Attention( Q, K, V )   =   softmax(  Q · Kᵀ / √8   +   M  ) · V", 15);

  const terms = [
    ["Q · Kᵀ", "every token's question is compared against every token's label — a large value means “you are relevant to me”"],
    ["÷ √8", "keeps those numbers in a sensible range as the network gets wider"],
    ["+ M", "the mask: minus infinity for anything not yet observable, so it gets zero weight"],
    ["softmax", "turns the scores into weights that are positive and add up to one"],
    ["· V", "the output is a weighted average of the other tokens' content"],
  ];
  let y = 3.2;
  terms.forEach(([t, d]) => {
    s.addText(t, { x: M + 0.1, y, w: 1.6, h: 0.62, valign: "middle",
      fontFace: MONO, fontSize: 14, bold: true, color: ORANGE, margin: 0 });
    s.addText(d, { x: M + 1.85, y, w: 9.9, h: 0.62, valign: "middle",
      fontFace: BODY, fontSize: 13.5, color: INK, margin: 0 });
    y += 0.66;
  });
  note(s, "Four of these run in parallel with separate learned weights, so the network can attend to several different things at once.", 6.6);
}

// ═══════════════════════════════════════════════════ 18 cross-attention
{
  const s = slide("Stage 4", "The crop asks the weather two questions",
    "The crop token for the current date queries each weather stream separately.");
  card(s, M, 2.0, 5.6, 2.0);
  mono(s, M + 0.35, 2.25, 5.0, "c_past = MHA( q , past )", 12);
  s.addText("“Which past weather explains the state I am in?”", {
    x: M + 0.35, y: 2.75, w: 5.0, h: 0.9, fontFace: BODY, fontSize: 14,
    italic: true, color: INK, margin: 0 });

  card(s, M + 6.1, 2.0, 5.6, 2.0);
  mono(s, M + 6.45, 2.25, 5.0, "c_fcst = MHA( q , forecast )", 12);
  s.addText("“Which forecast weather threatens what happens to me next?”", {
    x: M + 6.45, y: 2.75, w: 5.0, h: 0.9, fontFace: BODY, fontSize: 14,
    italic: true, color: INK, margin: 0 });

  card(s, M, 4.25, 11.7, 1.05, "EAF1FB");
  mono(s, M + 0.45, 4.53, 11.0, "h   =   LayerNorm(  q   +   c_past   +   c_fcst  )", 14);

  s.addText("LayerNorm rescales the vector to a stable range. Adding q back is a residual connection: information can bypass a layer that does not help it.", {
    x: M, y: 5.5, w: 7.0, h: 0.9, fontFace: BODY, fontSize: 13.5,
    color: MUTED, margin: 0 });
  card(s, M + 7.4, 5.4, 4.3, 1.2, "FDF1EA");
  s.addText("Result", { x: M + 7.7, y: 5.55, w: 3.8, h: 0.28, fontFace: BODY,
    fontSize: 12, bold: true, color: INK, margin: 0 });
  s.addText("one 32-value vector per district-season; 16 are kept", {
    x: M + 7.7, y: 5.86, w: 3.8, h: 0.6, fontFace: BODY, fontSize: 13,
    color: INK, margin: 0 });
}

// ═══════════════════════════════════════════════════ 19 training
{
  const s = slide("Stage 4", "How the network is trained",
    "Two steps, neither of which uses a harvest figure.");
  card(s, M, 2.0, 5.6, 2.05);
  s.addText("Step 1 — learn general crop development", { x: M + 0.35, y: 2.2,
    w: 5.0, h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("On coarse satellite data back to 2000, predict the next observation\nfrom the previous ones.", {
    x: M + 0.35, y: 2.58, w: 5.0, h: 0.7, fontFace: BODY, fontSize: 13,
    color: MUTED, margin: 0 });
  mono(s, M + 0.35, 3.3, 5.0, "AdamW · lr 8e−4 · 24 epochs", 11, MUTED);

  card(s, M + 6.1, 2.0, 5.6, 2.05);
  s.addText("Step 2 — specialise on recent, sharper data", { x: M + 6.45, y: 2.2,
    w: 5.0, h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("On fine satellite data from 2017, predict how each of the 126 crop\nmeasurements will change.", {
    x: M + 6.45, y: 2.58, w: 5.0, h: 0.7, fontFace: BODY, fontSize: 13,
    color: MUTED, margin: 0 });
  mono(s, M + 6.45, 3.3, 5.0, "AdamW · lr 6e−4 · 45 epochs · seeds 42, 73", 11, MUTED);

  card(s, M, 4.3, 11.7, 1.1, "EAF1FB");
  mono(s, M + 0.45, 4.58, 11.0, "Loss  =  SmoothL1( predicted change, actual change )   +   0.08 · BCE( predicted direction )", 12);

  s.addText("Smooth L1 behaves like squared error for small mistakes and like absolute error for large ones, so a cloud or a sensor fault cannot dominate training. Targets are standardised using the median and median-absolute-deviation for the same reason.", {
    x: M, y: 5.6, w: 11.7, h: 0.9, fontFace: BODY, fontSize: 13.5,
    color: MUTED, margin: 0 });
}

// ═══════════════════════════════════════════════════ 20 leakage
{
  const s = slide("Protocol", "Preventing the model from cheating",
    "Two mechanisms, both essential when the same districts appear in training and testing.");
  card(s, M, 2.05, 5.6, 2.55);
  s.addText("Time cutoffs", { x: M + 0.35, y: 2.25, w: 5, h: 0.3, fontFace: BODY,
    fontSize: 13, bold: true, color: INK, margin: 0 });
  bullets(s, M + 0.35, 2.62, 5.0, 1.8, [
    "For a target season, nothing from that season or later is used anywhere",
    "This includes the averages used to standardise inputs",
    "Weather forecasts must predate the clock by two days",
  ], 13);

  card(s, M + 6.1, 2.05, 5.6, 2.55);
  s.addText("District cross-fitting", { x: M + 6.45, y: 2.25, w: 5, h: 0.3,
    fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("Districts are split into three groups. The 16 network values for a district in group 1 are produced by a network trained without group 1.\n\nOtherwise those values would look informative simply because the network had memorised the district.", {
    x: M + 6.45, y: 2.62, w: 5.0, h: 1.85, fontFace: BODY, fontSize: 13,
    color: MUTED, lineSpacing: 20, margin: 0 });

  card(s, M, 4.9, 11.7, 1.85, INK);
  s.addText("Evaluation protocol", { x: M + 0.45, y: 5.1, w: 6, h: 0.3,
    fontFace: BODY, fontSize: 13, bold: true, color: SKY, margin: 0 });
  bullets(s, M + 0.45, 5.48, 11.0, 1.1, [
    "Rolling origin — the model for each season is trained only on earlier seasons",
    "Correction weights are chosen on development seasons and then held fixed",
    "Later seasons are used once, for confirmation, never to choose anything",
  ], 13, MIST);
}

// ═══════════════════════════════════════════════════ 21 distribution
{
  const s = slide("Stage 5", "Turning one number into a distribution",
    "Rather than assuming a bell curve, use the errors the model has actually made in the past.");
  const cols = [[0, 0.62, "center"], [0.62, 3.1], [3.72, 7.9]];
  table(s, M, 1.95, cols, [
    ["1", "Collect past errors", "e = y − ŷ, using only seasons before the target"],
    ["2", "Put them on a common scale", "a = max( recent SD, 7% of normal yield, 150 )    u = e / a"],
    ["3", "Weight seasons equally", "each row gets weight 1/n(t), so a big season cannot dominate"],
    ["4", "Take weighted quantiles", "interpolate u at the 19 probability levels"],
    ["5", "Rescale and recentre", "Q(α) = ŷ + a · u(α)"],
    ["6", "Enforce validity", "running maximum so quantiles cannot cross, then clip"],
  ], { w: 11.7, lh: 0.74, fs: 13.5, monoCols: [2],
       colours: { 0: NAVY, 2: MUTED } });
  note(s, "Step 2 matters: a 300 kg/ha error means something different in a stable district than a volatile one, so errors are divided by a district-specific size before being pooled.", 6.5);
}

// ═══════════════════════════════════════════════════ 22 probabilities
{
  const s = slide("Stage 5", "From quantiles to answers",
    "The 19 quantiles are the distribution. Every question anyone asks is a lookup on it.");
  card(s, M, 2.0, 5.6, 1.9);
  s.addText("The quantiles define F", { x: M + 0.35, y: 2.2, w: 5, h: 0.3,
    fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  mono(s, M + 0.35, 2.55, 5.0, "F( Q(α) )  =  α", 14);
  s.addText("F is the chance the harvest comes in at or below a given yield. It is known at 19 points and filled in between by straight lines.", {
    x: M + 0.35, y: 3.05, w: 5.0, h: 0.75, fontFace: BODY, fontSize: 13,
    color: MUTED, margin: 0 });

  card(s, M + 6.1, 2.0, 5.6, 1.9);
  s.addText("Every question is a lookup", { x: M + 6.45, y: 2.2, w: 5, h: 0.3,
    fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  mono(s, M + 6.45, 2.58, 5.0, "P( rises )      = 1 − F( last )", 11.5);
  mono(s, M + 6.45, 2.98, 5.0, "P( fall > p )   = F( (1−p) · last )", 11.5);
  s.addText("Published at 5%, 10%, 20% and 30%.", { x: M + 6.45, y: 3.45,
    w: 5.0, h: 0.35, fontFace: BODY, fontSize: 13, color: MUTED, margin: 0 });

  card(s, M, 4.2, 11.7, 1.45, "EAF1FB");
  s.addText("The density curve is the slope of F", { x: M + 0.4, y: 4.4, w: 8,
    h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("Where quantiles bunch together, a lot of probability sits in a narrow band of yields, so the curve is tall there. Where they spread out, the curve is low.", {
    x: M + 0.4, y: 4.76, w: 11.0, h: 0.6, fontFace: BODY, fontSize: 13.5,
    color: MUTED, margin: 0 });

  card(s, M, 5.95, 11.7, 1.0, INK);
  s.addText("Output: one row per district-season with the point forecast, all 19 quantiles, and every threshold probability.", {
    x: M + 0.45, y: 6.22, w: 11.0, h: 0.5, fontFace: BODY, fontSize: 14,
    color: WHITE, margin: 0 });
}

// ═══════════════════════════════════════════════════ 23 inference
{
  const s = slide("Operation", "Running the model for one district",
    "Nine steps, roughly three hours end to end from raw data on a single laptop.");
  const steps = [
    "Assemble lagged yield history and compute the baseline",
    "Compute the 78 features from weather, soil, satellite and price data",
    "Evaluate the five Blend members, apply the gate, apply the ×1.50 movement scale",
    "Run the four weather-forecast models; take their difference; add at ×1.75",
    "Run the two training-window models; take their difference; add at ×0.25",
    "Build the crop tensors, mask anything after the clock, run the network, take 16 values",
    "Run the two matched tree models with and without them; add the difference at ×2.25",
    "Scale the historical error distribution to the district and recentre it",
    "Interpolate the quantile curve and read off every threshold probability",
  ];
  let y = 2.0;
  steps.forEach((t, i) => {
    s.addText(String(i + 1), { x: M + 0.02, y, w: 0.45, h: 0.52, align: "center",
      valign: "middle", fontFace: MONO, fontSize: 13, bold: true,
      color: ORANGE, margin: 0 });
    s.addText(t, { x: M + 0.6, y, w: 11.1, h: 0.52, valign: "middle",
      fontFace: BODY, fontSize: 13.5, color: INK, margin: 0 });
    if (i < steps.length - 1) s.addShape(pres.ShapeType.line, {
      x: M + 0.6, y: y + 0.52, w: 11.0, h: 0, line: { color: LINE, width: 0.75 } });
    y += 0.545;
  });
}

// ═══════════════════════════════════════════════════ 24 worked example
{
  const s = slide("Worked example", "Rewari district, Haryana — 2022");
  s.addImage({ path: path.join(FIG, "02_worked_example.png"),
    x: 1.05, y: 1.3, w: 11.2, h: 5.15 });
}

// ═══════════════════════════════════════════════════ 25 worked forecast
{
  const s = slide("Worked example", "The forecast that is issued");
  s.addImage({ path: path.join(FIG, "07_probability_distribution.png"),
    x: 0.42, y: 1.4, w: 8.2, h: 4.26 });
  const rows = [["Point forecast", "4,162"], ["80% range", "3,822 – 4,529"],
                ["P(increase)", "8%"], ["P(fall > 5%)", "75%"],
                ["P(fall > 10%)", "46%"], ["P(fall > 20%)", "4%"]];
  let y = 1.55;
  rows.forEach(([k, v]) => {
    s.addText(k, { x: 8.95, y, w: 2.5, h: 0.45, valign: "middle",
      fontFace: BODY, fontSize: 13, color: MUTED, margin: 0 });
    s.addText(v, { x: 11.3, y, w: 1.65, h: 0.45, align: "right", valign: "middle",
      fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
    y += 0.52;
  });
  s.addText("Actual harvest  4,150", { x: 8.95, y: 4.8, w: 4.0, h: 0.4,
    fontFace: HEAD, fontSize: 16, bold: true, color: ORANGE, margin: 0 });
  s.addText("A 9.4% fall. The forecast placed it at the 10% boundary and put a decline at 92%.", {
    x: 8.95, y: 5.28, w: 4.0, h: 0.9, fontFace: BODY, fontSize: 12.5,
    color: MUTED, margin: 0 });
}

// ═══════════════════════════════════════════════════ 26 results march
{
  const s = slide("Results", "Accuracy of the 5 March forecast",
    "Measured on seasons the model never saw during fitting.");
  const cols = [[0, 3.6], [3.6, 2.0, "right"], [5.6, 2.0, "right"],
                [7.6, 2.0, "right"], [9.6, 2.0, "right"]];
  table(s, M, 2.05, cols, [
    ["Baseline (3-season average)", "—", "—", "333.1", "—"],
    ["Stage 1 — Blend", "257.0", "288.6", "273.3", "77.9%"],
    ["+ Stage 2 weather ahead", "254.1", "288.1", "271.7", "78.8%"],
    ["+ Stages 3 and 4 — full model", "248.6", "281.2", "265.4", "78.8%"],
  ], { w: 11.7, lh: 0.78, fs: 13.5, monoCols: [1, 2, 3, 4],
       head: ["Model", "2019–20", "2021–22", "four-year", "direction"],
       colours: { 3: NAVY }, boldCols: [3] });

  card(s, M, 5.4, 11.7, 1.3, "EAF1FB");
  s.addText("Typical error of 265 kg/ha is about 5.9% of a 4,500 kg/ha harvest. Direction accuracy is the share of districts where the forecast landed on the correct side of last season.", {
    x: M + 0.4, y: 5.7, w: 11.0, h: 0.75, fontFace: BODY, fontSize: 13.5,
    color: INK, margin: 0 });
}

// ═══════════════════════════════════════════════════ 27 january
{
  const s = slide("Results", "The same system run seven weeks earlier",
    "A forecast can also be issued on 15 January. It is usable, and clearly weaker.");
  const cols = [[0, 3.4], [3.4, 2.1, "right"], [5.5, 2.1, "right"],
                [7.6, 2.1, "right"], [9.7, 1.9, "right"]];
  table(s, M, 2.05, cols, [
    ["15 January", "296.8", "316.9", "307.0", "70.2%"],
    ["5 March", "248.6", "281.2", "265.4", "78.8%"],
  ], { w: 11.7, lh: 0.82, fs: 14, monoCols: [1, 2, 3, 4],
       head: ["Forecast date", "2019–20", "2021–22", "four-year", "direction"],
       colours: { 3: NAVY }, boldCols: [3] });

  card(s, M, 4.0, 5.6, 2.5);
  s.addText("What waiting buys", { x: M + 0.35, y: 4.2, w: 5, h: 0.3,
    fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  bullets(s, M + 0.35, 4.58, 5.0, 1.7, [
    "41.6 kg/ha less error — about 13% better",
    "8.6 more percentage points of direction accuracy",
    "The crop has entered grain filling and is visible to the satellite",
  ], 13);

  card(s, M + 6.1, 4.0, 5.6, 2.5, "FDF1EA");
  s.addText("Both dates are honestly calibrated", { x: M + 6.45, y: 4.2, w: 5,
    h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0 });
  const jc = [["80% ranges cover", "79.3%", "80.9%"],
              ["90% ranges cover", "90.5%", "91.8%"]];
  s.addText("15 Jan", { x: M + 8.9, y: 4.55, w: 1.2, h: 0.28, align: "right",
    fontFace: BODY, fontSize: 11, color: MUTED, margin: 0 });
  s.addText("5 Mar", { x: M + 10.2, y: 4.55, w: 1.2, h: 0.28, align: "right",
    fontFace: BODY, fontSize: 11, color: MUTED, margin: 0 });
  jc.forEach(([k, a, b], i) => {
    const y = 4.9 + i * 0.42;
    s.addText(k, { x: M + 6.45, y, w: 2.5, h: 0.32, fontFace: BODY,
      fontSize: 12.5, color: INK, margin: 0 });
    s.addText(a, { x: M + 8.9, y, w: 1.2, h: 0.32, align: "right",
      fontFace: MONO, fontSize: 12.5, bold: true, color: MUTED, margin: 0 });
    s.addText(b, { x: M + 10.2, y, w: 1.2, h: 0.32, align: "right",
      fontFace: MONO, fontSize: 12.5, bold: true, color: GREEN, margin: 0 });
  });
  s.addText("Both should read 80% and 90%.", { x: M + 6.45, y: 5.85, w: 5.0,
    h: 0.4, fontFace: BODY, fontSize: 12.5, color: MUTED, italic: true, margin: 0 });
}

// ═══════════════════════════════════════════════════ 28 january example
{
  const s = slide("Results", "One district, both forecast dates",
    "Rewari 2022. Last season 4,580 kg/ha; the harvest came in at 4,150.");
  s.addImage({ path: path.join(FIG, "11_january_vs_march.png"),
    x: 0.55, y: 1.85, w: 7.6, h: 3.65 });
  const cols = [[0, 1.85], [1.85, 1.15, "right"], [3.05, 1.15, "right"]];
  table(s, 8.45, 1.95, cols, [
    ["Forecast", "4,555", "4,162"],
    ["Error", "405", "12"],
    ["P(increase)", "43%", "8%"],
    ["P(fall > 10%)", "9%", "46%"],
  ], { w: 4.25, lh: 0.72, fs: 13, monoCols: [1, 2],
       head: ["", "15 Jan", "5 Mar"], colours: { 1: MUTED, 2: ORANGE },
       boldCols: [1, 2] });
  s.addText("In January the model expected a good year and put a serious decline at 9%. By March it had seen the crop, moved the forecast down 393 kg/ha, and raised that probability to 46%.", {
    x: 8.45, y: 5.05, w: 4.25, h: 1.5, fontFace: BODY, fontSize: 12.5,
    color: MUTED, margin: 0 });
}

// ═══════════════════════════════════════════════════ 29 calibration
{
  const s = slide("Results", "Are the probabilities honest?",
    "A probability is only useful if it is true. Across all 476 district-seasons of the 5 March forecast:");
  const cols = [[0, 4.2], [4.2, 2.6, "right"], [6.8, 2.8, "right"]];
  table(s, M, 2.15, cols, [
    ["Fall worse than 5%", "29.9%", "29.2%"],
    ["Fall worse than 10%", "11.5%", "10.9%"],
    ["Fall worse than 20%", "1.2%", "2.3%"],
    ["Any increase", "45.9%", "43.5%"],
  ], { w: 9.9, lh: 0.8, fs: 14, monoCols: [1, 2],
       head: ["When the model says…", "stated", "actually happened"],
       colours: { 1: MUTED, 2: GREEN }, boldCols: [1, 2] });

  card(s, 11.0, 2.5, 1.75, 2.4, "EAF1FB");
  s.addText("80.9%", { x: 11.0, y: 3.05, w: 1.75, h: 0.6, align: "center",
    fontFace: HEAD, fontSize: 24, bold: true, color: GREEN, margin: 0 });
  s.addText("of harvests\nland inside\nthe 80%\nrange", { x: 11.0, y: 3.7,
    w: 1.75, h: 1.0, align: "center", fontFace: BODY, fontSize: 11.5,
    color: MUTED, lineSpacing: 15, margin: 0 });

  s.addText("An event the model calls 10% likely happens on about 10% of occasions. That is what makes these numbers usable for a decision rather than decoration.", {
    x: M, y: 5.55, w: 11.7, h: 0.5, fontFace: BODY, fontSize: 14,
    color: INK, margin: 0 });
  note(s, "One honest exception: the extreme tail is mildly over-confident — 1.2% stated for declines worse than 20%, against 2.3% observed.", 6.25);
}

// ═══════════════════════════════════════════════════ 30 close
{
  const s = dark();
  s.addText("The architecture in four sentences", { x: M, y: 0.85, w: 11.9,
    h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, margin: 0 });
  const items = [
    ["A conservative baseline", "a weighted average of the last three harvests; every model predicts the residual around it, so none has to learn what a district is normally worth"],
    ["Four corrections", "each measured as the difference between two otherwise identical models, and each deliberately shrunk before it is applied"],
    ["One small network", "67,359 parameters, trained on how crops develop rather than on yield, contributing 16 numbers to the final estimate"],
    ["An empirical distribution", "the model's own past errors, rescaled and recentred, giving 19 quantiles and every threshold probability"],
  ];
  let y = 1.85;
  items.forEach(([k, v]) => {
    s.addText(k, { x: M, y, w: 4.0, h: 0.5, fontFace: HEAD, fontSize: 16,
      bold: true, color: SKY, margin: 0 });
    s.addText(v, { x: M + 4.2, y, w: 7.6, h: 1.0, fontFace: BODY, fontSize: 13,
      color: MIST, lineSpacing: 20, margin: 0 });
    y += 1.2;
  });
  s.addText("Output: a point forecast, a calibrated distribution, and an early warning of collapse — for 119 districts, eight weeks before harvest.", {
    x: M, y: 6.65, w: 11.9, h: 0.6, fontFace: BODY, fontSize: 13.5,
    color: ORANGE, italic: true, margin: 0 });
}

const out = path.join(ROOT, "wheat_yield_architecture.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote " + out));
