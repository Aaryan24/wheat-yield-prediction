// Builds the presentation deck from the verified walkthrough numbers.
// Palette matches the figures exactly (navy / burnt orange / ink) so slides and
// charts read as one system.
const pptxgen = require("pptxgenjs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "figures");

const INK = "1B2430";
const NAVY = "1D4E89";
const ORANGE = "C2410C";
const GREEN = "15803D";
const MUTED = "6B7A8C";
const PALE = "F4F6F8";
const WHITE = "FFFFFF";

const HEAD = "Cambria";
const BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5 -- set before any slide
pres.author = "Wheat yield forecasting";
pres.title = "Forecasting wheat yield eight weeks before harvest";

const W = 13.3, H = 7.5, M = 0.7;

// ---------------------------------------------------------------- helpers
function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: INK };
  return s;
}
function lightSlide(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.42, w: 8, h: 0.28, fontFace: BODY, fontSize: 11,
      color: ORANGE, bold: true, charSpacing: 2, margin: 0,
    });
  }
  if (title) {
    s.addText(title, {
      x: M, y: kicker ? 0.74 : 0.5, w: W - 2 * M, h: 0.72,
      fontFace: HEAD, fontSize: 32, bold: true, color: INK, margin: 0,
    });
  }
  return s;
}
function fig(slide, file, opts) {
  slide.addImage({ path: path.join(FIG, file), ...opts });
}
// big number + label, used for stat rows
function stat(slide, x, y, w, value, label, colour) {
  slide.addText(value, {
    x, y, w, h: 0.85, fontFace: HEAD, fontSize: 40, bold: true,
    color: colour || NAVY, margin: 0,
  });
  slide.addText(label, {
    x, y: y + 0.86, w, h: 0.62, fontFace: BODY, fontSize: 12.5,
    color: MUTED, margin: 0,
  });
}
function card(slide, x, y, w, h) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06, fill: { color: PALE }, line: { color: "E4E9EE" },
  });
}

// =====================================================================  1
{
  const s = darkSlide();
  s.addText("Forecasting wheat yield\neight weeks before harvest", {
    x: M, y: 1.9, w: 9.6, h: 2.1, fontFace: HEAD, fontSize: 44, bold: true,
    color: WHITE, lineSpacing: 52, margin: 0,
  });
  s.addText(
    "119 districts across Haryana, Punjab and Uttar Pradesh.\n" +
    "A point forecast, a full probability distribution, and an early warning of collapse.",
    { x: M, y: 4.25, w: 9.6, h: 1.0, fontFace: BODY, fontSize: 16,
      color: "AFC0D4", lineSpacing: 26, margin: 0 });
  s.addText("Every figure in this deck is machine-verified — 96 checks, 96 pass", {
    x: M, y: 6.45, w: 9.6, h: 0.36, fontFace: BODY, fontSize: 12,
    color: ORANGE, bold: true, margin: 0,
  });
  s.addNotes("The deck describes a 5 March forecasting system. All numbers are recomputed from source artifacts by scripts/verify_walkthrough.py.");
}

// =====================================================================  2
{
  const s = lightSlide("Predict the harvest before it happens", "The task");
  stat(s, M, 2.05, 3.0, "5 March", "the forecast date — eight weeks before harvest");
  stat(s, M + 4.0, 2.05, 3.0, "119", "districts, averaging 3,400 km² each");
  stat(s, M + 8.0, 2.05, 3.4, "4,500", "kg per hectare — the typical yield");

  card(s, M, 4.35, W - 2 * M, 2.15);
  s.addText("Why it is hard", {
    x: M + 0.4, y: 4.6, w: 5, h: 0.35, fontFace: BODY, fontSize: 13,
    bold: true, color: INK, margin: 0,
  });
  s.addText([
    { text: "The crop is not finished — a late-March heatwave can still destroy 20% of it", options: { bullet: true, breakLine: true } },
    { text: "Districts are huge — half can fail while the other half thrives", options: { bullet: true, breakLine: true } },
    { text: "Ground truth arrives once a year, and neighbouring districts fail together", options: { bullet: true } },
  ], { x: M + 0.4, y: 5.0, w: W - 2 * M - 0.8, h: 1.3, fontFace: BODY,
       fontSize: 14.5, color: INK, paraSpaceAfter: 6, margin: 0 });
}

// =====================================================================  3
{
  const s = lightSlide("The sharpest data starts too late", "The data");
  fig(s, "01_data_timeline.png", { x: M, y: 1.75, w: 11.9, h: 4.76 });
  s.addText("Sentinel-2 begins in 2017, leaving four testable harvests. MODIS reaches back to 2000 — nineteen.", {
    x: M, y: 6.62, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13,
    color: MUTED, italic: true, margin: 0,
  });
}

// =====================================================================  4
{
  const s = lightSlide("Five stages, each a small correction", "The model");
  const rows = [
    ["0", "Baseline", "Weighted average of the last three harvests", "333"],
    ["1", "The Blend", "Five simple models, a disagreement gate, movement calibration", "273"],
    ["2", "Weather ahead", "What changes when the model may see the 10-day forecast", "272"],
    ["3", "Training data", "Train on more seasons  (small, uncertain)", "—"],
    ["4", "Crop vision", "A transformer reads satellite crop condition", "265"],
  ];
  let y = 1.95;
  rows.forEach(([n, name, what, err], i) => {
    const last = i === rows.length - 1;
    card(s, M, y, W - 2 * M, 0.82);
    s.addShape(pres.ShapeType.ellipse, {
      x: M + 0.22, y: y + 0.19, w: 0.44, h: 0.44,
      fill: { color: last ? ORANGE : NAVY },
    });
    s.addText(n, { x: M + 0.22, y: y + 0.19, w: 0.44, h: 0.44, align: "center",
      valign: "middle", fontFace: BODY, fontSize: 14, bold: true, color: WHITE, margin: 0 });
    s.addText(name, { x: M + 0.85, y: y + 0.13, w: 2.3, h: 0.56, valign: "middle",
      fontFace: BODY, fontSize: 15, bold: true, color: INK, margin: 0 });
    s.addText(what, { x: M + 3.2, y: y + 0.13, w: 7.0, h: 0.56, valign: "middle",
      fontFace: BODY, fontSize: 13.5, color: MUTED, margin: 0 });
    s.addText(err, { x: W - M - 1.5, y: y + 0.13, w: 1.3, h: 0.56, align: "right",
      valign: "middle", fontFace: HEAD, fontSize: 19, bold: true,
      color: last ? ORANGE : NAVY, margin: 0 });
    y += 0.94;
  });
  s.addText("typical error, kg/ha  →", {
    x: W - M - 3.2, y: 1.6, w: 3.0, h: 0.3, align: "right",
    fontFace: BODY, fontSize: 11, color: MUTED, margin: 0 });
  s.addText("Never let a new component predict the harvest alone. Let it nudge a conservative estimate.", {
    x: M, y: 6.72, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13,
    color: INK, italic: true, margin: 0 });
}

// =====================================================================  5
{
  const s = lightSlide("A network that never sees a harvest figure", "Stage 4 — crop vision");
  s.addText([
    { text: "The problem  ", options: { bold: true, color: INK } },
    { text: "Thousands of examples of how wheat develops. Only a handful of harvests.", options: { color: MUTED } },
  ], { x: M, y: 1.95, w: 6.0, h: 0.7, fontFace: BODY, fontSize: 14.5, margin: 0 });

  s.addText([
    { text: "The solution  ", options: { bold: true, color: INK } },
    { text: "Train it on the data-rich question — how will the crop change between January and March — then hand its internal summary to a simple model for the data-poor one.", options: { color: MUTED } },
  ], { x: M, y: 2.85, w: 6.0, h: 1.5, fontFace: BODY, fontSize: 14.5, margin: 0 });

  card(s, M, 4.6, 6.0, 1.9);
  s.addText("Attention, in one line", { x: M + 0.35, y: 4.8, w: 5.3, h: 0.3,
    fontFace: BODY, fontSize: 12, bold: true, color: INK, margin: 0 });
  s.addText("Each piece of input decides how much to listen to every other piece — then takes a weighted average.", {
    x: M + 0.35, y: 5.15, w: 5.3, h: 1.1, fontFace: BODY, fontSize: 13.5,
    color: MUTED, margin: 0 });

  const bx = 7.4;
  stat(s, bx, 2.0, 5.2, "67,359", "learned numbers — one layer of a small language model");
  stat(s, bx, 3.5, 5.2, "16", "values it hands to the yield model, down from 30", ORANGE);
  stat(s, bx, 5.0, 5.2, "±130", "kg/ha — how far it can move a forecast");
}

// =====================================================================  6
{
  const s = lightSlide("Removing inputs made it stronger", "Stage 4 — the key finding");
  card(s, M, 2.0, 5.5, 4.4);
  s.addText("Before", { x: M + 0.4, y: 2.25, w: 4.5, h: 0.35, fontFace: BODY,
    fontSize: 13, bold: true, color: MUTED, margin: 0 });
  s.addText("30", { x: M + 0.4, y: 2.65, w: 4.5, h: 1.0, fontFace: HEAD,
    fontSize: 52, bold: true, color: MUTED, margin: 0 });
  s.addText([
    { text: "6 raw satellite averages needing no network at all", options: { bullet: true, breakLine: true } },
    { text: "8 weak change summaries", options: { bullet: true, breakLine: true } },
    { text: "16 genuine network outputs", options: { bullet: true } },
  ], { x: M + 0.4, y: 3.85, w: 4.7, h: 1.9, fontFace: BODY, fontSize: 13,
       color: INK, paraSpaceAfter: 5, margin: 0 });

  card(s, M + 6.1, 2.0, 5.5, 4.4);
  s.addText("After", { x: M + 6.5, y: 2.25, w: 4.5, h: 0.35, fontFace: BODY,
    fontSize: 13, bold: true, color: ORANGE, margin: 0 });
  s.addText("16", { x: M + 6.5, y: 2.65, w: 4.5, h: 1.0, fontFace: HEAD,
    fontSize: 52, bold: true, color: ORANGE, margin: 0 });
  s.addText([
    { text: "The 6 raw averages correlated 0.024 with the model's errors — nothing", options: { bullet: true, breakLine: true } },
    { text: "They made accuracy worse", options: { bullet: true, breakLine: true } },
    { text: "The remaining 16 could then be trusted at nearly double the weight", options: { bullet: true } },
  ], { x: M + 6.5, y: 3.85, w: 4.7, h: 1.9, fontFace: BODY, fontSize: 13,
       color: INK, paraSpaceAfter: 5, margin: 0 });

  s.addText("Removing features let the remaining signal be believed harder.", {
    x: M, y: 6.65, w: 11.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true,
    color: INK, margin: 0 });
}

// =====================================================================  7
{
  const s = lightSlide("Rewari district, Haryana — 2022", "Worked example");
  fig(s, "02_worked_example.png", { x: 1.15, y: 1.72, w: 11.0, h: 5.05 });
}

// =====================================================================  8
{
  const s = lightSlide("The forecast is a distribution, not a number", "Worked example");
  fig(s, "07_probability_distribution.png", { x: 1.55, y: 1.72, w: 10.2, h: 5.3 });
}

// =====================================================================  9
{
  const s = lightSlide("Every threshold a planner asks about", "Worked example");
  fig(s, "08_event_probabilities.png", { x: 0.95, y: 1.9, w: 11.4, h: 4.47 });
  s.addText("Rewari fell 9.4%. The model called a decline 92% likely and put it right at the 10% boundary.", {
    x: M, y: 6.55, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5,
    color: INK, italic: true, margin: 0 });
}

// ===================================================================== 10
{
  const s = lightSlide("The probabilities are honest", "Does it work?");
  const rows = [
    ["When the model says…", "Stated", "Actually happened"],
    ["Fall worse than 5%", "29.9%", "29.2%"],
    ["Fall worse than 10%", "11.5%", "10.9%"],
    ["Fall worse than 20%", "1.2%", "2.3%"],
    ["Any increase", "45.9%", "43.5%"],
  ];
  let y = 2.1;
  rows.forEach((r, i) => {
    const head = i === 0;
    if (!head) card(s, M, y, 8.6, 0.72);
    s.addText(r[0], { x: M + (head ? 0 : 0.35), y, w: 4.6, h: 0.72, valign: "middle",
      fontFace: BODY, fontSize: head ? 12 : 15, bold: head,
      color: head ? MUTED : INK, margin: 0 });
    s.addText(r[1], { x: M + 4.7, y, w: 1.7, h: 0.72, align: "right", valign: "middle",
      fontFace: head ? BODY : HEAD, fontSize: head ? 12 : 17, bold: !head,
      color: head ? MUTED : MUTED, margin: 0 });
    s.addText(r[2], { x: M + 6.5, y, w: 1.75, h: 0.72, align: "right", valign: "middle",
      fontFace: head ? BODY : HEAD, fontSize: head ? 12 : 17, bold: !head,
      color: head ? MUTED : GREEN, margin: 0 });
    y += head ? 0.5 : 0.82;
  });
  s.addText("Across all 476 district-seasons", { x: M, y: 5.72, w: 8.6, h: 0.3,
    fontFace: BODY, fontSize: 11.5, color: MUTED, margin: 0 });

  card(s, 10.0, 2.1, 2.6, 3.3);
  s.addText("80%", { x: 10.0, y: 2.5, w: 2.6, h: 0.8, align: "center",
    fontFace: HEAD, fontSize: 34, bold: true, color: NAVY, margin: 0 });
  s.addText("ranges cover", { x: 10.0, y: 3.25, w: 2.6, h: 0.3, align: "center",
    fontFace: BODY, fontSize: 12, color: MUTED, margin: 0 });
  s.addText("80.9%", { x: 10.0, y: 3.62, w: 2.6, h: 0.8, align: "center",
    fontFace: HEAD, fontSize: 34, bold: true, color: GREEN, margin: 0 });
  s.addText("of the time", { x: 10.0, y: 4.38, w: 2.6, h: 0.3, align: "center",
    fontFace: BODY, fontSize: 12, color: MUTED, margin: 0 });
  s.addText("A 10% chance\nhappens 10% of the time.", {
    x: 10.0, y: 4.78, w: 2.6, h: 0.6, align: "center", fontFace: BODY,
    fontSize: 11.5, color: INK, italic: true, margin: 0 });

  s.addText("Slightly over-confident about catastrophes: 1.2% stated for falls worse than 20%, where 2.3% occur.", {
    x: M, y: 6.6, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 12.5,
    color: ORANGE, margin: 0 });
}

// ===================================================================== 11
{
  const s = lightSlide("Better in every season available", "Does it work?");
  fig(s, "04_per_season.png", { x: 0.75, y: 1.8, w: 7.9, h: 3.67 });
  const bx = 9.0;
  stat(s, bx, 1.95, 3.6, "265", "kg/ha typical error — 5.9% of a harvest", NAVY);
  stat(s, bx, 3.35, 3.6, "4 of 4", "seasons improved", GREEN);
  stat(s, bx, 4.75, 3.6, "78.8%", "up-or-down called correctly");
  s.addText("Never “100% confident” — with four seasons a resampling test reports certainty automatically whenever all four agree. Say “4 of 4” and let the reader judge.", {
    x: M, y: 5.95, w: 11.9, h: 0.85, fontFace: BODY, fontSize: 13,
    color: MUTED, italic: true, margin: 0 });
}

// ===================================================================== 12
{
  const s = lightSlide("What waiting until March buys you", "Forecast date");
  fig(s, "11_january_vs_march.png", { x: 1.35, y: 1.72, w: 10.6, h: 5.1 });
}

// ===================================================================== 13
{
  const s = lightSlide("The same district, seven weeks apart", "Forecast date");
  const rows = [
    ["", "15 January", "5 March"],
    ["Point forecast", "4,555", "4,162"],
    ["Error", "405", "12"],
    ["P(increase)", "43%", "8%"],
    ["P(fall over 10%)", "9%", "46%"],
  ];
  let y = 2.15;
  rows.forEach((r, i) => {
    const head = i === 0;
    if (!head) card(s, M, y, 7.9, 0.78);
    s.addText(r[0], { x: M + (head ? 0 : 0.35), y, w: 3.6, h: 0.78, valign: "middle",
      fontFace: BODY, fontSize: head ? 12 : 15, bold: head, color: head ? MUTED : INK, margin: 0 });
    s.addText(r[1], { x: M + 3.9, y, w: 1.8, h: 0.78, align: "right", valign: "middle",
      fontFace: head ? BODY : HEAD, fontSize: head ? 12 : 18, bold: true,
      color: head ? MUTED : MUTED, margin: 0 });
    s.addText(r[2], { x: M + 5.9, y, w: 1.8, h: 0.78, align: "right", valign: "middle",
      fontFace: head ? BODY : HEAD, fontSize: head ? 12 : 18, bold: true,
      color: head ? MUTED : ORANGE, margin: 0 });
    y += head ? 0.52 : 0.88;
  });
  card(s, 9.3, 2.15, 3.3, 3.3);
  s.addText("4,150", { x: 9.3, y: 2.75, w: 3.3, h: 0.85, align: "center",
    fontFace: HEAD, fontSize: 38, bold: true, color: INK, margin: 0 });
  s.addText("the actual harvest", { x: 9.3, y: 3.6, w: 3.3, h: 0.3, align: "center",
    fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0 });
  s.addText("January said a serious\ndecline was 9% likely.\nMarch said 46%.", {
    x: 9.3, y: 4.15, w: 3.3, h: 1.0, align: "center", fontFace: BODY,
    fontSize: 13, color: INK, margin: 0 });
  s.addText("The information arriving in February and early March is what turns a wrong answer into a right one.", {
    x: M, y: 6.35, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5,
    color: INK, italic: true, margin: 0 });
}

// ===================================================================== 14
{
  const s = lightSlide("Four seasons can point the wrong way", "The methodology");
  fig(s, "05_four_year_illusion.png", { x: 0.8, y: 1.85, w: 7.6, h: 3.71 });
  s.addText("A rebuilt network looked like a certain win on four seasons and a loss on nineteen. The sign flips.", {
    x: 0.8, y: 5.75, w: 7.6, h: 0.7, fontFace: BODY, fontSize: 13,
    color: MUTED, margin: 0 });

  card(s, 8.9, 1.95, 3.7, 4.3);
  s.addText("The noise floor", { x: 9.2, y: 2.2, w: 3.1, h: 0.35, fontFace: BODY,
    fontSize: 13, bold: true, color: INK, margin: 0 });
  s.addText("Re-run the model changing only the order the input columns are listed in. No information changes — but the score moves anyway.", {
    x: 9.2, y: 2.6, w: 3.1, h: 1.5, fontFace: BODY, fontSize: 12.5,
    color: MUTED, margin: 0 });
  s.addText("±0.6", { x: 9.2, y: 4.15, w: 3.1, h: 0.7, fontFace: HEAD,
    fontSize: 34, bold: true, color: ORANGE, margin: 0 });
  s.addText("kg/ha of pure noise", { x: 9.2, y: 4.85, w: 3.1, h: 0.3,
    fontFace: BODY, fontSize: 12, color: MUTED, margin: 0 });
  s.addText("The earlier model's headline result was 0.9 — inside its own noise floor.", {
    x: 9.2, y: 5.25, w: 3.1, h: 0.85, fontFace: BODY, fontSize: 12.5,
    color: INK, italic: true, margin: 0 });
}

// ===================================================================== 15
{
  const s = lightSlide("Most things did not work", "The methodology");
  const rows = [
    ["Rebuilt network, better architecture", "worse over 19 seasons"],
    ["Sub-district tiles into the existing model", "no effect"],
    ["Network output as a direct correction", "correlation 0.024 with real errors"],
    ["Network output in the uncertainty model", "coverage collapsed 79% → 60%"],
    ["Fine-tuning the network directly on yield", "378 kg/ha versus 265"],
    ["Collapsing all five stages into one model", "−77 kg/ha"],
  ];
  let y = 1.95;
  rows.forEach(([what, outcome]) => {
    card(s, M, y, 11.9, 0.66);
    s.addText(what, { x: M + 0.35, y, w: 6.6, h: 0.66, valign: "middle",
      fontFace: BODY, fontSize: 14, color: INK, margin: 0 });
    s.addText(outcome, { x: M + 7.1, y, w: 4.4, h: 0.66, valign: "middle",
      align: "right", fontFace: BODY, fontSize: 14, bold: true,
      color: ORANGE, margin: 0 });
    y += 0.76;
  });
  s.addText("Every intervention that supplied more or better data helped. Every cleverer network on the same data did not.", {
    x: M, y: 6.7, w: 11.9, h: 0.4, fontFace: HEAD, fontSize: 15.5, bold: true,
    color: INK, margin: 0 });
}

// ===================================================================== 16
{
  const s = darkSlide();
  s.addText("What it delivers", {
    x: M, y: 0.85, w: 11.9, h: 0.75, fontFace: HEAD, fontSize: 34, bold: true,
    color: WHITE, margin: 0 });
  const items = [
    ["265", "kg/ha typical error — 5.9% of a harvest"],
    ["80.9%", "of harvests land inside the 80% range"],
    ["0.82", "skill at ranking a >10% collapse"],
    ["3 hrs", "to run the whole pipeline on one laptop"],
  ];
  items.forEach(([v, l], i) => {
    const x = M + (i % 2) * 6.1;
    const y = 2.15 + Math.floor(i / 2) * 1.75;
    s.addText(v, { x, y, w: 5.6, h: 0.85, fontFace: HEAD, fontSize: 38,
      bold: true, color: i === 1 ? "7FD1A6" : "8FB4E3", margin: 0 });
    s.addText(l, { x, y: y + 0.87, w: 5.6, h: 0.55, fontFace: BODY,
      fontSize: 13.5, color: "AFC0D4", margin: 0 });
  });
  s.addText("The largest verified improvement came from removing fourteen useless inputs — not from adding anything.", {
    x: M, y: 6.15, w: 11.9, h: 0.5, fontFace: BODY, fontSize: 15,
    color: ORANGE, italic: true, margin: 0 });
  s.addNotes("Close on the methodology point: resolution and sample size were the binding constraints, not architecture.");
}

const out = path.join(ROOT, "wheat_yield_forecast.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote " + out));
