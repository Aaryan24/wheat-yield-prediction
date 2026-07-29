#!/usr/bin/env python3
"""Architecture diagrams: the full system, and the crop-vision network.

Box heights are computed from the line count rather than guessed, so text can
never spill past its container.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)

INK, NAVY, ORANGE, MUTED = "#1b2430", "#1d4e89", "#c2410c", "#94a3b8"
PALE, EDGE = "#f2f5f8", "#cfd8e3"
plt.rcParams.update({"font.size": 9, "text.color": INK, "figure.facecolor": "white"})

TITLE_PT, BODY_PT, SPACING = 10.0, 8.3, 1.5


class Canvas:
    """Axis-unit helper that knows how tall a line of text actually is."""

    def __init__(self, width: float, height: float):
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.pt = 1.0 / (height * 72.0)          # one typographic point, in axis units

    def height_for(self, lines: list[str]) -> float:
        head = TITLE_PT * 2.7 * self.pt
        body = len(lines) * BODY_PT * SPACING * self.pt
        return head + body + 14 * self.pt

    def box(self, x, y, w, lines, title, fill=PALE, edge=EDGE, tc=INK):
        h = self.height_for(lines)
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018",
            facecolor=fill, edgecolor=edge, linewidth=1.2))
        self.ax.text(x + 0.028, y + h - 9 * self.pt, title, fontsize=TITLE_PT,
                     fontweight="bold", color=tc, va="top", ha="left")
        if lines:
            self.ax.text(x + 0.028, y + h - TITLE_PT * 2.7 * self.pt,
                         "\n".join(lines), fontsize=BODY_PT, color=tc,
                         va="top", ha="left", linespacing=SPACING)
        return h

    def arrow(self, x1, y1, x2, y2, colour=NAVY, rad=0.0, lw=1.6):
        self.ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
            color=colour, linewidth=lw, connectionstyle=f"arc3,rad={rad}"))

    def title(self, text):
        self.ax.text(0.008, 0.982, text, fontsize=13.5, fontweight="bold",
                     color=INK, va="top")

    def save(self, name):
        self.fig.tight_layout(pad=0.3)
        self.fig.savefig(FIG / name, dpi=175)
        plt.close(self.fig)


# ---------------------------------------------------------------- system
def system_diagram() -> None:
    c = Canvas(13.2, 7.0)
    c.title("System architecture — the 5 March forecast")

    top = 0.575
    c.box(0.008, top, 0.175, [
        "Yield history     1990–2022",
        "MODIS satellite   2000–2022",
        "Sentinel-2        2017–2022",
        "Daily weather     2010–2023",
        "10-day forecast, MSP price"],
        "INPUT DATA", fill="#eef2f7", edge="#c3d0e0")

    c.box(0.008, 0.285, 0.175, [
        "78 numeric inputs",
        "   yield history       18",
        "   observed weather    16",
        "   forecast weather    10",
        "   stress indices      31",
        "   economics            3",
        "+ 3 state indicators"],
        "FEATURE PANEL")

    c.box(0.008, 0.045, 0.175, [
        "b = 0.60·y(t−1)",
        "      + 0.25·y(t−2)",
        "      + 0.15·y(t−3)",
        "",
        "models predict r = y − b"],
        "BASELINE  b", fill="#fdf1ea", edge="#e8c3ac")

    xs = [0.208, 0.404, 0.600, 0.796]
    c.box(xs[0], top, 0.180, [
        "H  weighted history   0.50",
        "R  weather+satellite  0.15",
        "P  physics            0.00",
        "E  economic           0.15",
        "C  global transfer    0.20",
        "",
        "+ disagreement gate",
        "+ movement scale ×1.50"],
        "STAGE 1 — Blend")

    c.box(xs[1], top, 0.180, [
        "four matched models,",
        "differing only in whether",
        "the forecast is visible",
        "",
        "NO-FUTURE = control",
        "FULL / EFFECT / BROAD",
        "",
        "c = mean(3) − control ×1.75"],
        "STAGE 2 — Weather ahead")

    c.box(xs[2], top, 0.180, [
        "same model, two spans",
        "of training seasons",
        "",
        "from 2013  →  952 rows",
        "from 2017  →  238 rows",
        "",
        "",
        "c = wide − narrow  ×0.25"],
        "STAGE 3 — Training data")

    c.box(xs[3], top, 0.190, [
        "transformer turns satellite",
        "crop state into 16 values",
        "",
        "two matched models:",
        "with them, and without",
        "",
        "",
        "c = with − without ×2.25"],
        "STAGE 4 — Crop vision", fill="#fdf1ea", edge="#e8c3ac")

    mid = top + 0.16
    c.arrow(0.185, mid, 0.206, mid)
    for i in range(3):
        c.arrow(xs[i] + 0.182, mid, xs[i + 1] - 0.002, mid)

    c.box(0.208, 0.285, 0.376, [
        "ŷ = Blend + 1.75·c(ahead)",
        "          + 0.25·c(data) + 2.25·c(crop)",
        "",
        "each c is a matched-model difference —",
        "two identical models, one extra input"],
        "POINT FORECAST", fill="#eaf1fb", edge="#bcd0ea")

    c.box(0.600, 0.285, 0.386, [
        "empirical distribution of the model's",
        "own past errors, scaled by district",
        "volatility, seasons weighted equally,",
        "recentred on ŷ",
        "→ 19 quantiles  q05 … q95"],
        "STAGE 5 — Uncertainty", fill="#eaf1fb", edge="#bcd0ea")

    c.arrow(0.40, top - 0.004, 0.40, 0.478)
    c.arrow(0.586, 0.38, 0.598, 0.38)

    c.box(0.208, 0.045, 0.778, [
        "point forecast (kg/ha)   ·   19 quantiles q05 … q95   ·   P(yield rises)",
        "P(fall > 5%)   ·   P(fall > 10%)   ·   P(fall > 20%)   ·   P(fall > 30%)",
        "",
        "every probability read off the interpolated cumulative distribution"],
        "OUTPUT — per district, per season", fill="#1b2430", edge="#1b2430",
        tc="white")
    c.arrow(0.79, 0.283, 0.79, 0.215, colour=ORANGE)
    c.save("A1_system_architecture.png")


# ---------------------------------------------------------------- network
def network_diagram() -> None:
    c = Canvas(13.2, 7.0)
    c.title("Crop-vision network — 67,359 parameters, width 32, 4 attention heads")

    r1, r2, r3 = 0.655, 0.365, 0.075
    c.box(0.008, r1, 0.175, [
        "6 vegetation indices ×",
        "3 spatial views ×",
        "7 time summaries",
        "",
        "later clocks masked"],
        "Crop state  3 × 126", fill="#eaf1fb", edge="#bcd0ea")
    c.box(0.008, r2, 0.175, [
        "six windows before",
        "the forecast clock",
        "",
        "temperature, rain, solar,",
        "humidity, soil moisture"],
        "Past weather  6 × 16", fill="#eef2f7", edge="#c3d0e0")
    c.box(0.008, r3, 0.175, [
        "ten 5-day windows",
        "ahead of the clock",
        "",
        "issued ≥ 2 days before,",
        "so nothing leaks"],
        "Forecast weather  10 × 16", fill="#eef2f7", edge="#c3d0e0")

    for cy, lbl in ((r1 + 0.095, "32 × 126"), (r2 + 0.095, "32 × 16"),
                    (r3 + 0.095, "32 × 16")):
        c.ax.add_patch(FancyBboxPatch(
            (0.203, cy - 0.042), 0.070, 0.084,
            boxstyle="round,pad=0,rounding_size=0.016",
            facecolor="white", edgecolor=NAVY, linewidth=1.3))
        c.ax.text(0.238, cy + 0.012, "linear", fontsize=8.5, ha="center",
                  color=NAVY, fontweight="bold")
        c.ax.text(0.238, cy - 0.026, lbl, fontsize=7.5, ha="center", color=MUTED)
        c.arrow(0.185, cy, 0.201, cy)
        c.arrow(0.275, cy, 0.297, cy)

    c.box(0.299, r1, 0.185, [
        "self-attention over the",
        "3 time tokens",
        "",
        "4 heads · head width 8",
        "FFN 32→64→32 · GELU",
        "pre-LayerNorm · drop .08"],
        "Crop encoder  ×2 layers")
    c.box(0.299, r2, 0.185, [
        "self-attention over",
        "the 6 windows",
        "",
        "same block design"],
        "Past-weather encoder ×1")
    c.box(0.299, r3, 0.185, [
        "self-attention over",
        "the 10 windows",
        "",
        "same block design"],
        "Forecast encoder ×1")

    c.box(0.500, r1, 0.185, [
        "the crop token for the",
        "current forecast clock",
        "",
        "on 5 March it has already",
        "attended to Jan and Feb"],
        "Query  q", fill="#fdf1ea", edge="#e8c3ac")
    c.box(0.500, r2, 0.185, [
        "Q = q ,  K = V = past",
        "",
        "“which weather explains",
        " the state I am in?”"],
        "Cross-attention → past", fill="white", edge=NAVY)
    c.box(0.500, r3, 0.185, [
        "Q = q ,  K = V = forecast",
        "",
        "zeroed in the",
        "no-forecast branch"],
        "Cross-attn → forecast", fill="white", edge=NAVY)

    c.arrow(0.486, r1 + 0.095, 0.498, r1 + 0.095)
    c.arrow(0.486, r2 + 0.085, 0.498, r2 + 0.085)
    c.arrow(0.486, r3 + 0.085, 0.498, r3 + 0.085)
    c.arrow(0.5925, r1 - 0.004, 0.5925, r2 + 0.190)
    c.arrow(0.5925, r2 - 0.004, 0.5925, r3 + 0.190)

    c.box(0.702, r1, 0.288, [
        "h = LayerNorm( q + c_past + c_fcst )",
        "",
        "one 32-dimensional vector per",
        "district and season — 16 kept"],
        "Fusion", fill="#eaf1fb", edge="#bcd0ea")
    c.box(0.702, r2, 0.288, [
        "MODIS next token    32→48→35",
        "Crop change         32→64→126",
        "Change sign         32→126",
        "",
        "Loss = SmoothL1(β 0.5) + 0.08·BCE",
        "",
        "discarded after training — no yield",
        "label ever fits this network"],
        "Training heads", fill="white", edge="#e8c3ac")
    c.box(0.702, r3, 0.288, [
        "16 values per district-season,",
        "handed to two matched",
        "decision-tree models"],
        "Output → the yield model", fill="#1b2430", edge="#1b2430", tc="white")

    c.arrow(0.688, r1 + 0.095, 0.700, r1 + 0.095)
    # both cross-attention outputs enter Fusion at its LEFT edge, so the
    # connectors never cross the training-heads box sitting directly below it
    c.arrow(0.688, r2 + 0.200, 0.699, r1 + 0.105, rad=0.16)
    c.arrow(0.688, r3 + 0.200, 0.699, r1 + 0.055, rad=0.24)
    c.arrow(0.846, r1 - 0.004, 0.846, r2 + 0.247, colour=ORANGE)
    c.arrow(0.846, r2 - 0.004, 0.846, r3 + 0.148, colour=ORANGE)
    c.save("A2_network_architecture.png")


if __name__ == "__main__":
    system_diagram()
    network_diagram()
    print("wrote A1_system_architecture.png and A2_network_architecture.png")
