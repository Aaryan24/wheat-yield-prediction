#!/usr/bin/env python3
"""How the forecast changes as the season unfolds.

An earlier stage of this project issued forecasts at three dates -- 15 January,
15 February and 5 March -- for the same four harvests.  Those artifacts are used
here to answer a question the current 5 March system cannot: what is the value
of waiting?

Two honest observations fall out of the data and are drawn on the figure:

  * the January and February POINT forecasts are identical BY DESIGN -- that
    system locked one model for both early dates and a different, stronger one
    for March, which its own documentation states explicitly.  The probability
    layer did update between January and February; the point forecast did not.
  * the March forecast is clearly better, and the gain shows up in every
    measure at once -- accuracy, direction, and the ability to flag a collapse.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
RAPID = V16.parent
FIG = V16 / "figures"
FIG.mkdir(exist_ok=True)
SOURCE = RAPID / "v13_crop_response_final" / "artifacts" / "final_predictions.parquet"

INK, COOL, ACCENT, MUTED, GREEN = "#1b2430", "#1d4e89", "#c2410c", "#94a3b8", "#15803d"
CLOCKS = ["jan15", "feb15", "mar05"]
LABEL = {"jan15": "15 January", "feb15": "15 February", "mar05": "5 March"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def main() -> None:
    d = pd.read_parquet(SOURCE)
    # last season's yield is needed for the direction test; it lives in the
    # current model's output, not in this older artifact
    reference = pd.read_parquet(
        RAPID / "v15_complete_hierarchy" / "artifacts" / "final_predictions.parquet"
    )[["district_id", "season_start_year", "lag_1_yield"]]
    d = d.merge(reference, on=["district_id", "season_start_year"])

    def auc(probability, outcome):
        order = np.argsort(probability)
        ranks = np.empty(len(probability)); ranks[order] = np.arange(1, len(probability) + 1)
        P, N = outcome.sum(), (1 - outcome).sum()
        return (ranks[outcome == 1].sum() - P * (P + 1) / 2) / (P * N)

    stats = []
    for clock in CLOCKS:
        s = d[d.clock.eq(clock)]
        stats.append({
            "clock": clock,
            "error": float(np.sqrt(np.mean((s.prediction - s.actual) ** 2))),
            # Direction is judged the SAME way as everywhere else in this
            # project: does the point forecast sit on the correct side of last
            # season?  An earlier version of this figure used "is the stated
            # probability above 0.5", which is a different question and gave
            # numbers 3 points lower that did not reconcile with any other table.
            "direction": float(np.mean(
                (s.actual > s.lag_1_yield) == (s.prediction > s.lag_1_yield))),
            "auc_drop": float(auc(s.severe_probability.values,
                                  s.severe_target.values.astype(float)))})
    stats = pd.DataFrame(stats)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    x = np.arange(3)
    colours = [MUTED, MUTED, COOL]

    ax = axes[0]
    ax.bar(x, stats.error, color=colours, width=0.62)
    for i, v in enumerate(stats.error):
        ax.text(i, v + 4, f"{v:.0f}", ha="center", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CLOCKS], fontsize=9)
    ax.set_ylabel("typical error (kg/ha)")
    ax.set_ylim(0, 360)
    ax.set_title("Accuracy", fontsize=11.5, fontweight="bold", loc="left")
    ax.annotate("same model\n(by design)", xy=(0.5, 322), fontsize=8.5,
                ha="center", color=MUTED, fontweight="bold")
    ax.plot([0, 1], [312, 312], color=ACCENT, linewidth=1.4)

    ax = axes[1]
    ax.bar(x, stats.direction, color=colours, width=0.62)
    for i, v in enumerate(stats.direction):
        ax.text(i, v + 0.012, f"{v:.0%}", ha="center", fontsize=11.5,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CLOCKS], fontsize=9)
    ax.set_ylim(0, 0.92); ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_yticklabels(["0%", "25%", "50%", "75%"])
    ax.set_ylabel("up-or-down called correctly")
    ax.set_title("Direction", fontsize=11.5, fontweight="bold", loc="left")

    ax = axes[2]
    ax.bar(x, stats.auc_drop, color=colours, width=0.62)
    for i, v in enumerate(stats.auc_drop):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=11.5,
                fontweight="bold")
    ax.axhline(0.5, color=ACCENT, linestyle=":", linewidth=1.5)
    ax.text(2.45, 0.515, "no skill", fontsize=8.5, color=ACCENT, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in CLOCKS], fontsize=9)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("skill at flagging a >10% collapse")
    ax.set_title("Early warning", fontsize=11.5, fontweight="bold", loc="left")

    fig.suptitle("What waiting until March buys you  —  same four harvests, "
                 "three forecast dates",
                 fontsize=13, fontweight="bold", x=0.007, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "09_forecast_dates.png", dpi=170)
    plt.close(fig)

    # --- the Rewari story: the March information changes the answer ---
    ex = d[(d.district_id == "IND013018") & (d.season_start_year == 2022)]
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    positions = np.arange(3)
    for i, clock in enumerate(CLOCKS):
        s = ex[ex.clock.eq(clock)].iloc[0]
        ax.plot([i, i], [s.lo80_integrated, s.hi80_integrated], color=COOL,
                linewidth=9, alpha=0.30, solid_capstyle="round")
        ax.plot(i, s.prediction, "o", color=COOL, markersize=11, zorder=3)
        ax.text(i, s.prediction + 60, f"{s.prediction:,.0f}", ha="center",
                fontsize=10.5, fontweight="bold", color=COOL)
        # probabilities sit inside the panel, below each interval, so they
        # never collide with the title
        ax.text(i, s.lo80_integrated - 130,
                f"P(rise) {s.increase_probability:.0%}\n"
                f"P(collapse) {s.severe_probability:.0%}",
                ha="center", va="top", fontsize=10,
                color=ACCENT if s.severe_probability > 0.3 else MUTED,
                fontweight="bold")
    actual = float(ex.actual.iloc[0])
    ax.axhline(actual, color=ACCENT, linestyle="--", linewidth=2.2)
    ax.text(2.42, actual - 55, f"actual harvest {actual:,.0f}", color=ACCENT,
            fontsize=10.5, fontweight="bold", ha="right")
    ax.axhline(float(ex.anchor.iloc[0]) if "anchor" in ex else 4580,
               color=MUTED, linestyle=":", linewidth=1.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABEL[c] for c in CLOCKS], fontsize=10.5)
    ax.set_ylabel("wheat yield (kg/ha)")
    ax.set_xlim(-0.45, 2.55)
    ax.set_ylim(3550, 5320)
    ax.set_title("Rewari, Haryana 2022 — the March data changes the answer\n"
                 "In January the model expected a good year; by March it saw the "
                 "crop failing",
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "10_rewari_across_dates.png", dpi=170)
    plt.close(fig)
    print("wrote 09_forecast_dates.png and 10_rewari_across_dates.png")


if __name__ == "__main__":
    main()
