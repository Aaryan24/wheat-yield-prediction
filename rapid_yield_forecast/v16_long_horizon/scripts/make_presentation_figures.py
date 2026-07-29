#!/usr/bin/env python3
"""Figures for the model walkthrough document."""
from __future__ import annotations

import sys
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
V15A = RAPID / "v15_complete_hierarchy" / "artifacts"
T = "yield_kg_per_ha"

INK = "#1b2430"
ACCENT = "#c2410c"
COOL = "#1d4e89"
MUTED = "#94a3b8"
GOOD = "#15803d"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def fig_data_timeline() -> None:
    rows = [("Yield records (DES / ICRISAT)", 1990, 2022, COOL, "3,658 district-seasons"),
            ("MODIS satellite, district", 2000, 2022, COOL, "2,737"),
            ("MODIS satellite, 9 km tiles", 2000, 2022, GOOD, "169,809 tiles"),
            ("Weather (NASA POWER, daily)", 2010, 2023, COOL, "546,805 days"),
            ("Sentinel-2 crop state", 2017, 2022, ACCENT, "2,142  <- the bottleneck"),
            ("Evaluation window", 2019, 2022, ACCENT, "476 rows / 4 seasons")]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for i, (name, start, end, colour, note) in enumerate(rows):
        y = len(rows) - i
        ax.barh(y, end - start, left=start, height=0.55, color=colour, alpha=0.85)
        ax.text(start + 0.3, y, name, va="center", ha="left", fontsize=9.5,
                color="white", fontweight="bold")
        ax.text(end + 0.4, y, note, va="center", ha="left", fontsize=9, color=INK)
    ax.set_xlim(1988, 2040)
    ax.set_ylim(0.3, len(rows) + 0.8)
    ax.set_yticks([])
    ax.set_xticks(range(1990, 2025, 5))
    ax.set_xlabel("season")
    ax.set_title("What data exists, and when it starts\n"
                 "Sentinel (2017) forced a 4-season evaluation; MODIS (2000) allows 19",
                 fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "01_data_timeline.png", dpi=170)
    plt.close(fig)


def fig_worked_example() -> None:
    anchor, window, crop = 4297.01, -39.0947, -55.8428
    w_gamma, c_gamma = 0.25, 2.25
    steps = [("Stage 1 — the Blend\n(history + weather\n+ satellite + transfer)", 4274.66),
             ("Stage 2\n+ weather-ahead\ncorrection", anchor),
             (f"Stage 3\n+ training data\n{w_gamma} x ({window:.1f})",
              anchor + w_gamma * window),
             (f"Stage 4\n+ crop vision\n{c_gamma} x ({crop:.1f})",
              anchor + w_gamma * window + c_gamma * crop)]
    actual = 4150.0
    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs = np.arange(len(steps))
    values = [v for _, v in steps]
    ax.plot(xs, values, "-o", color=COOL, linewidth=2.4, markersize=9, zorder=3)
    for x, (label, value) in zip(xs, steps):
        ax.annotate(f"{value:,.0f}", (x, value), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=10.5,
                    fontweight="bold", color=COOL)
    ax.axhline(actual, color=ACCENT, linestyle="--", linewidth=2)
    ax.text(len(steps) - 0.55, actual - 32, f"actual harvest  {actual:,.0f}",
            color=ACCENT, fontsize=10.5, fontweight="bold", ha="right")
    ax.axhline(4580, color=MUTED, linestyle=":", linewidth=1.6)
    ax.text(0.02, 4588, "last season 4,580", color=MUTED, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([s for s, _ in steps], fontsize=8.5)
    ax.set_ylabel("predicted yield (kg/ha)")
    ax.set_title("Rewari district, Haryana, 2022 harvest\n"
                 "Each stage moves the estimate; the crop network does the work",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.set_ylim(4080, 4680)
    fig.tight_layout()
    fig.savefig(FIG / "02_worked_example.png", dpi=170)
    plt.close(fig)


def fig_distribution() -> None:
    q = {5: 3707, 10: 3822, 25: 3986, 50: 4162, 75: 4352, 90: 4529, 95: 4674}
    actual, last = 4150.0, 4580.0
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    bands = [(5, 95, 0.16, "90% range"), (10, 90, 0.26, "80% range"),
             (25, 75, 0.42, "50% range")]
    for lo, hi, alpha, label in bands:
        ax.fill_betweenx([0, 1], q[lo], q[hi], color=COOL, alpha=alpha,
                         label=f"{label}  {q[lo]:,}-{q[hi]:,}")
    ax.axvline(q[50], color=COOL, linewidth=2.5)
    ax.text(q[50], 1.06, f"point forecast {q[50]:,}", ha="center",
            fontsize=10, fontweight="bold", color=COOL)
    ax.axvline(actual, color=ACCENT, linewidth=2.5, linestyle="--")
    ax.text(actual, -0.14, f"actual {actual:,.0f}", ha="center",
            fontsize=10, fontweight="bold", color=ACCENT)
    ax.axvline(last, color=MUTED, linewidth=1.8, linestyle=":")
    ax.text(last, -0.14, f"last season {last:,.0f}", ha="center",
            fontsize=9.5, color=MUTED)
    ax.set_xlim(3550, 4820)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("yield (kg/ha)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("The forecast is a distribution, not a number\n"
                 "Rewari 2022: P(yield falls vs last season) = 96%",
                 fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "03_distribution.png", dpi=170)
    plt.close(fig)


def fig_per_season() -> None:
    f = pd.read_parquet(V16 / "artifacts" / "final_predictions.parquet")
    years, v15s, fins = [], [], []
    for year, block in f.groupby("season_start_year"):
        truth = block[T].to_numpy(float)
        years.append(int(year))
        v15s.append(float(np.sqrt(np.mean(
            (block.v15_point_prediction.to_numpy(float) - truth) ** 2))))
        fins.append(float(np.sqrt(np.mean(
            (block.final_point.to_numpy(float) - truth) ** 2))))
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar(x - 0.2, v15s, 0.4, label="earlier model", color=MUTED)
    ax.bar(x + 0.2, fins, 0.4, label="this model", color=COOL)
    for i, (a, b) in enumerate(zip(v15s, fins)):
        ax.text(i + 0.2, b + 4, f"-{a-b:.1f}", ha="center", fontsize=9,
                color=GOOD, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("typical error, kg/ha (lower is better)")
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("Improvement in every one of the four available seasons\n"
                 "Gains are small: 1-7 kg/ha on a ~4,500 kg/ha harvest",
                 fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "04_per_season.png", dpi=170)
    plt.close(fig)


def fig_four_year_illusion() -> None:
    labels = ["earlier crop\ntransformer", "rebuilt\ntransformer",
              "sub-district\ntiles"]
    four = [0.93, 2.68, 1.35]
    nineteen = [np.nan, -2.19, 4.78]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x - 0.2, four, 0.4, label="measured on 4 seasons", color=ACCENT, alpha=0.85)
    ax.bar(x + 0.2, nineteen, 0.4, label="measured on 19 seasons", color=COOL)
    ax.axhline(0, color=INK, linewidth=1.1)
    ax.annotate("sign flips", xy=(1.2, -2.19), xytext=(1.75, -1.4),
                fontsize=10.5, color=ACCENT, fontweight="bold", ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("measured gain, kg/ha")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("Why four seasons is not enough\n"
                 "The same rebuilt transformer looks like a certain win on 4 "
                 "seasons and a loss on 19",
                 fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "05_four_year_illusion.png", dpi=170)
    plt.close(fig)


def fig_variance_split() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.barh([1], [354], color=ACCENT, height=0.5)
    ax.barh([0], [289], color=COOL, height=0.5)
    ax.text(360, 1, "354 kg/ha", va="center", fontsize=11, fontweight="bold",
            color=ACCENT)
    ax.text(295, 0, "289 kg/ha", va="center", fontsize=11, fontweight="bold",
            color=COOL)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['"What kind of season\nis this?" (shared)',
                        '"Which district beats\nits neighbours?"'], fontsize=9.5)
    ax.set_xlim(0, 470)
    ax.set_xlabel("size of the swing it causes (standard deviation, kg/ha)")
    ax.set_title("The bigger half of the problem was never modelled\n"
                 "The transformer works entirely on the smaller half",
                 fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "06_variance_split.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_data_timeline()
    fig_worked_example()
    fig_distribution()
    fig_per_season()
    fig_four_year_illusion()
    fig_variance_split()
    print(f"wrote {len(list(FIG.glob('*.png')))} figures to {FIG}")
