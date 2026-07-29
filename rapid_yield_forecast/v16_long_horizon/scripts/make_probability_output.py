#!/usr/bin/env python3
"""Turn the 19 forecast quantiles into the outputs a decision-maker asks for.

The model predicts 19 quantiles of yield.  That is the complete forecast, but
nobody makes a decision from a list of quantiles.  This script converts them
into the two things people actually want:

  * a probability DENSITY curve -- yield on the x axis, how likely on the y
    axis -- which is the picture everyone has in their head when they say
    "what might the harvest be?";
  * a table of event probabilities: chance of any increase, and chance of a
    fall past 5 / 10 / 20 / 30 percent.

Method.  The quantiles ARE the cumulative distribution: q05 is the yield with a
5% chance of being undershot, and so on.  So the CDF is known at 19 points.
Density is its derivative, estimated by finite differences between adjacent
quantiles and then smoothed, since 19 points give a ragged derivative.

    F(q_a) = a           for a in {0.05, 0.10, ..., 0.95}
    f(y)   = dF/dy   ~=  (a_{k+1} - a_k) / (q_{k+1} - q_k)

Event probabilities come from interpolating F and reading it off:

    P(rise)            = 1 - F(y_last)
    P(fall beyond p%)  =     F((1 - p) * y_last)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
ARTIFACTS = V16 / "artifacts"
FIG = V16 / "figures"
FIG.mkdir(exist_ok=True)
TARGET = "yield_kg_per_ha"
LEVELS = np.array([round(0.05 * i, 2) for i in range(1, 20)])
QCOLUMNS = [f"final_q{int(round(a * 100)):02d}" for a in LEVELS]
DECLINES = [0.05, 0.10, 0.20, 0.30]
INCREASES = [0.05, 0.10, 0.20]
# Density is naturally "probability per kg/ha", which is a number like 0.0012 --
# true but unreadable.  Multiplying by BAND turns it into "probability of
# landing in a 100 kg/ha window", which is directly interpretable: 0.12 means a
# 12% chance the harvest lands within +-50 kg/ha of that point.
BAND = 100.0

INK, COOL, ACCENT, MUTED = "#1b2430", "#1d4e89", "#c2410c", "#94a3b8"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def cdf_interpolator(quantiles: np.ndarray):
    """F(y): probability the harvest comes in at or below y."""
    xs = np.concatenate([[quantiles[0] - 400.0], quantiles,
                         [quantiles[-1] + 400.0]])
    ps = np.concatenate([[0.0], LEVELS, [1.0]])
    order = np.argsort(xs)
    return lambda y: float(np.interp(y, xs[order], ps[order]))


def density(quantiles: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Smoothed derivative of the CDF, evaluated on a regular grid."""
    widths = np.diff(quantiles)
    widths = np.where(widths > 1e-6, widths, 1e-6)
    heights = np.diff(LEVELS) / widths
    midpoints = (quantiles[:-1] + quantiles[1:]) / 2.0
    raw = np.interp(grid, midpoints, heights, left=0.0, right=0.0)
    # Gaussian smoothing: 19 quantiles give a ragged derivative
    step = grid[1] - grid[0]
    sigma = max((quantiles[-1] - quantiles[0]) / 25.0, step)
    half = int(3 * sigma / step)
    offsets = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(raw, kernel, mode="same")
    area = np.trapezoid(smoothed, grid) if hasattr(np, "trapezoid") \
        else np.trapz(smoothed, grid)
    return smoothed / area if area > 0 else smoothed


def event_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        quantiles = row[QCOLUMNS].to_numpy(float)
        F = cdf_interpolator(quantiles)
        last = float(row.lag_1_yield)
        record = {"district_id": row.district_id,
                  "district": row.get("district_name", ""),
                  "state": row.state_name,
                  "season": int(row.season_start_year),
                  "last_season": last,
                  "point_forecast": float(row.final_point),
                  "p_increase": 1.0 - F(last)}
        for decline in DECLINES:
            record[f"p_fall_over_{int(decline * 100)}pct"] = F((1 - decline) * last)
        for rise in INCREASES:
            record[f"p_rise_over_{int(rise * 100)}pct"] = 1.0 - F((1 + rise) * last)
        record["q10"] = float(row[f"final_q10"])
        record["q90"] = float(row[f"final_q90"])
        record["actual"] = float(row[TARGET]) if np.isfinite(row[TARGET]) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def plot_district(row: pd.Series, path: Path) -> None:
    quantiles = row[QCOLUMNS].to_numpy(float)
    last = float(row.lag_1_yield)
    actual = float(row[TARGET])
    F = cdf_interpolator(quantiles)
    grid = np.linspace(quantiles[0] - 250, quantiles[-1] + 250, 900)
    pdf = density(quantiles, grid)

    pdf = pdf * BAND        # probability per 100 kg/ha band, so the axis reads

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(grid, pdf, color=COOL, linewidth=2.2)
    ax.fill_between(grid, pdf, color=COOL, alpha=0.13)

    # shade the region below a 10% fall -- the food-security tail
    threshold = 0.90 * last
    mask = grid <= threshold
    ax.fill_between(grid[mask], pdf[mask], color=ACCENT, alpha=0.42,
                    label=f"fall worse than 10%   P = {F(threshold):.0%}")

    # forecast and actual can sit a few kg/ha apart, so labels are placed above
    # the curve with leader lines rather than written on the verticals
    top = ax.get_ylim()[1]
    markers = sorted([(float(row.final_point), COOL, "-",
                       f"forecast\n{row.final_point:,.0f}"),
                      (last, MUTED, ":", f"last season\n{last:,.0f}"),
                      (actual, ACCENT, "--", f"actual\n{actual:,.0f}")],
                     key=lambda m: m[0])
    span = grid[-1] - grid[0]
    offsets, previous = [], -np.inf
    for value, *_ in markers:
        anchor = value if value - previous > 0.10 * span else previous + 0.10 * span
        offsets.append(anchor)
        previous = anchor
    for (value, colour, style, label), anchor in zip(markers, offsets):
        ax.axvline(value, color=colour, linestyle=style, linewidth=2.1)
        ax.annotate(label, xy=(value, top * 0.80), xytext=(anchor, top * 1.02),
                    ha="center", fontsize=9.5, color=colour, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=1.0,
                                    shrinkA=0, shrinkB=2))
    ax.set_ylim(0, top * 1.22)

    ax.set_xlabel("wheat yield (kg/ha)")
    ax.set_ylabel(f"chance of landing within ±{BAND/2:.0f} kg/ha of this point")
    ticks = ax.get_yticks()
    ticks = ticks[(ticks >= 0) & (ticks <= ax.get_ylim()[1])]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.0%}" for t in ticks])
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    name = row.get("district_name", row.district_id)
    ax.set_title(f"{name}, {row.state_name} — {int(row.season_start_year)} harvest\n"
                 "The complete forecast: every outcome and how likely it is",
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_event_panel(record: pd.Series, path: Path) -> None:
    """Symmetric view: how bad could it get, and how good could it get."""
    GREEN = "#15803d"
    falls = [(f"fall\n>{int(d*100)}%", record[f"p_fall_over_{int(d*100)}pct"])
             for d in reversed(DECLINES)]
    rises = [(f"rise\n>{int(r*100)}%", record[f"p_rise_over_{int(r*100)}pct"])
             for r in INCREASES]
    entries = falls + [("any\nincrease", record.p_increase)] + rises
    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colours = [ACCENT] * len(falls) + [GREEN] * (1 + len(rises))

    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    bars = ax.bar(labels, values, color=colours, alpha=0.88, width=0.66)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.028,
                f"{value:.0%}", ha="center", fontsize=11, fontweight="bold")
    # divider between the downside and upside halves
    ax.axvline(len(falls) - 0.5, color=MUTED, linewidth=1.2, linestyle=":")
    ax.text(len(falls) / 2 - 0.5, 1.05, "WORSE than last season",
            ha="center", fontsize=9.5, color=ACCENT, fontweight="bold")
    ax.text(len(falls) + len(rises) / 2, 1.05, "BETTER than last season",
            ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("probability")
    ax.set_title(f"{record['district']}, {record['state']} — "
                 f"{record['season']} harvest outlook\n"
                 f"every threshold, relative to last season's "
                 f"{record.last_season:,.0f} kg/ha",
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    frame = pd.read_parquet(ARTIFACTS / "final_predictions.parquet")
    if "district_name" not in frame.columns:
        source = pd.read_parquet(
            V16.parent / "v15_complete_hierarchy" / "artifacts"
            / "final_predictions.parquet")[["district_id", "district_name"]]
        frame = frame.merge(source.drop_duplicates(), on="district_id", how="left")

    table = event_table(frame)
    table.to_csv(ARTIFACTS / "forecast_probabilities.csv", index=False)
    print(f"wrote probabilities for {len(table)} district-seasons")

    target = frame[(frame.district_id == "IND013018")
                   & (frame.season_start_year == 2022)]
    if target.empty:
        target = frame[frame.season_start_year == 2022].head(1)
    row = target.iloc[0]
    plot_district(row, FIG / "07_probability_distribution.png")
    record = event_table(target).iloc[0]
    plot_event_panel(record, FIG / "08_event_probabilities.png")

    print(f"\n=== {record['district']}, {record['state']} "
          f"{record['season']} ===")
    print(f"  last season      {record.last_season:>8,.0f} kg/ha")
    print(f"  point forecast   {record.point_forecast:>8,.0f} kg/ha")
    print(f"  80% range        {record.q10:>8,.0f} - {record.q90:,.0f}")
    print(f"  actual harvest   {record.actual:>8,.0f} kg/ha")
    print()
    for decline in reversed(DECLINES):
        print(f"  P(fall over {int(decline*100):>2}%)     "
              f"{record[f'p_fall_over_{int(decline*100)}pct']:>7.1%}")
    print(f"  P(any increase)      {record.p_increase:>7.1%}")
    for rise in INCREASES:
        print(f"  P(rise over {int(rise*100):>2}%)     "
              f"{record[f'p_rise_over_{int(rise*100)}pct']:>7.1%}")

    print("\n=== Are these probabilities honest? (all 476 district-seasons) ===")
    print(f"{'stated probability':<22}{'predicted':>11}{'actually happened':>19}")
    for decline in reversed(DECLINES):
        column = f"p_fall_over_{int(decline * 100)}pct"
        happened = (table.actual < (1 - decline) * table.last_season).mean()
        print(f"  fall over {int(decline*100):>2}%{'':<10}{table[column].mean():>11.1%}"
              f"{happened:>19.1%}")
    rose = (table.actual > table.last_season).mean()
    print(f"  any increase{'':<10}{table.p_increase.mean():>11.1%}{rose:>19.1%}")
    for rise in INCREASES:
        column = f"p_rise_over_{int(rise * 100)}pct"
        happened = (table.actual > (1 + rise) * table.last_season).mean()
        print(f"  rise over {int(rise*100):>2}%{'':<10}{table[column].mean():>11.1%}"
              f"{happened:>19.1%}")


if __name__ == "__main__":
    main()
