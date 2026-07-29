#!/usr/bin/env python3
"""Calibrated probability forecasts for the 15 January clock.

The January point forecast exists in the earlier multi-date artifacts, but its
published uncertainty ranges were badly calibrated -- intervals labelled 80%
contained the outcome only 39.5% of the time.  They cannot be presented.

This rebuilds January's uncertainty from scratch using exactly the method the
5 March model uses (walkthrough section 9): take the model's own out-of-sample
errors, scale them by district volatility, weight seasons equally, read weighted
quantiles, then rescale onto each district.

Because January forecasts exist for only four seasons, quantiles for 2021-2022
are built from 2019-2020 errors alone.  That is thin, and is stated rather than
hidden -- but it is honest, whereas the published ranges were not.
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
RAPID = V16.parent
ART = V16 / "artifacts"
FIG = V16 / "figures"
TARGET = "yield_kg_per_ha"
LEVELS = np.array([round(0.05 * i, 2) for i in range(1, 20)])
QCOLUMNS = [f"q{int(round(a * 100)):02d}" for a in LEVELS]
DECLINES = [0.05, 0.10, 0.20, 0.30]
INCREASES = [0.05, 0.10, 0.20]
CLIP = (500.0, 7000.0)
BAND = 100.0

INK, COOL, ACCENT, MUTED = "#1b2430", "#1d4e89", "#c2410c", "#94a3b8"
JAN, MAR = "#7c9cc4", "#1d4e89"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def build() -> pd.DataFrame:
    v13 = pd.read_parquet(RAPID / "v13_crop_response_final" / "artifacts"
                          / "final_predictions.parquet")
    v13 = v13[v13.clock.eq("jan15")][
        ["district_id", "season_start_year", "actual", "prediction"]].rename(
        columns={"actual": TARGET, "prediction": "point"})
    reference = pd.read_parquet(
        RAPID / "v15_complete_hierarchy" / "artifacts" / "final_predictions.parquet"
    )[["district_id", "district_name", "state_name", "season_start_year",
       "lag_1_yield"]]
    frame = v13.merge(reference, on=["district_id", "season_start_year"])

    history = pd.read_parquet(RAPID / "v15_complete_hierarchy" / "data"
                              / "long_yield_1990_2022.parquet")
    volatility = (history.sort_values("season_start_year")
                  .groupby("district_id")[TARGET]
                  .apply(lambda s: s.rolling(5, min_periods=3).std().mean()))
    frame["recent_sd"] = frame.district_id.map(volatility)
    frame["error_scale"] = np.maximum.reduce([
        frame.recent_sd.fillna(0).to_numpy(float),
        0.07 * frame.point.to_numpy(float),
        np.full(len(frame), 150.0)])

    pieces = []
    for year in sorted(frame.season_start_year.unique()):
        past = frame[frame.season_start_year < year]
        test = frame[frame.season_start_year.eq(year)].copy()
        if len(past) < 100:
            continue
        scaled = ((past[TARGET] - past.point) / past.error_scale).to_numpy(float)
        weights = past.groupby("season_start_year")[TARGET].transform(
            lambda s: 1.0 / len(s)).to_numpy(float)
        order = np.argsort(scaled)
        values, w = scaled[order], weights[order]
        cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
        shape = np.interp(LEVELS, cumulative, values)

        grid = np.clip(test.point.to_numpy(float)[:, None]
                       + test.error_scale.to_numpy(float)[:, None] * shape[None, :],
                       *CLIP)
        grid = np.maximum.accumulate(grid, axis=1)
        for i, column in enumerate(QCOLUMNS):
            test[column] = grid[:, i]
        pieces.append(test)
    return pd.concat(pieces, ignore_index=True)


def probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in frame.iterrows():
        q = row[QCOLUMNS].to_numpy(float)
        last = float(row.lag_1_yield)
        xs = np.concatenate([[q[0] - 400.0], q, [q[-1] + 400.0]])
        ps = np.concatenate([[0.0], LEVELS, [1.0]])
        order = np.argsort(xs)
        F = lambda y: float(np.interp(y, xs[order], ps[order]))  # noqa: E731
        record = {"district_id": row.district_id, "district": row.district_name,
                  "state": row.state_name, "season": int(row.season_start_year),
                  "last_season": last, "point_forecast": float(row.point),
                  "q10": float(row.q10), "q90": float(row.q90),
                  "actual": float(row[TARGET]), "p_increase": 1.0 - F(last)}
        for d in DECLINES:
            record[f"p_fall_over_{int(d*100)}pct"] = F((1 - d) * last)
        for r in INCREASES:
            record[f"p_rise_over_{int(r*100)}pct"] = 1.0 - F((1 + r) * last)
        records.append(record)
    return pd.DataFrame(records)


def density(q, grid):
    heights = np.diff(LEVELS) / np.maximum(np.diff(q), 1e-6)
    mid = (q[:-1] + q[1:]) / 2
    raw = np.interp(grid, mid, heights, left=0, right=0)
    step = grid[1] - grid[0]
    sigma = max((q[-1] - q[0]) / 25.0, step)
    half = int(3 * sigma / step)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * step / sigma) ** 2)
    smoothed = np.convolve(raw, k / k.sum(), mode="same")
    area = np.trapezoid(smoothed, grid) if hasattr(np, "trapezoid") \
        else np.trapz(smoothed, grid)
    return smoothed / area * BAND if area > 0 else smoothed


def main() -> None:
    january = build()
    table = probabilities(january)
    table.to_csv(ART / "january_forecast_probabilities.csv", index=False)
    january.to_parquet(ART / "january_distribution.parquet", index=False)

    y = january[TARGET].to_numpy(float)
    q = january[QCOLUMNS].to_numpy(float)
    print(f"January distributions built for {len(january)} district-seasons "
          f"({sorted(january.season_start_year.unique())})")
    print(f"  80% coverage {np.mean((y >= q[:, 1]) & (y <= q[:, 17])):.1%}  "
          f"(published ranges were 39.5%)")
    print(f"  90% coverage {np.mean((y >= q[:, 0]) & (y <= q[:, 18])):.1%}")

    print("\n=== Are January probabilities honest? ===")
    print(f"{'':<22}{'stated':>10}{'happened':>11}")
    for d in reversed(DECLINES):
        column = f"p_fall_over_{int(d*100)}pct"
        actual = (table.actual < (1 - d) * table.last_season).mean()
        print(f"  fall over {int(d*100):>2}%{'':<9}{table[column].mean():>10.1%}"
              f"{actual:>11.1%}")
    print(f"  any increase{'':<9}{table.p_increase.mean():>10.1%}"
          f"{(table.actual > table.last_season).mean():>11.1%}")

    # Rewari at both clocks
    march = pd.read_parquet(ART / "final_predictions.parquet")
    mrow = march[(march.district_id == "IND013018")
                 & (march.season_start_year == 2022)].iloc[0]
    jrow = january[(january.district_id == "IND013018")
                   & (january.season_start_year == 2022)]
    jtab = table[(table.district_id == "IND013018") & (table.season == 2022)]
    mtab = pd.read_csv(ART / "forecast_probabilities.csv")
    mtab = mtab[(mtab.district == "Rewari") & (mtab.season == 2022)].iloc[0]
    if jrow.empty:
        print("\n(no January distribution for Rewari 2022)")
        return
    jrow, jtab = jrow.iloc[0], jtab.iloc[0]

    print("\n=== Rewari, Haryana 2022 — January vs March ===")
    print(f"  last season {jrow.lag_1_yield:,.0f}   actual {jrow[TARGET]:,.0f} kg/ha\n")
    print(f"{'':<20}{'15 January':>14}{'5 March':>14}")
    print(f"{'point forecast':<20}{jrow.point:>14,.0f}{mrow.final_point:>14,.0f}")
    print(f"{'error':<20}{abs(jrow.point-jrow[TARGET]):>14,.0f}"
          f"{abs(mrow.final_point-mrow[TARGET]):>14,.0f}")
    print(f"{'80% range':<20}{f'{jrow.q10:,.0f}-{jrow.q90:,.0f}':>14}"
          f"{f'{mrow.final_q10:,.0f}-{mrow.final_q90:,.0f}':>14}")
    print(f"{'P(increase)':<20}{jtab.p_increase:>14.0%}{mtab.p_increase:>14.0%}")
    for d in (5, 10, 20):
        print(f"{f'P(fall >{d}%)':<20}{jtab[f'p_fall_over_{d}pct']:>14.0%}"
              f"{mtab[f'p_fall_over_{d}pct']:>14.0%}")

    # figure: both distributions on one axis
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    for row, columns, colour, label in (
            (jrow, QCOLUMNS, JAN, "15 January"),
            (mrow, [f"final_{c}" for c in QCOLUMNS], MAR, "5 March")):
        grid_q = row[columns].to_numpy(float)
        xs = np.linspace(grid_q[0] - 250, grid_q[-1] + 250, 800)
        ax.plot(xs, density(grid_q, xs), color=colour, linewidth=2.4, label=label)
        ax.fill_between(xs, density(grid_q, xs), color=colour, alpha=0.12)
    actual = float(jrow[TARGET])
    ax.axvline(actual, color=ACCENT, linestyle="--", linewidth=2.2)
    ax.text(actual - 40, ax.get_ylim()[1] * 0.92, f"actual {actual:,.0f}",
            rotation=90, ha="right", va="top", color=ACCENT, fontweight="bold",
            fontsize=10)
    ax.axvline(float(jrow.lag_1_yield), color=MUTED, linestyle=":", linewidth=1.8)
    ax.text(float(jrow.lag_1_yield) + 30, ax.get_ylim()[1] * 0.92,
            f"last season {jrow.lag_1_yield:,.0f}", rotation=90, va="top",
            color=MUTED, fontsize=9.5)
    ax.set_xlabel("wheat yield (kg/ha)")
    ax.set_ylabel(f"chance of landing within ±{BAND/2:.0f} kg/ha")
    ticks = ax.get_yticks(); ticks = ticks[ticks >= 0]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:.0%}" for t in ticks])
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.set_title("Rewari, Haryana 2022 — the forecast sharpens and shifts\n"
                 "January is wide and centred too high; March finds the truth",
                 fontsize=12, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "11_january_vs_march.png", dpi=170)
    plt.close(fig)
    print("\nwrote figures/11_january_vs_march.png")


if __name__ == "__main__":
    main()
