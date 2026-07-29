#!/usr/bin/env python3
"""Forecasts at three clocks: 15 January, 15 February and 5 March.

Everything else in this project answers the question on 5 March only.  But a
forecast issued in January is worth more to a planner than one issued in March,
if it is good enough -- and the honest way to present the system is to show how
much accuracy is bought by waiting.

A multi-date feature table exists covering four cutoffs under two data-latency
assumptions.  The `documented_latency` profile is used here, which applies
realistic publication delays (weather 3 days, solar 7 days, forecasts issued 2
days before the clock) rather than assuming same-day data.  That is the profile
that matches how the system would actually run.

At each clock, the same procedure runs end to end:

  1. predict the residual around the three-season weighted baseline, using a
     depth-2 decision-tree ensemble, trained only on earlier seasons;
  2. turn past out-of-sample errors into 19 quantiles by the method in the
     walkthrough -- scale errors by district volatility, weight seasons
     equally, take weighted quantiles, rescale to the target district;
  3. read event probabilities off the interpolated distribution.

The comparison across clocks is the point: it shows what information arrives
between January and March, and what it is worth.
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
UGP = RAPID.parent
sys.path.insert(0, str(UGP))
sys.path.insert(0, str(V16 / "scripts"))
from rapid_yield_forecast.v14_anomaly_distribution.scripts import (  # noqa: E402
    run_v14_lab as lab)
from v16_common import year_block_bootstrap  # noqa: E402

MULTIDATE = (RAPID / "v4" / "agent_multidate" / "artifacts" / "multidate"
             / "feature_table_multidate.parquet")
ARTIFACTS = V16 / "artifacts"
FIG = V16 / "figures"
TARGET = "yield_kg_per_ha"
BASELINE = "baseline_weighted_recent"
CLOCKS = ["jan15", "feb15", "mar05"]
CLOCK_LABEL = {"jan15": "15 January", "feb15": "15 February", "mar05": "5 March"}
TEST_YEARS = list(range(2014, 2023))
LEVELS = np.array([round(0.05 * i, 2) for i in range(1, 20)])
QCOLUMNS = [f"q{int(round(a * 100)):02d}" for a in LEVELS]
DECLINES = [0.05, 0.10, 0.20, 0.30]
INCREASES = [0.05, 0.10, 0.20]
CLIP = (500.0, 7000.0)

INK, COOL, ACCENT, MUTED = "#1b2430", "#1d4e89", "#c2410c", "#94a3b8"
CLOCK_COLOUR = {"jan15": "#94a3b8", "feb15": "#3b7dd8", "mar05": "#1d4e89"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})


def load_clock(clock: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(MULTIDATE)
    frame = frame[frame.availability_profile.eq("documented_latency")
                  & frame.clock.eq(clock)].copy()
    frame[BASELINE] = (0.60 * frame.lag_1_yield + 0.25 * frame.lag_2_yield
                       + 0.15 * frame.lag_3_yield)
    frame = frame[frame[TARGET].notna() & frame[BASELINE].notna()]
    skip = {TARGET, "season_start_year", "season_end_year", "area_ha",
            "production_tonnes", "yield_ton_per_ha", BASELINE}
    features = [c for c in frame.columns
                if frame[c].dtype.kind in "fi" and c not in skip]
    return frame.reset_index(drop=True), features


def quantiles_from_errors(history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The walkthrough's method: scale errors, weight seasons equally, rescale."""
    errors = (history[TARGET] - history["prediction"]).to_numpy(float)
    scale_history = history["error_scale"].to_numpy(float)
    scaled = errors / scale_history
    weights = history.groupby("season_start_year")[TARGET].transform(
        lambda s: 1.0 / len(s)).to_numpy(float)

    order = np.argsort(scaled)
    values, w = scaled[order], weights[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
    shape = np.interp(LEVELS, cumulative, values)

    point = test["prediction"].to_numpy(float)[:, None]
    scale_test = test["error_scale"].to_numpy(float)[:, None]
    grid = np.clip(point + scale_test * shape[None, :], *CLIP)
    return np.maximum.accumulate(grid, axis=1)


def event_probabilities(quantiles: np.ndarray, last: np.ndarray) -> pd.DataFrame:
    records = []
    for row, previous in zip(quantiles, last):
        xs = np.concatenate([[row[0] - 400.0], row, [row[-1] + 400.0]])
        ps = np.concatenate([[0.0], LEVELS, [1.0]])
        order = np.argsort(xs)
        F = lambda y: float(np.interp(y, xs[order], ps[order]))  # noqa: E731
        record = {"p_increase": 1.0 - F(previous)}
        for decline in DECLINES:
            record[f"p_fall_over_{int(decline*100)}pct"] = F((1 - decline) * previous)
        for rise in INCREASES:
            record[f"p_rise_over_{int(rise*100)}pct"] = 1.0 - F((1 + rise) * previous)
        records.append(record)
    return pd.DataFrame(records)


def run_clock(clock: str) -> pd.DataFrame:
    panel, features = load_clock(clock)
    blocks = []
    for year in TEST_YEARS:
        train = panel[panel.season_start_year.between(2010, year - 1)]
        test = panel[panel.season_start_year.eq(year)]
        if test.empty or train.season_start_year.nunique() < 3:
            continue
        block = test[["district_id", "district_name", "state_name",
                      "season_start_year", TARGET, "lag_1_yield",
                      BASELINE]].copy()
        block["prediction"] = lab.xgb_residual_predict(train, test, features, 2)
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)

    # district volatility, used to put errors on a common scale
    recent = panel.groupby("district_id")[TARGET].apply(
        lambda s: s.rolling(5, min_periods=3).std().shift(1).mean())
    frame["recent_sd"] = frame.district_id.map(recent)
    frame["error_scale"] = np.maximum.reduce([
        frame.recent_sd.fillna(0).to_numpy(float),
        0.07 * frame[BASELINE].to_numpy(float),
        np.full(len(frame), 150.0)])

    # quantiles from errors of seasons already forecast
    pieces = []
    for year in sorted(frame.season_start_year.unique()):
        history = frame[frame.season_start_year < year]
        test = frame[frame.season_start_year.eq(year)].copy()
        if len(history) < 200:
            continue
        grid = quantiles_from_errors(history, test)
        for i, column in enumerate(QCOLUMNS):
            test[column] = grid[:, i]
        probabilities = event_probabilities(
            grid, test.lag_1_yield.to_numpy(float))
        test = pd.concat([test.reset_index(drop=True), probabilities], axis=1)
        pieces.append(test)
    out = pd.concat(pieces, ignore_index=True)
    out["clock"] = clock
    return out


def metrics(frame: pd.DataFrame) -> dict:
    error = frame.prediction.to_numpy(float) - frame[TARGET].to_numpy(float)
    y = frame[TARGET].to_numpy(float)
    lag = frame.lag_1_yield.to_numpy(float)
    q = frame[QCOLUMNS].to_numpy(float)
    pin = np.mean([np.mean(np.maximum(a * (y - q[:, i]), (a - 1) * (y - q[:, i])))
                   for i, a in enumerate(LEVELS)])
    fell = (y < 0.90 * lag).astype(float)
    p = frame.p_fall_over_10pct.to_numpy(float)
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    P, N = fell.sum(), (1 - fell).sum()
    auc = ((ranks[fell == 1].sum() - P * (P + 1) / 2) / (P * N)) if P and N else np.nan
    return {"rows": len(frame),
            "typical_error": float(np.sqrt(np.mean(error ** 2))),
            "mean_error": float(np.mean(np.abs(error))),
            "direction_accuracy": float(np.mean((y > lag) == (
                frame.prediction.to_numpy(float) > lag))),
            "crps": float(2 * pin),
            "coverage_80": float(np.mean((y >= q[:, 1]) & (y <= q[:, 17]))),
            "mean_width_80": float(np.mean(q[:, 17] - q[:, 1])),
            "auc_severe_drop": float(auc)}


def main() -> None:
    everything = []
    rows = []
    for clock in CLOCKS:
        frame = run_clock(clock)
        everything.append(frame)
        rows.append({"clock": CLOCK_LABEL[clock], **metrics(frame)})
        print(f"  {CLOCK_LABEL[clock]}: {len(frame)} forecasts", flush=True)
    combined = pd.concat(everything, ignore_index=True)
    combined.to_parquet(ARTIFACTS / "multiclock_predictions.parquet", index=False)
    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "multiclock_metrics.csv", index=False)

    print("\n=== How much does waiting buy you? (2014-2022, 9 seasons) ===")
    print(report.to_string(index=False))

    print("\n=== Is the later forecast really better? (season-resampled) ===")
    wide = combined.pivot_table(
        index=["district_id", "season_start_year"], columns="clock",
        values="prediction").reset_index()
    truth = combined[combined.clock.eq("mar05")][
        ["district_id", "season_start_year", TARGET, "lag_1_yield", "state_name"]]
    wide = wide.merge(truth, on=["district_id", "season_start_year"])
    for earlier, later in (("jan15", "feb15"), ("feb15", "mar05"),
                           ("jan15", "mar05")):
        b = year_block_bootstrap(wide, later, earlier)
        print(f"  {CLOCK_LABEL[later]} vs {CLOCK_LABEL[earlier]:<12} "
              f"gain {b['mean_gain']:+7.2f} "
              f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
              f"P(>0)={b['probability_positive']:.3f}")

    # per-clock probabilities for one district
    example = combined[(combined.district_name.eq("Rewari"))
                       & (combined.season_start_year.eq(2022))]
    if example.empty:
        example = combined[combined.season_start_year.eq(2022)].head(3)
    print(f"\n=== {example.district_name.iloc[0]}, "
          f"{example.state_name.iloc[0]} 2022 at each clock ===")
    print(f"  last season {example.lag_1_yield.iloc[0]:,.0f}  |  "
          f"actual {example[TARGET].iloc[0]:,.0f} kg/ha\n")
    print(f"{'clock':<14}{'forecast':>10}{'80% range':>20}"
          f"{'P(fall>10%)':>13}{'P(increase)':>13}")
    for clock in CLOCKS:
        r = example[example.clock.eq(clock)]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"{CLOCK_LABEL[clock]:<14}{r.prediction:>10,.0f}"
              f"{f'{r.q10:,.0f} - {r.q90:,.0f}':>20}"
              f"{r.p_fall_over_10pct:>13.0%}{r.p_increase:>13.0%}")

    plot_clock_comparison(report, combined, example)


def plot_clock_comparison(report, combined, example) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    ax = axes[0]
    x = np.arange(len(report))
    ax.bar(x, report.typical_error, color=[CLOCK_COLOUR[c] for c in CLOCKS],
           width=0.6)
    for i, value in enumerate(report.typical_error):
        ax.text(i, value + 3, f"{value:.0f}", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(report.clock)
    ax.set_ylabel("typical error (kg/ha)")
    ax.set_title("Accuracy improves as the season unfolds",
                 fontsize=11.5, fontweight="bold", loc="left")

    ax = axes[1]
    if not example.empty:
        for clock in CLOCKS:
            r = example[example.clock.eq(clock)]
            if r.empty:
                continue
            r = r.iloc[0]
            grid = r[QCOLUMNS].to_numpy(float)
            widths = np.maximum(np.diff(grid), 1e-6)
            heights = np.diff(LEVELS) / widths
            mid = (grid[:-1] + grid[1:]) / 2
            xs = np.linspace(grid[0] - 200, grid[-1] + 200, 600)
            ys = np.interp(xs, mid, heights, left=0, right=0)
            sigma = (grid[-1] - grid[0]) / 20
            step = xs[1] - xs[0]
            k = np.exp(-0.5 * (np.arange(-int(3*sigma/step),
                                         int(3*sigma/step)+1)*step/sigma)**2)
            ys = np.convolve(ys, k/k.sum(), mode="same")
            ys = ys / np.trapezoid(ys, xs) * 100 if hasattr(np, "trapezoid") \
                else ys / np.trapz(ys, xs) * 100
            ax.plot(xs, ys, color=CLOCK_COLOUR[clock], linewidth=2.2,
                    label=CLOCK_LABEL[clock])
        ax.axvline(example[TARGET].iloc[0], color=ACCENT, linestyle="--",
                   linewidth=2, label=f"actual {example[TARGET].iloc[0]:,.0f}")
        ax.set_xlabel("wheat yield (kg/ha)")
        ax.set_ylabel("chance per 100 kg/ha band")
        ax.set_yticklabels([f"{t:.0%}" for t in ax.get_yticks()])
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"{example.district_name.iloc[0]} 2022: the forecast sharpens",
                     fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "09_clock_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
