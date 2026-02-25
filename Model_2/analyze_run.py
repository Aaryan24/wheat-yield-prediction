#!/usr/bin/env python3
"""
analyze_run.py — Standard Deviation Analysis for a Specific Run
================================================================
Reads predictions.csv from a given run_X folder and prints a detailed
summary of standard deviation analysis. All analysis is done separately
for EACH YEAR and then for TOTAL (all data combined).

Usage:
  python Model_2/analyze_run.py --run 1
  python Model_2/analyze_run.py --run 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _header(title: str, width: int = 72) -> str:
    pad = (width - len(title) - 2) // 2
    return f"\n{'═' * pad} {title} {'═' * pad}"


def _section_banner(label: str, width: int = 72) -> str:
    pad = (width - len(label) - 4) // 2
    return f"\n{'▓' * pad}  {label}  {'▓' * pad}"


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis functions (operate on a filtered DataFrame)
# ═══════════════════════════════════════════════════════════════════════════════

def _overall_error_analysis(df: pd.DataFrame) -> None:
    """Overall error distribution statistics."""
    print(_header("ERROR DISTRIBUTION"))
    errors = df["error_kg_per_ha"].to_numpy()
    abs_errors = df["abs_error_kg_per_ha"].to_numpy()
    mape = df["mape_percent"].dropna().to_numpy()

    print(f"  Total predictions        : {len(df)}")
    print(f"  Mean Error               : {errors.mean():>10.2f} kg/ha")
    print(f"  Std Dev of Error         : {errors.std():>10.2f} kg/ha")
    print(f"  Mean Absolute Error      : {abs_errors.mean():>10.2f} kg/ha")
    print(f"  Std Dev of Abs Error     : {abs_errors.std():>10.2f} kg/ha")
    print(f"  Min Error                : {errors.min():>10.2f} kg/ha")
    print(f"  Max Error                : {errors.max():>10.2f} kg/ha")
    print(f"  Median Error             : {np.median(errors):>10.2f} kg/ha")
    print(f"  25th Percentile (Error)  : {np.percentile(errors, 25):>10.2f} kg/ha")
    print(f"  75th Percentile (Error)  : {np.percentile(errors, 75):>10.2f} kg/ha")
    print(f"  IQR (Error)              : {np.percentile(errors, 75) - np.percentile(errors, 25):>10.2f} kg/ha")

    print(f"\n  Mean MAPE                : {mape.mean():>10.2f}%")
    print(f"  Std Dev of MAPE          : {mape.std():>10.2f}%")
    print(f"  Median MAPE              : {np.median(mape):>10.2f}%")

    # Outlier detection.
    threshold_2s = errors.mean() + 2 * errors.std()
    threshold_2s_low = errors.mean() - 2 * errors.std()
    outliers = df[(df["error_kg_per_ha"] > threshold_2s) | (df["error_kg_per_ha"] < threshold_2s_low)]
    print(f"\n  Outliers (beyond ±2σ)    : {len(outliers)} / {len(df)} ({len(outliers)/len(df)*100:.1f}%)")
    threshold_3s = errors.mean() + 3 * errors.std()
    threshold_3s_low = errors.mean() - 3 * errors.std()
    outliers_3 = df[(df["error_kg_per_ha"] > threshold_3s) | (df["error_kg_per_ha"] < threshold_3s_low)]
    print(f"  Outliers (beyond ±3σ)    : {len(outliers_3)} / {len(df)} ({len(outliers_3)/len(df)*100:.1f}%)")


def _sigma_band_analysis(df: pd.DataFrame) -> None:
    """How many predictions fall within ±1σ, ±2σ, ±3σ of mean error."""
    print(_header("SIGMA BAND ANALYSIS"))
    errors = df["error_kg_per_ha"].to_numpy()
    mu = errors.mean()
    sigma = errors.std()

    if sigma < 1e-9:
        print("  ⚠ Standard deviation is ~0, skipping sigma band analysis.")
        return

    print(f"  Mean Error (μ)           : {mu:>10.2f} kg/ha")
    print(f"  Std Dev (σ)              : {sigma:>10.2f} kg/ha")
    print()

    for k in [1, 2, 3]:
        lo = mu - k * sigma
        hi = mu + k * sigma
        within = np.sum((errors >= lo) & (errors <= hi))
        pct = within / len(errors) * 100
        expected = {1: 68.27, 2: 95.45, 3: 99.73}[k]
        status = "✓" if pct >= expected * 0.9 else "⚠"
        print(f"  Within ±{k}σ [{lo:>+9.1f}, {hi:>+9.1f}] : {within:>5d} / {len(errors)} ({pct:>5.1f}%)  expected ~{expected:.1f}%  {status}")

    print(f"\n  Interpretation:")
    print(f"    If errors are normally distributed, ~68% should be within ±1σ,")
    print(f"    ~95% within ±2σ, and ~99.7% within ±3σ.")


def _per_state_analysis(df: pd.DataFrame) -> None:
    """Per-state standard deviation breakdown."""
    print(_header("PER-STATE ANALYSIS"))
    fmt = "  {:<20s}  {:>6d}  {:>10.2f}  {:>10.2f}  {:>10.2f}  {:>10.2f}  {:>8.2f}%"
    print(f"  {'State':<20s}  {'N':>6s}  {'Mean Err':>10s}  {'Std Err':>10s}  {'Mean |Err|':>10s}  {'Std |Err|':>10s}  {'Mean MAPE':>8s}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*9}")

    for state in sorted(df["state_name"].unique()):
        sd = df[df["state_name"] == state]
        errs = sd["error_kg_per_ha"].to_numpy()
        abs_errs = sd["abs_error_kg_per_ha"].to_numpy()
        mape = sd["mape_percent"].dropna().to_numpy()
        print(fmt.format(
            state, len(sd),
            errs.mean(), errs.std() if len(errs) > 1 else 0.0,
            abs_errs.mean(), abs_errs.std() if len(abs_errs) > 1 else 0.0,
            mape.mean() if len(mape) > 0 else float("nan"),
        ))


def _yield_variability(df: pd.DataFrame) -> None:
    """Compare actual yield variability vs prediction variability."""
    print(_header("YIELD VARIABILITY (Actual vs Predicted)"))

    actual = df["actual_yield_kg_per_ha"].to_numpy()
    predicted = df["predicted_yield_kg_per_ha"].to_numpy()

    print(f"  {'Metric':<30s}  {'Actual':>12s}  {'Predicted':>12s}  {'Diff':>12s}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*12}  {'─'*12}")
    print(f"  {'Mean yield (kg/ha)':<30s}  {actual.mean():>12.2f}  {predicted.mean():>12.2f}  {predicted.mean()-actual.mean():>+12.2f}")
    print(f"  {'Std Dev yield (kg/ha)':<30s}  {actual.std():>12.2f}  {predicted.std():>12.2f}  {predicted.std()-actual.std():>+12.2f}")
    cv_actual = actual.std() / actual.mean() * 100 if actual.mean() != 0 else 0
    cv_pred = predicted.std() / predicted.mean() * 100 if predicted.mean() != 0 else 0
    print(f"  {'Coeff of Variation (%)':<30s}  {cv_actual:>12.2f}  {cv_pred:>12.2f}  {'':>12s}")
    print(f"  {'Min (kg/ha)':<30s}  {actual.min():>12.2f}  {predicted.min():>12.2f}  {'':>12s}")
    print(f"  {'Max (kg/ha)':<30s}  {actual.max():>12.2f}  {predicted.max():>12.2f}  {'':>12s}")
    print(f"  {'Range (kg/ha)':<30s}  {actual.max()-actual.min():>12.2f}  {predicted.max()-predicted.min():>12.2f}  {'':>12s}")

    corr = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0.0
    print(f"\n  Pearson Correlation       : {corr:.4f}")

    ratio = predicted.std() / actual.std() if actual.std() > 0 else 0
    print(f"  Pred/Actual Std Ratio    : {ratio:.4f}")
    if ratio < 0.8:
        print(f"  ⚠ Model UNDER-estimates variability (ratio < 0.8)")
    elif ratio > 1.2:
        print(f"  ⚠ Model OVER-estimates variability (ratio > 1.2)")
    else:
        print(f"  ✓ Model captures variability reasonably well")


def _accuracy_classification(df: pd.DataFrame) -> None:
    """MAPE-based accuracy classification."""
    print(_header("ACCURACY CLASSIFICATION"))

    mape = df["mape_percent"].to_numpy()
    total = len(df)

    def _cat(m):
        if np.isnan(m): return "unknown"
        if m < 2: return "accurate"
        if m < 5: return "somewhat_accurate"
        if m < 10: return "somewhat_inaccurate"
        return "inaccurate"

    cats = np.array([_cat(m) for m in mape])
    labels = [
        ("accurate", "Accurate (MAPE < 2%)"),
        ("somewhat_accurate", "Somewhat Accurate (2-5%)"),
        ("somewhat_inaccurate", "Somewhat Inaccurate (5-10%)"),
        ("inaccurate", "Inaccurate (> 10%)"),
    ]
    for key, label in labels:
        cnt = int((cats == key).sum())
        print(f"  {label:<35s} : {cnt:>5d} ({cnt/total*100:>5.1f}%)")


def _trend_analysis(df: pd.DataFrame) -> None:
    """District-wise trend direction analysis (needs ≥2 seasons)."""
    seasons = sorted(df["season_year"].unique())
    if len(seasons) < 2:
        print(_header("TREND ANALYSIS"))
        print("  ⚠ Only 1 season — trend analysis needs ≥2 seasons, skipping.")
        return

    print(_header("TREND ANALYSIS"))

    THRESHOLD_PCT = 0.5
    rows = []

    for i in range(len(seasons) - 1):
        y_from, y_to = seasons[i], seasons[i + 1]
        df_from = df[df["season_year"] == y_from].set_index("district_id")
        df_to = df[df["season_year"] == y_to].set_index("district_id")
        common = df_from.index.intersection(df_to.index)

        for did in common:
            a_from = df_from.loc[did, "actual_yield_kg_per_ha"]
            a_to = df_to.loc[did, "actual_yield_kg_per_ha"]
            p_from = df_from.loc[did, "predicted_yield_kg_per_ha"]
            p_to = df_to.loc[did, "predicted_yield_kg_per_ha"]

            a_chg = ((a_to - a_from) / max(abs(a_from), 1e-6)) * 100
            p_chg = ((p_to - p_from) / max(abs(p_from), 1e-6)) * 100

            a_dir = "increase" if a_chg > THRESHOLD_PCT else ("decrease" if a_chg < -THRESHOLD_PCT else "stable")
            p_dir = "increase" if p_chg > THRESHOLD_PCT else ("decrease" if p_chg < -THRESHOLD_PCT else "stable")

            rows.append({"actual_direction": a_dir, "predicted_direction": p_dir})

    if not rows:
        print("  ⚠ No valid transitions found.")
        return

    tdf = pd.DataFrame(rows)
    total = len(tdf)
    correct = int((tdf["actual_direction"] == tdf["predicted_direction"]).sum())
    print(f"  Transitions evaluated    : {total}")
    print(f"  Direction matches        : {correct} ({correct/total*100:.1f}%)")

    directions = ["increase", "decrease", "stable"]
    print(f"\n  {'Direction':<12s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>8s}  {'Support':>8s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")

    precs, recs, f1s, supports = [], [], [], []
    for d in directions:
        tp = int(((tdf["actual_direction"] == d) & (tdf["predicted_direction"] == d)).sum())
        fp = int(((tdf["actual_direction"] != d) & (tdf["predicted_direction"] == d)).sum())
        fn = int(((tdf["actual_direction"] == d) & (tdf["predicted_direction"] != d)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        precs.append(precision); recs.append(recall); f1s.append(f1); supports.append(support)
        print(f"  {d.title():<12s}  {precision:>10.4f}  {recall:>8.4f}  {f1:>8.4f}  {support:>8d}")

    macro_p, macro_r, macro_f1 = np.mean(precs), np.mean(recs), np.mean(f1s)
    ts = sum(supports)
    if ts > 0:
        wp = sum(p*s for p, s in zip(precs, supports)) / ts
        wr = sum(r*s for r, s in zip(recs, supports)) / ts
        wf = sum(f*s for f, s in zip(f1s, supports)) / ts
    else:
        wp = wr = wf = 0.0
    print(f"\n  Macro Avg    :  P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}")
    print(f"  Weighted Avg :  P={wp:.4f}  R={wr:.4f}  F1={wf:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all analysis on a subset
# ═══════════════════════════════════════════════════════════════════════════════

def _run_full_analysis(df: pd.DataFrame, label: str) -> None:
    """Run every analysis section on the given DataFrame subset."""
    n = len(df)
    seasons = sorted(df["season_year"].unique())
    split_info = ""
    if "split" in df.columns:
        splits = sorted(df["split"].unique())
        split_info = f", split={splits}"
    print(_section_banner(f"{label}  ({n} predictions, seasons={list(seasons)}{split_info})"))

    _overall_error_analysis(df)
    _sigma_band_analysis(df)
    _per_state_analysis(df)
    _yield_variability(df)
    _accuracy_classification(df)
    _trend_analysis(df)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standard deviation analysis for a specific run."
    )
    parser.add_argument(
        "--run", type=int, required=True,
        help="Run number to analyze (e.g. --run 1 for run_1).",
    )
    parser.add_argument(
        "--analysis-dir", type=str, default="Model_2/analysis",
        help="Path to analysis directory.",
    )
    args = parser.parse_args()

    run_dir = Path(args.analysis_dir) / f"run_{args.run}"
    pred_path = run_dir / "predictions.csv"

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)
    if not pred_path.exists():
        print(f"ERROR: predictions.csv not found in {run_dir}")
        sys.exit(1)

    df = pd.read_csv(pred_path)
    all_years = sorted(df["season_year"].unique())

    print(f"\n{'█' * 72}")
    print(f"  STANDARD DEVIATION ANALYSIS — Run {args.run}")
    print(f"  Source: {run_dir}")
    print(f"  Total predictions: {len(df)}")
    print(f"  Seasons: {all_years}")
    print(f"{'█' * 72}")

    # Show saved metrics for quick reference.
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(_header("SAVED METRICS (from training)"))
        for split in ["train", "val", "test"]:
            m = metrics[split]
            print(f"  {split.title():<6s} — RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, "
                  f"MAPE={m['mape']:.2f}%, R²={m['r2']:.4f}")

    # ── Per-year analysis ────────────────────────────────────────────────
    for year in all_years:
        year_df = df[df["season_year"] == year]
        split = year_df["split"].iloc[0].upper() if "split" in year_df.columns else "?"
        _run_full_analysis(year_df, f"YEAR {year} ({split})")

    # ── Total (all data) ─────────────────────────────────────────────────
    _run_full_analysis(df, "TOTAL (ALL YEARS COMBINED)")

    print(f"\n{'█' * 72}")
    print(f"  Analysis complete for run_{args.run}")
    print(f"{'█' * 72}\n")


if __name__ == "__main__":
    main()
