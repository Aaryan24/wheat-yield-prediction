#!/usr/bin/env python3
"""Does the training-window fix hold up beyond the four V15 test seasons?

The +3.35 kg/ha window gain was measured on 2019-2022 only, because that is
where V15's own artifacts live.  Four seasons is exactly the sample size that
made every other V15 claim unresolvable, so the fix deserves the same scrutiny
it was used to apply to everything else.

V15's 78 physical/weather/economic panel starts in 2010, so the widest honest
test is nine rolling-origin seasons (2014-2022) rather than nineteen.  For each
test season the same model is trained two ways:

    NARROW   the most recent 2 seasons only  -- V15's effective window
    WIDE     every season from 2010 onward   -- what the data supports

Nothing else differs: same features, same depth-2 two-seed XGBoost, same
residual target, same fold cutoffs.  The gap between them is the value of the
training data V15 was discarding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
RAPID = V16.parent
UGP = RAPID.parent
sys.path.insert(0, str(UGP))
sys.path.insert(0, str(V16 / "scripts"))
from rapid_yield_forecast.v14_anomaly_distribution.scripts import (  # noqa: E402
    run_v14_lab as lab)
from v16_common import metrics, year_block_bootstrap  # noqa: E402

ARTIFACTS = V16 / "artifacts"
TARGET = "yield_kg_per_ha"
TEST_YEARS = list(range(2014, 2023))
NARROW_SEASONS = 2          # V15 trains 2017-2018 for its 2019 fold
PANEL_START = 2010


def main() -> None:
    panel, groups, _ = lab.load_panel()
    physical = groups["physical"]
    panel = panel[panel[TARGET].notna()
                  & panel["baseline_weighted_recent"].notna()].copy()

    blocks = []
    for year in TEST_YEARS:
        train_end = year - 1
        wide = panel[panel.season_start_year.between(PANEL_START, train_end)]
        narrow = panel[panel.season_start_year.between(
            train_end - NARROW_SEASONS + 1, train_end)]
        test = panel[panel.season_start_year.eq(year)]
        if test.empty or narrow.empty:
            continue
        block = test[["district_id", "state_name", "season_start_year",
                      TARGET, "lag_1_yield", "baseline_weighted_recent"]].copy()
        block["narrow"] = lab.xgb_residual_predict(narrow, test, physical, 2)
        block["wide"] = lab.xgb_residual_predict(wide, test, physical, 2)
        block["narrow_rows"] = len(narrow)
        block["wide_rows"] = len(wide)
        blocks.append(block)
        print(f"  {year}: narrow {len(narrow)} rows "
              f"({train_end - NARROW_SEASONS + 1}-{train_end}), "
              f"wide {len(wide)} rows ({PANEL_START}-{train_end})", flush=True)

    frame = pd.concat(blocks, ignore_index=True)
    frame["baseline"] = frame["baseline_weighted_recent"]

    print("\n=== Narrow vs wide training window, nine rolling seasons ===")
    rows = []
    for label, column in (("naive 3-season baseline", "baseline"),
                          ("NARROW window (V15 style)", "narrow"),
                          ("WIDE window (2010+)", "wide")):
        rows.append({"model": label, **metrics(frame, column)})
        late = frame[frame.season_start_year.isin([2019, 2020, 2021, 2022])]
        rows.append({"model": f"{label} [2019-22 only]", **metrics(late, column)})
    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "window_multifold_metrics.csv", index=False)
    print(report[["model", "rows", "years", "rmse", "mae", "bias",
                  "direction_accuracy"]].to_string(index=False))

    print("\n=== Season-resampled bootstrap: wide beats narrow? ===")
    boot = []
    for label, subset in (("all nine seasons", frame),
                          ("2019-22 only", frame[frame.season_start_year.isin(
                              [2019, 2020, 2021, 2022])])):
        b = year_block_bootstrap(subset, "wide", "narrow")
        boot.append({"period": label, **b})
        print(f"  {label:<20} gain {b['mean_gain']:+7.2f} "
              f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
              f"P(>0)={b['probability_positive']:.3f}")
    pd.DataFrame(boot).to_csv(ARTIFACTS / "window_multifold_bootstrap.csv",
                              index=False)

    per_year = pd.DataFrame([{
        "season_start_year": int(year),
        "narrow_rmse": metrics(block, "narrow")["rmse"],
        "wide_rmse": metrics(block, "wide")["rmse"],
        "gain": metrics(block, "narrow")["rmse"] - metrics(block, "wide")["rmse"],
        "train_rows_narrow": int(block.narrow_rows.iloc[0]),
        "train_rows_wide": int(block.wide_rows.iloc[0]),
    } for year, block in frame.groupby("season_start_year")])
    per_year.to_csv(ARTIFACTS / "window_multifold_per_year.csv", index=False)
    print("\n=== Per season ===")
    print(per_year.to_string(index=False))
    print(f"\nseasons where the wider window wins: "
          f"{int((per_year.gain > 0).sum())} of {len(per_year)}")
    frame.to_parquet(ARTIFACTS / "window_multifold_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
