#!/usr/bin/env python3
"""Can the January forecast be improved the same way March was?

The 5 March model gained by training on more seasons: its satellite-aware models
had been restricted to 2017+ because one data source started then, while the
weather and economic inputs reached back further.  Widening the training window
was worth more than any modelling change.

The January forecast has never had that treatment.  It is a locked model scoring
307.0 kg/ha over four seasons.  A multi-date feature table exists with January
features running 2010-2022, so the identical experiment is possible:

    correction = (model trained on the long window)
               - (model trained on the short window)

    forecast   = existing January forecast + gamma * correction

with gamma chosen only on 2019-2020 and applied unchanged to 2021-2022, using
the same regularized near-tie rule as the March model -- among weights within
tol = max(0.35, 0.0015 x best) of the best development score, take the smallest.

This is a cheap experiment: two gradient-boosted fits per season, no networks.
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
from v16_common import year_block_bootstrap  # noqa: E402

ART = V16 / "artifacts"
MULTIDATE = (RAPID / "v4" / "agent_multidate" / "artifacts" / "multidate"
             / "feature_table_multidate.parquet")
V13 = (RAPID / "v13_crop_response_final" / "artifacts" / "final_predictions.parquet")
V15A = RAPID / "v15_complete_hierarchy" / "artifacts"
TARGET = "yield_kg_per_ha"
BASELINE = "baseline_weighted_recent"
YEARS = (2019, 2020, 2021, 2022)
WIDE_START, NARROW_START = 2010, 2017
GAMMA_GRID = [round(0.25 * i, 2) for i in range(0, 13)]


def rmse(prediction, truth):
    return float(np.sqrt(np.mean((np.asarray(prediction, float)
                                  - np.asarray(truth, float)) ** 2)))


def near_tie(scores: dict[float, float]) -> float:
    best = min(scores.values())
    tolerance = max(0.35, 0.0015 * best)
    return min(g for g, s in scores.items() if s <= best + tolerance)


def main() -> None:
    table = pd.read_parquet(MULTIDATE)
    table = table[table.availability_profile.eq("documented_latency")
                  & table.clock.eq("jan15")].copy()
    table[BASELINE] = (0.60 * table.lag_1_yield + 0.25 * table.lag_2_yield
                       + 0.15 * table.lag_3_yield)
    table = table[table[TARGET].notna() & table[BASELINE].notna()]
    skip = {TARGET, "season_start_year", "season_end_year", "area_ha",
            "production_tonnes", "yield_ton_per_ha", BASELINE}
    features = [c for c in table.columns
                if table[c].dtype.kind in "fi" and c not in skip]
    print(f"January feature table: {len(table)} rows, {len(features)} features, "
          f"seasons {table.season_start_year.min()}-{table.season_start_year.max()}")

    blocks = []
    for year in YEARS:
        wide = table[table.season_start_year.between(WIDE_START, year - 1)]
        narrow = table[table.season_start_year.between(NARROW_START, year - 1)]
        test = table[table.season_start_year.eq(year)]
        block = test[["district_id", "season_start_year"]].copy()
        block["wide"] = lab.xgb_residual_predict(wide, test, features, 2)
        block["narrow"] = lab.xgb_residual_predict(narrow, test, features, 2)
        blocks.append(block)
        print(f"  {year}: wide {len(wide)} rows, narrow {len(narrow)} rows",
              flush=True)
    predictions = pd.concat(blocks, ignore_index=True)

    anchor = pd.read_parquet(V13)
    anchor = anchor[anchor.clock.eq("jan15")][[
        "district_id", "season_start_year", "actual", "prediction"]].rename(
        columns={"prediction": "january_anchor", "actual": TARGET})
    reference = pd.read_parquet(V15A / "final_predictions.parquet")[[
        "district_id", "season_start_year", "state_name", "lag_1_yield"]]
    frame = (anchor.merge(predictions, on=["district_id", "season_start_year"])
             .merge(reference, on=["district_id", "season_start_year"]))

    truth = frame[TARGET].to_numpy(float)
    base = frame.january_anchor.to_numpy(float)
    correction = (frame.wide - frame.narrow).to_numpy(float)
    dev = frame.season_start_year.isin([2019, 2020]).to_numpy()
    late = ~dev

    scores = {g: rmse((base + g * correction)[dev], truth[dev])
              for g in GAMMA_GRID}
    gamma = near_tie(scores)
    improved = np.clip(base + gamma * correction, 500, 7000)
    frame["january_improved"] = improved
    print(f"\nweight chosen on 2019-2020 only: gamma = {gamma}")

    print("\n=== January forecast, before and after ===")
    print(f"{'':<26}{'2019-20':>10}{'2021-22':>10}{'four-year':>12}"
          f"{'direction':>11}")
    for label, values in (("existing January model", base),
                          ("with longer training window", improved)):
        d = direction = np.mean(
            (truth > frame.lag_1_yield.to_numpy(float))
            == (values > frame.lag_1_yield.to_numpy(float)))
        print(f"{label:<26}{rmse(values[dev], truth[dev]):>10.1f}"
              f"{rmse(values[late], truth[late]):>10.1f}"
              f"{rmse(values, truth):>12.1f}{d:>11.1%}")
    print(f"{'5 March, for reference':<26}{257.0:>10.1f}{288.6:>10.1f}"
          f"{273.3:>12.1f}{0.779:>11.1%}")

    frame["anchor"] = base
    print("\n=== Season-resampled bootstrap ===")
    rows = []
    for label, mask in (("2021-22 (untouched)", late),
                        ("four-year", np.ones(len(frame), bool))):
        b = year_block_bootstrap(frame[mask], "january_improved", "anchor")
        rows.append({"period": label, **b})
        print(f"  {label:<22} gain {b['mean_gain']:+7.2f} "
              f"[{b['p025']:+7.2f}, {b['p975']:+7.2f}] "
              f"seasons improved: "
              f"{sum(rmse(frame[frame.season_start_year.eq(y)].anchor, frame[frame.season_start_year.eq(y)][TARGET]) > rmse(frame[frame.season_start_year.eq(y)].january_improved, frame[frame.season_start_year.eq(y)][TARGET]) for y in frame[mask].season_start_year.unique())}"
              f"/{frame[mask].season_start_year.nunique()}")
    pd.DataFrame(rows).to_csv(ART / "january_improvement_bootstrap.csv", index=False)

    print("\n=== Per season ===")
    for year in YEARS:
        block = frame[frame.season_start_year.eq(year)]
        before = rmse(block.anchor, block[TARGET])
        after = rmse(block.january_improved, block[TARGET])
        print(f"  {year}: {before:7.1f} -> {after:7.1f}   ({before - after:+6.2f})")

    frame.to_parquet(ART / "january_improved_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
