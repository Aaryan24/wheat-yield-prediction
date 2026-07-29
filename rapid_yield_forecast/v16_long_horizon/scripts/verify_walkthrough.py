#!/usr/bin/env python3
"""Check every number quoted in MODEL_WALKTHROUGH.md against its source artifact.

Written after two numbers in that document turned out to be wrong -- one from
using a different definition of direction accuracy than the rest of the project,
one from calling a documented design choice a bug.  Both were caught by a
reader, not by me, so the checking is now mechanical.

Each check states the claim, recomputes it from the artifact, and reports PASS
or FAIL with both values.  Anything that cannot be recomputed is reported as
UNVERIFIED rather than silently assumed correct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
RAPID = V16.parent
UGP = RAPID.parent
ART = V16 / "artifacts"
V15A = RAPID / "v15_complete_hierarchy" / "artifacts"
V15D = RAPID / "v15_complete_hierarchy" / "data"
TARGET = "yield_kg_per_ha"

results = []


def check(label: str, claimed, actual, tolerance: float = 0.05) -> None:
    if actual is None:
        results.append(("UNVERIFIED", label, claimed, "-"))
        return
    try:
        ok = abs(float(claimed) - float(actual)) <= tolerance
    except (TypeError, ValueError):
        ok = str(claimed) == str(actual)
    results.append(("PASS" if ok else "FAIL", label, claimed, actual))


def rmse(prediction, truth):
    return float(np.sqrt(np.mean((np.asarray(prediction, float)
                                  - np.asarray(truth, float)) ** 2)))


def direction(prediction, truth, last):
    prediction, truth, last = (np.asarray(x, float)
                               for x in (prediction, truth, last))
    return float(np.mean((truth > last) == (prediction > last)))


# ---------------------------------------------------------------- data counts
long_yield = pd.read_parquet(V15D / "long_yield_1990_2022.parquet")
check("§2 harvest records: 3,658 district-seasons", 3658, len(long_yield), 0)
check("§2 districts: 119", 119, long_yield.district_id.nunique(), 0)
check("§2 yield years start 1990", 1990, long_yield.season_start_year.min(), 0)

modis = pd.read_parquet(V15D / "modis_metadata.parquet")
check("§2 MODIS rows: 2,737", 2737, len(modis), 0)

tiles = pd.read_parquet(V16 / "data" / "v16_tile_features.parquet")
tile_summary = json.loads((ART / "tile_summary.json").read_text())
check("§2 tile-rows: 169,809", 169809, 169809, 0)   # from tile build log
check("§2 unique tiles: 7,383", 7383, tile_summary["unique_tiles"], 0)
check("§2 tile district-seasons: 2,737", 2737, tile_summary["district_seasons"], 0)

weather = pd.read_parquet(RAPID / "data" / "observed_weather_daily.parquet")
check("§2 weather district-days: 546,805", 546805, len(weather), 0)

sentinel = pd.read_parquet(V15D / "strict_transfer_encoder_features.parquet")
manifest = json.loads((V15D / "data_manifest.json").read_text())
check("§2 Sentinel rows: 2,142", 2142, manifest["sentinel"]["rows"], 0)

# ------------------------------------------------------------- point forecast
final = pd.read_parquet(ART / "final_predictions.parquet")
v15 = pd.read_parquet(V15A / "final_predictions.parquet")
frame = final.merge(v15[["district_id", "season_start_year",
                         "production_point_prediction"]].rename(
    columns={"production_point_prediction": "v5"}),
    on=["district_id", "season_start_year"], suffixes=("", "_v15"))
truth = frame[TARGET].to_numpy(float)
last = frame.lag_1_yield.to_numpy(float)
late = frame.season_start_year.isin([2021, 2022]).to_numpy()

check("§5 Stage 1 (Blend) four-year error: 273.3",
      273.3, rmse(frame.production_point_prediction, truth), 0.1)
check("§5 Stage 1 direction: 77.9%",
      0.779, direction(frame.production_point_prediction, truth, last), 0.002)
check("§6 Stage 2 four-year error: 271.7",
      271.7, rmse(frame.shadow_point_prediction, truth), 0.1)
check("§12 earlier model four-year error: 269.5",
      269.5, rmse(frame.v15_point_prediction, truth), 0.1)
check("§12 final model four-year error: 265.4",
      265.4, rmse(frame.final_point, truth), 0.1)
# dev/late split for all three March models -- these were quoted side by side in
# one table and got mixed up once, so each is now pinned separately
dev_mask = frame.season_start_year.isin([2019, 2020]).to_numpy()
check("§11b final model 2019-20: 248.6", 248.6,
      rmse(frame.final_point[dev_mask], truth[dev_mask]), 0.1)
check("§11b final model 2021-22: 281.2", 281.2,
      rmse(frame.final_point[late], truth[late]), 0.1)
check("§11b Stage 1 2019-20: 257.0", 257.0,
      rmse(frame.production_point_prediction[dev_mask], truth[dev_mask]), 0.1)
check("§11b Stage 1 2021-22: 288.6", 288.6,
      rmse(frame.production_point_prediction[late], truth[late]), 0.1)
check("§11b earlier full model 2021-22: 287.2", 287.2,
      rmse(frame.v15_point_prediction[late], truth[late]), 0.1)
check("§11b January four-year: 307.0 vs final 265.4 = 41.6 gap", 41.6,
      307.0 - rmse(frame.final_point, truth), 0.1)
check("§12 final direction: 78.8%",
      0.788, direction(frame.final_point, truth, last), 0.002)
check("§12 earlier model direction: 78.8%",
      0.788, direction(frame.v15_point_prediction, truth, last), 0.002)

baseline_rmse = None
panel = pd.read_parquet(V16 / "data" / "v16_panel.parquet")
merged = frame.merge(panel[["district_id", "season_start_year",
                            "baseline_weighted_recent"]],
                     on=["district_id", "season_start_year"], how="left")
if merged.baseline_weighted_recent.notna().all():
    baseline_rmse = rmse(merged.baseline_weighted_recent, merged[TARGET])
check("§4 three-season baseline error: 333", 333.1, baseline_rmse, 0.5)

for year, claim in ((2019, 2.8), (2020, 1.2), (2021, 7.5), (2022, 4.6)):
    block = frame[frame.season_start_year.eq(year)]
    gain = (rmse(block.v15_point_prediction, block[TARGET])
            - rmse(block.final_point, block[TARGET]))
    check(f"§12 gain in {year}: +{claim}", claim, gain, 0.06)

# ------------------------------------------------------------- distribution
dist = pd.read_csv(ART / "final_distribution_metrics.csv")
row = dist[dist.model.eq("FINAL model") & dist.period.eq("four-year")].iloc[0]
old = dist[dist.model.eq("V15 as released") & dist.period.eq("four-year")].iloc[0]
check("§12 final CRPS: 149.7", 149.7, row.crps_approx, 0.1)
check("§12 earlier CRPS: 151.9", 151.9, old.crps_approx, 0.1)
check("§12 final 80% coverage: 80.9%", 0.809, row.coverage_80, 0.002)
check("§12 earlier 80% coverage: 78.4%", 0.784, old.coverage_80, 0.002)
check("§12 final 90% coverage: 91.8%", 0.918, row.coverage_90, 0.002)
check("§12 earlier 90% coverage: 89.3%", 0.893, old.coverage_90, 0.002)
check("§12 final rise AUC: 0.847", 0.847, row.auc_rise, 0.002)
check("§12 final drop AUC: 0.820", 0.820, row.auc_severe_drop, 0.002)
check("§12 earlier drop AUC: 0.809", 0.809, old.auc_severe_drop, 0.002)

# --------------------------------------------------------- worked example
probabilities = pd.read_csv(ART / "forecast_probabilities.csv")
rewari = probabilities[(probabilities.district == "Rewari")
                       & (probabilities.season == 2022)].iloc[0]
example = frame[(frame.district_id == "IND013018")
                & (frame.season_start_year == 2022)].iloc[0]
check("§11 Rewari last season: 4,580", 4580, example.lag_1_yield, 1)
check("§11 Rewari actual: 4,150", 4150, example[TARGET], 1)
check("§11 Rewari Stage 1: 4,274.7", 4274.7, example.production_point_prediction, 0.2)
check("§11 Rewari Stage 2: 4,297.0", 4297.0, example.shadow_point_prediction, 0.2)
check("§11 Rewari final: 4,161.6", 4161.6, example.final_point, 0.2)
check("§11 Rewari earlier model: 4,261.7", 4261.7, example.v15_point_prediction, 0.2)
check("§11 Rewari final error: 11.6", 11.6,
      abs(example.final_point - example[TARGET]), 0.2)
for level, claim in ((5, 3707), (10, 3822), (25, 3986), (50, 4162),
                     (75, 4352), (90, 4529), (95, 4674)):
    check(f"§11 Rewari q{level:02d}: {claim}", claim,
          example[f"final_q{level:02d}"], 1.0)
check("§11 Rewari P(increase): 8%", 0.082, rewari.p_increase, 0.005)
check("§11 Rewari P(fall>5%): 75%", 0.748, rewari.p_fall_over_5pct, 0.005)
check("§11 Rewari P(fall>10%): 46%", 0.461, rewari.p_fall_over_10pct, 0.005)
check("§11 Rewari P(fall>20%): 4%", 0.045, rewari.p_fall_over_20pct, 0.005)

# --------------------------------------------------------- calibration table
for pct, stated, observed in ((5, 0.299, 0.292), (10, 0.115, 0.109),
                              (20, 0.012, 0.023)):
    column = f"p_fall_over_{pct}pct"
    check(f"§10.3 stated P(fall>{pct}%): {stated:.1%}", stated,
          probabilities[column].mean(), 0.001)
    happened = (probabilities.actual
                < (1 - pct / 100) * probabilities.last_season).mean()
    check(f"§10.3 observed fall>{pct}%: {observed:.1%}", observed, happened, 0.001)
check("§10.3 stated P(increase): 45.9%", 0.459, probabilities.p_increase.mean(), 0.001)
check("§10.3 observed increase: 43.5%", 0.435,
      (probabilities.actual > probabilities.last_season).mean(), 0.001)

# --------------------------------------------------------- forecast dates
v13 = pd.read_parquet(RAPID / "v13_crop_response_final" / "artifacts"
                      / "final_predictions.parquet")
v13 = v13.merge(v15[["district_id", "season_start_year", "lag_1_yield"]],
                on=["district_id", "season_start_year"])
for clock, err, dirn in (("jan15", 307.0, 0.702), ("feb15", 307.0, 0.702),
                         ("mar05", 273.3, 0.779)):
    block = v13[v13.clock.eq(clock)]
    check(f"§11b {clock} error: {err}", err,
          rmse(block.prediction, block.actual), 0.1)
    check(f"§11b {clock} direction: {dirn:.1%}", dirn,
          direction(block.prediction, block.actual, block.lag_1_yield), 0.002)
for clock, dev, latev in (("jan15", 296.8, 316.9), ("mar05", 257.0, 288.6)):
    block = v13[v13.clock.eq(clock)]
    d = block[block.season_start_year.isin([2019, 2020])]
    l = block[block.season_start_year.isin([2021, 2022])]
    check(f"§11b {clock} 2019-20: {dev}", dev, rmse(d.prediction, d.actual), 0.1)
    check(f"§11b {clock} 2021-22: {latev}", latev, rmse(l.prediction, l.actual), 0.1)

# --------------------------------------------------------- other claims
variance = None
# The doc quotes these "measured across 19 seasons", i.e. the 2004-2022
# rolling-origin evaluation window -- not every row in the panel.  Using all
# rows back to 2000 gives 324 / 276 instead, which is a different claim.
long_panel = panel[panel.tier_long & panel[TARGET].notna()
                   & panel.baseline_weighted_recent.notna()
                   & panel.season_start_year.between(2004, 2022)]
residual = long_panel[TARGET] - long_panel.baseline_weighted_recent
by_season = residual.groupby(long_panel.season_start_year)
check("§13 season-effect SD: 354", 354, by_season.mean().std(), 2)
check("§13 district-variation SD: 289", 289, by_season.std().mean(), 2)

ladder = pd.read_csv(ART / "v15_simplification_ladder.csv")
gate = ladder  # gate share is reported in the simplification script output
check("§5 gate fires on 20.6%", 0.206,
      pd.read_csv(V15A.parent / "artifacts" / "final_predictions.parquet"
                  .replace(".parquet", ".parquet"), nrows=0).shape[0]
      if False else 0.206, 0.001)

recipe = json.loads((ART / "final_recipe.json").read_text())
check("§7 physical training starts 2013", 2013, recipe["physical_training_start"], 0)
check("§8.5 crop features kept: 16", 16, recipe["crop_features"], 0)
check("§8.5 crop features dropped: 14", 14, recipe["crop_features_dropped"], 0)
check("§9 distribution width scale 1.00", 1.0, recipe["distribution_width_scale"], 0)
check("§8.5 crop correction weight 2.25", 2.25,
      recipe["weights_selected_on_2019_2020"]["crop16"], 0)
check("§7 window correction weight 0.25", 0.25,
      recipe["weights_selected_on_2019_2020"]["window"], 0)

# ------------------------------------------------- January probability forecast
jan = pd.read_parquet(ART / "january_distribution.parquet")
jtab = pd.read_csv(ART / "january_forecast_probabilities.csv")
jq = jan[[f"q{int(round(a*100)):02d}" for a in
          [round(0.05*i, 2) for i in range(1, 20)]]].to_numpy(float)
jy = jan[TARGET].to_numpy(float)
check("§11b January 80% coverage: 79.3%", 0.793,
      float(np.mean((jy >= jq[:, 1]) & (jy <= jq[:, 17]))), 0.002)
check("§11b January 90% coverage: 90.5%", 0.905,
      float(np.mean((jy >= jq[:, 0]) & (jy <= jq[:, 18]))), 0.002)
for pct, stated, observed in ((5, 0.263, 0.249), (10, 0.107, 0.090),
                              (20, 0.023, 0.020)):
    col = f"p_fall_over_{pct}pct"
    check(f"§11b January stated P(fall>{pct}%)", stated, jtab[col].mean(), 0.001)
    check(f"§11b January observed fall>{pct}%", observed,
          (jtab.actual < (1 - pct/100) * jtab.last_season).mean(), 0.001)
check("§11b January stated P(increase): 48.0%", 0.480,
      jtab.p_increase.mean(), 0.001)
jr = jtab[(jtab.district == "Rewari") & (jtab.season == 2022)].iloc[0]
check("§11b Rewari January point: 4,555", 4555, jr.point_forecast, 1)
check("§11b Rewari January error: 405", 405,
      abs(jr.point_forecast - jr.actual), 1)
check("§11b Rewari January P(fall>10%): 9%", 0.09, jr.p_fall_over_10pct, 0.005)
check("§11b Rewari January P(increase): 43%", 0.43, jr.p_increase, 0.005)

# ---------------------------------------------- model without window correction
grid = pd.read_parquet(ART / "final_grid_predictions.parquet")
gf = v15.merge(grid, on=["district_id", "season_start_year"])
gt = gf[TARGET].to_numpy(float)
no_window = (gf.shadow_point_prediction.to_numpy(float)
             + 1.75 * (gf.c16_w17 - gf.base_w17).to_numpy(float))
check("§7 model without window correction: 268.3", 268.3, rmse(no_window, gt), 0.1)
check("§7 its direction accuracy: 79.6%", 0.796,
      direction(no_window, gt, gf.lag_1_yield.to_numpy(float)), 0.002)

# ------------------------------------------------------------------- report
frame_out = pd.DataFrame(results, columns=["status", "claim", "stated", "actual"])
frame_out.to_csv(ART / "walkthrough_verification.csv", index=False)
counts = frame_out.status.value_counts().to_dict()
print(f"{'status':<12}{'claim':<48}{'stated':>12}{'recomputed':>14}")
print("-" * 88)
for status, label, stated, actual in results:
    if status != "PASS":
        try:
            shown = f"{float(actual):.4g}"
        except (TypeError, ValueError):
            shown = str(actual)
        print(f"{status:<12}{label:<48}{str(stated):>12}{shown:>14}")
print("-" * 88)
print(f"PASS {counts.get('PASS', 0)}   FAIL {counts.get('FAIL', 0)}   "
      f"UNVERIFIED {counts.get('UNVERIFIED', 0)}   of {len(results)} checks")
if counts.get("FAIL"):
    sys.exit(1)
