#!/usr/bin/env python3
"""Season-shock hierarchy: model the biggest term in the problem first.

Measured on nineteen rolling-origin folds:

    sd of the season-mean yield shock   354 kg/ha
    sd of district variation within a season   289 kg/ha

The shared "what kind of season is this" term is LARGER than the district
detail, yet V15 spent its entire crop-Transformer budget on district detail and
left the season term to a three-season history baseline.  V15 did build a
state-shock stage but rejected it for failing to beat V14 as a standalone
predictor -- the wrong test, since a shock layer is a component, not a rival.

The decomposition here is

    log(y / n) = alpha_d + beta_d * g_(s,t) + e_(d,t)

with n the three-season weighted normal, g the shared state-season shock,
beta_d the district's shrunk sensitivity to it, and e the district-specific
remainder.  g is predicted from STATE-AGGREGATED satellite anomalies: averaging
over ~40 districts cancels most of the per-district satellite noise, so the
shared term is measured far more precisely than any single district's state.

Everything is fitted strictly on seasons before the target season.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V16 / "scripts"))
from v16_common import (  # noqa: E402
    BASELINE, TARGET, metrics, xgb_residual_predict, year_block_bootstrap,
)

DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"
TEST_YEARS = list(range(2014, 2023))  # weather-backed shock layer needs 2010+
LATE = [2019, 2020, 2021, 2022]
ANOMALY_CLIP = 0.60
MIN_TRAIN_YEARS = 4


def log_anomaly(frame: pd.DataFrame) -> np.ndarray:
    ratio = frame[TARGET].to_numpy(float) / frame[BASELINE].to_numpy(float)
    return np.log(np.clip(ratio, 0.3, 3.0))


def state_shock_table(train: pd.DataFrame) -> pd.DataFrame:
    """Observed shared shock per state and season, from training seasons only."""
    return (train.assign(_a=log_anomaly(train))
            .groupby(["state_name", "season_start_year"])["_a"]
            .mean().rename("shock").reset_index())


def district_exposure(train: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    """Shrunk per-district intercept and sensitivity to the shared shock.

    Districts with few seasons are pulled toward the population values
    (alpha = 0, beta = 1) so that a short history cannot produce a wild slope.
    """
    frame = train.assign(_a=log_anomaly(train)).merge(
        shocks, on=["state_name", "season_start_year"], how="left")
    rows = []
    for district, block in frame.groupby("district_id"):
        ok = block[["_a", "shock"]].notna().all(axis=1)
        block = block[ok]
        n = len(block)
        if n >= 3 and block["shock"].var() > 1e-9:
            beta_raw = float(np.cov(block["_a"], block["shock"])[0, 1]
                             / np.var(block["shock"], ddof=1))
            alpha_raw = float(block["_a"].mean() - beta_raw * block["shock"].mean())
        else:
            beta_raw, alpha_raw = 1.0, float(block["_a"].mean()) if n else 0.0
        beta = (n * beta_raw + 8.0) / (n + 8.0)
        alpha = (n / (n + 12.0)) * alpha_raw
        rows.append({"district_id": district,
                     "alpha": float(np.clip(alpha, -0.15, 0.15)),
                     "beta": float(np.clip(beta, 0.25, 1.75)),
                     "exposure_seasons": n})
    return pd.DataFrame(rows)


# Two pre-clock weather aggregates, chosen on physical grounds before fitting:
# winter rainfall (cloud, waterlogging, disease pressure) and the sunlight
# actually available for grain filling.  Both are measured within-state
# year-over-year at |r| = 0.61-0.76 in all three states independently.
#
# A RidgeCV over all 80 weather columns was tried first and scored R2 = -1.7
# out of sample: with only ~10 usable seasons, a wide model destroys a signal
# that a two-term model recovers.  The parsimony here is the result, not a
# stylistic choice.
SHOCK_FEATURES = ["st_wx_full_preclock_precip_sum", "st_wx_dec_feb_solar_mean"]


def fit_shock_model(train: pd.DataFrame, features: list[str]):
    """State fixed effects plus a two-term weather slope.

    Demeaning within state uses training seasons only, so the level of each
    state is learned from the past rather than from the target season.
    """
    frame = train.dropna(subset=["shock"] + features)
    if frame.season_start_year.nunique() < MIN_TRAIN_YEARS:
        return None
    centres = frame.groupby("state_name")[["shock"] + features].mean()
    x = np.column_stack(
        [np.ones(len(frame))]
        + [frame[f].to_numpy(float) - centres.loc[frame.state_name, f].to_numpy()
           for f in features])
    y = (frame["shock"].to_numpy(float)
         - centres.loc[frame.state_name, "shock"].to_numpy())
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta, centres, features


def apply_shock_model(fitted, design: pd.DataFrame) -> np.ndarray:
    beta, centres, features = fitted
    known = design.state_name.isin(centres.index)
    out = np.zeros(len(design))
    if not known.any():
        return out
    block = design[known]
    x = np.column_stack(
        [np.ones(len(block))]
        + [block[f].to_numpy(float) - centres.loc[block.state_name, f].to_numpy()
           for f in features])
    x = np.where(np.isfinite(x), x, 0.0)
    out[known.to_numpy()] = (x @ beta
                             + centres.loc[block.state_name, "shock"].to_numpy())
    return out


def main() -> None:
    panel = pd.read_parquet(DATA / "v16_panel.parquet")
    groups = json.loads((DATA / "v16_feature_groups.json").read_text())
    panel = panel[panel.tier_long & panel[TARGET].notna()
                  & panel[BASELINE].notna()].copy()

    district_features = groups["history"] + groups["modis_anomaly"]
    state_weather = pd.read_parquet(DATA / "v16_weather_state.parquet")

    blocks, shock_rows = [], []
    for year in TEST_YEARS:
        train = panel[panel.season_start_year < year]
        test = panel[panel.season_start_year.eq(year)].copy()
        if test.empty or train.season_start_year.nunique() < MIN_TRAIN_YEARS:
            continue

        shocks = state_shock_table(train)
        exposure = district_exposure(train, shocks)

        # shock design: observed shock joined to pre-clock state weather
        design_all = state_weather.merge(
            pd.concat([shocks, state_shock_table(test)]),
            on=["state_name", "season_start_year"], how="left")
        train_design = design_all[design_all.season_start_year < year]
        test_design = design_all[design_all.season_start_year.eq(year)]
        fitted = fit_shock_model(train_design, SHOCK_FEATURES)
        if fitted is None or test_design.empty:
            predicted_shock = pd.DataFrame({
                "state_name": test.state_name.unique(),
                "season_start_year": year,
                "predicted_shock": 0.0})
        else:
            predicted_shock = pd.DataFrame({
                "state_name": test_design.state_name.values,
                "season_start_year": test_design.season_start_year.values,
                "predicted_shock": apply_shock_model(fitted, test_design),
            })
        actual = state_shock_table(test).rename(columns={"shock": "actual_shock"})
        shock_rows.append(predicted_shock.merge(
            actual, on=["state_name", "season_start_year"], how="left"))

        # district remainder after removing the shared shock
        train_full = train.merge(shocks, on=["state_name", "season_start_year"],
                                 how="left").merge(exposure, on="district_id",
                                                   how="left")
        train_full["_a"] = log_anomaly(train_full)
        train_full["remainder"] = (train_full["_a"] - train_full["alpha"]
                                   - train_full["beta"] * train_full["shock"])
        train_fit = train_full.dropna(subset=["remainder"]).copy()
        # reuse the shared XGBoost by expressing the remainder in kg/ha terms
        train_fit[TARGET] = (train_fit[BASELINE]
                             * np.exp(train_fit["remainder"].clip(-ANOMALY_CLIP,
                                                                  ANOMALY_CLIP)))
        remainder_pred = xgb_residual_predict(train_fit, test, district_features)
        remainder_log = np.log(np.clip(
            remainder_pred / test[BASELINE].to_numpy(float), 0.3, 3.0))

        block = test[["district_id", "state_name", "season_start_year",
                      TARGET, "lag_1_yield", BASELINE]].copy()
        block = block.merge(predicted_shock, on=["state_name", "season_start_year"],
                            how="left").merge(exposure, on="district_id", how="left")
        block["alpha"] = block["alpha"].fillna(0.0)
        block["beta"] = block["beta"].fillna(1.0)
        block["remainder_log"] = remainder_log
        blocks.append(block)
        print(f"  fold {year}: {len(test)} rows, "
              f"{train.season_start_year.nunique()} train seasons", flush=True)

    out = pd.concat(blocks, ignore_index=True)
    shock_eval = pd.concat(shock_rows, ignore_index=True).dropna()

    print("\n=== Can the shared season shock be predicted at all? ===")
    pred, act = shock_eval.predicted_shock, shock_eval.actual_shock
    ss_res = float(((act - pred) ** 2).sum())
    ss_tot = float(((act - act.mean()) ** 2).sum())
    print(f"  state-season rows {len(shock_eval)}   corr {np.corrcoef(pred, act)[0,1]:.3f}"
          f"   R2 vs mean {1 - ss_res/ss_tot:+.3f}")
    print(f"  sd of actual shock {act.std():.4f} log units "
          f"(~{act.std()*4500:.0f} kg/ha)")
    print(f"  sd of residual     {(act-pred).std():.4f} log units "
          f"(~{(act-pred).std()*4500:.0f} kg/ha)")
    shock_eval.to_csv(ARTIFACTS / "shock_prediction.csv", index=False)

    def reconstruct(frame: pd.DataFrame, w_shock: float, w_rem: float) -> np.ndarray:
        total = (frame["alpha"] + w_shock * frame["beta"] * frame["predicted_shock"]
                 + w_rem * frame["remainder_log"])
        return np.clip(frame[BASELINE]
                       * np.exp(total.clip(-ANOMALY_CLIP, ANOMALY_CLIP)), 500, 7000)

    GRID = [(s, r) for s in (0.0, 0.25, 0.5, 0.75, 1.0)
            for r in (0.0, 0.5, 0.75, 1.0)]
    PRIOR = (0.5, 0.75)

    # ---------------------------------------------------------------------
    # Prequential weight selection.  The combination weights for season t are
    # chosen using ONLY the out-of-sample errors of seasons already forecast
    # (< t).  Reading them off the full-period grid, as a first pass did, is
    # selection on the test set and inflates the apparent gain.
    # ---------------------------------------------------------------------
    out = out.sort_values("season_start_year").reset_index(drop=True)
    chosen, predictions = [], np.full(len(out), np.nan)
    for year in sorted(out.season_start_year.unique()):
        earlier = out[out.season_start_year < year]
        if earlier.season_start_year.nunique() < 2:
            weights = PRIOR
        else:
            scores = []
            for w_shock, w_rem in GRID:
                pred = reconstruct(earlier, w_shock, w_rem)
                scores.append((float(np.sqrt(np.mean(
                    (pred - earlier[TARGET].to_numpy(float)) ** 2))),
                    w_shock, w_rem))
            _, w_shock, w_rem = min(scores)
            weights = (w_shock, w_rem)
        mask = (out.season_start_year == year).to_numpy()
        predictions[mask] = reconstruct(out[mask], *weights)
        chosen.append({"season_start_year": int(year),
                       "w_shock": weights[0], "w_remainder": weights[1]})
    out["prediction"] = predictions
    out["prediction_no_shock"] = reconstruct(out, 0.0, 1.0)
    out["prediction_naive"] = out[BASELINE]

    print("\n=== Weights chosen prequentially (from earlier seasons only) ===")
    print(pd.DataFrame(chosen).to_string(index=False))

    print("\n=== Hierarchy results, honest weight selection ===")
    rows = []
    for label, column in (("naive weighted history", "prediction_naive"),
                          ("district layer only (no shock)", "prediction_no_shock"),
                          ("full hierarchy with shock layer", "prediction")):
        for period, subset in (("2014-2022 (9 folds)", out),
                               ("2019-2022 (V15 window)",
                                out[out.season_start_year.isin(LATE)])):
            rows.append({"model": label, "period": period,
                         **metrics(subset, column)})
    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "hierarchy_metrics.csv", index=False)
    print(report[["model", "period", "rows", "rmse", "mae", "bias",
                  "direction_accuracy"]].to_string(index=False))

    print("\n=== Season-resampled bootstrap vs the naive baseline ===")
    boot = []
    for label, column in (("district layer only", "prediction_no_shock"),
                          ("full hierarchy", "prediction")):
        for period, subset in (("2014-2022", out),
                               ("2019-2022", out[out.season_start_year.isin(LATE)])):
            b = year_block_bootstrap(subset, column, "prediction_naive")
            boot.append({"model": label, "period": period, **b})
            print(f"  {label:<22}{period:<12} gain {b['mean_gain']:+7.1f} "
                  f"[{b['p025']:+7.1f}, {b['p975']:+7.1f}] "
                  f"P(>0)={b['probability_positive']:.3f}")
    pd.DataFrame(boot).to_csv(ARTIFACTS / "hierarchy_bootstrap.csv", index=False)

    print("\n=== Per-season: where does the shock layer earn its keep? ===")
    per_year = pd.DataFrame([{
        "season_start_year": int(year),
        "season_shock_kg": float((block[TARGET] - block[BASELINE]).mean()),
        "naive_rmse": metrics(block, "prediction_naive")["rmse"],
        "no_shock_rmse": metrics(block, "prediction_no_shock")["rmse"],
        "hierarchy_rmse": metrics(block, "prediction")["rmse"],
    } for year, block in out.groupby("season_start_year")])
    per_year["gain_from_shock"] = (per_year.no_shock_rmse
                                   - per_year.hierarchy_rmse)
    per_year.to_csv(ARTIFACTS / "hierarchy_per_year.csv", index=False)
    print(per_year.to_string(index=False))

    out.to_parquet(ARTIFACTS / "hierarchy_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
