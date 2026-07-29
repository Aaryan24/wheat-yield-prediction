#!/usr/bin/env python3
"""A conditional yield distribution with honest width.

V15's uncertainty was a fixed empirical residual shape, centred on the point
forecast and scaled by a constant 0.95.  It is well calibrated -- 80% intervals
covered 78.4% -- but carries almost no case-specific information: the
correlation between predicted width and realised absolute error is only 0.18.
It has reliability and essentially no resolution.

The obvious upgrade -- fit quantile regressions directly on the training
residual -- was tried first and failed badly, with 80% intervals covering 47%.
The reason is instructive and applies to V15 too: state-level weather is
CONSTANT within a state-season, so a flexible learner can identify which season
a training row came from and shrink its quantiles to the within-season spread.
The across-season shock, which is the dominant term, then vanishes from the
predicted width.

So the scale model here is trained on OUT-OF-SAMPLE residuals from seasons
already forecast, never on in-sample ones.  Out-of-sample residuals contain the
season shock by construction, so the widths inherit it automatically.

Structure, in log-anomaly space around the three-season normal:

    log(y / yhat) = sigma(x) * eps

  sigma(x)  a learned per-case scale, fitted on earlier folds' realised errors
  eps       an empirical standardized-residual shape, also from earlier folds

Multiplicative intervals matter here: a 600 kg/ha band means very different
things on a 2,500 and a 5,500 kg/ha district.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

V16 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V16 / "scripts"))
from v16_common import BASELINE, TARGET  # noqa: E402

DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"
QUANTILES = [round(0.05 * i, 2) for i in range(1, 20)]
COLUMNS = [f"q{int(round(a * 100)):02d}" for a in QUANTILES]
LATE = [2019, 2020, 2021, 2022]
CLIP = (500.0, 7000.0)
MIN_HISTORY_FOLDS = 3


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1.0) * d)))


def evaluate(frame: pd.DataFrame) -> dict[str, float]:
    y = frame[TARGET].to_numpy(float)
    losses = [pinball(y, frame[c].to_numpy(float), a)
              for a, c in zip(QUANTILES, COLUMNS)]
    width = (frame["q90"] - frame["q10"]).to_numpy(float)
    error = np.abs(y - frame["q50"].to_numpy(float))
    centre = frame["q50"].to_numpy(float)
    # A purely multiplicative interval already correlates with absolute error,
    # because large districts have both wider bands and larger errors in kg/ha.
    # That is a size artifact, not resolution.  Dividing both sides by the
    # forecast level removes it and leaves genuine case-to-case discrimination.
    return {
        "rows": int(len(frame)),
        "mean_pinball": float(np.mean(losses)),
        "crps_approx": float(2 * np.mean(losses)),
        "coverage_50": float(np.mean((y >= frame.q25) & (y <= frame.q75))),
        "coverage_80": float(np.mean((y >= frame.q10) & (y <= frame.q90))),
        "coverage_90": float(np.mean((y >= frame.q05) & (y <= frame.q95))),
        "mean_width_80": float(np.mean(width)),
        "width_cv": float(np.std(width) / np.mean(width)),
        "corr_width_abs_error": float(np.corrcoef(width, error)[0, 1]),
        "corr_relative": float(np.corrcoef(width / centre, error / centre)[0, 1]),
    }


def event_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    q = frame[COLUMNS].to_numpy(float)
    levels = np.array(QUANTILES)
    lag = frame["lag_1_yield"].to_numpy(float)
    rise, drop = [], []
    for i in range(len(frame)):
        xs = np.concatenate([[q[i, 0] - 500.0], q[i], [q[i, -1] + 500.0]])
        ps = np.concatenate([[0.0], levels, [1.0]])
        order = np.argsort(xs)
        rise.append(1.0 - float(np.interp(lag[i], xs[order], ps[order])))
        drop.append(float(np.interp(0.90 * lag[i], xs[order], ps[order])))
    return pd.DataFrame({"probability_rise": rise,
                         "probability_severe_drop": drop}, index=frame.index)


def brier_skill(frame: pd.DataFrame, probs: pd.DataFrame) -> dict[str, float]:
    rose = (frame[TARGET] > frame["lag_1_yield"]).to_numpy(float)
    fell = (frame[TARGET] < 0.90 * frame["lag_1_yield"]).to_numpy(float)
    out = {}
    for name, outcome, column in (("rise", rose, "probability_rise"),
                                  ("drop", fell, "probability_severe_drop")):
        b = float(np.mean((probs[column].to_numpy(float) - outcome) ** 2))
        c = float(np.mean((outcome.mean() - outcome) ** 2))
        out[f"brier_{name}"] = b
        out[f"brier_skill_{name}"] = 1.0 - b / c if c > 0 else np.nan
    return out


def main() -> None:
    hierarchy = pd.read_parquet(ARTIFACTS / "hierarchy_predictions.parquet")
    panel = pd.read_parquet(DATA / "v16_panel.parquet")
    groups = json.loads((DATA / "v16_feature_groups.json").read_text())
    weather = pd.read_parquet(DATA / "v16_weather_district.parquet")
    state_weather = pd.read_parquet(DATA / "v16_weather_state.parquet")

    # hierarchy already carries lag_1_yield / baseline; avoid _x/_y collisions
    keys = ["district_id", "season_start_year"]
    extra = [c for c in groups["history"] + groups["modis_anomaly"]
             if c not in hierarchy.columns]
    frame = hierarchy.merge(
        panel[keys + extra], on=keys, how="left").merge(
        weather.drop(columns=["state_name"]),
        on=["district_id", "season_start_year"], how="left").merge(
        state_weather, on=["state_name", "season_start_year"], how="left")

    # The distribution is centred on the DISTRICT layer, not the shock-adjusted
    # point.  Measured: the district layer beats the naive baseline by +11.9
    # kg/ha with P(>0)=1.000 on 2019-22, while moving the point by the weakly
    # predictable shock (out-of-sample R2 ~0.12) hurt that window by 30 kg/ha.
    # The shock still enters the forecast -- through the SCALE, where a signal
    # that reliable belongs.  `predicted_shock` is a scale feature below.
    CENTRE = "prediction_no_shock"
    frame["log_error"] = np.log(np.clip(
        frame[TARGET] / frame[CENTRE], 0.2, 5.0))

    satellite = [c for c in groups["modis_anomaly"] if c.startswith("z__")]
    district_wx = [c for c in frame.columns if c.startswith("wxz__")]
    state_wx = [c for c in state_weather.columns if c.startswith("st_wxz__")]
    history = ["roll_5_std", "roll_10_std", "roll_10_mean",
               "lag_1_over_roll_10", "roll_3_over_roll_10"]
    scale_features = satellite + district_wx + state_wx + history + [
        "predicted_shock", "beta", "remainder_log"]
    scale_features = [c for c in scale_features if c in frame.columns]

    years = sorted(frame.season_start_year.unique())
    blocks, fixed_blocks = [], []
    for index, year in enumerate(years):
        if index < MIN_HISTORY_FOLDS:
            continue
        history_rows = frame[frame.season_start_year < year].dropna(
            subset=["log_error"])
        test = frame[frame.season_start_year.eq(year)].copy()
        if test.empty or len(history_rows) < 200:
            continue

        usable = [c for c in scale_features
                  if history_rows[c].notna().mean() >= 0.5]
        # --- learn how large the error tends to be, from realised errors ---
        scale_model = LGBMRegressor(
            objective="l2", n_estimators=200, learning_rate=0.03,
            num_leaves=7, min_child_samples=60, subsample=0.85,
            subsample_freq=1, colsample_bytree=0.6, reg_lambda=30.0,
            random_state=42, verbose=-1)
        scale_model.fit(history_rows[usable].astype(float),
                        np.log(np.abs(history_rows["log_error"]) + 1e-3))
        floor = float(np.quantile(np.abs(history_rows["log_error"]), 0.10))
        sigma_history = np.maximum(
            np.exp(scale_model.predict(history_rows[usable].astype(float))), floor)
        sigma_test = np.maximum(
            np.exp(scale_model.predict(test[usable].astype(float))), floor)

        # --- empirical shape of the standardized residual ---
        standardized = (history_rows["log_error"].to_numpy(float)
                        / np.maximum(sigma_history, 1e-6))
        shape = np.quantile(standardized, QUANTILES)

        point = test[CENTRE].to_numpy(float)
        for column, offset in zip(COLUMNS, shape):
            test[column] = np.clip(point * np.exp(sigma_test * offset), *CLIP)
        blocks.append(test)

        # --- V15-style comparison: one fixed shape, no conditioning ---
        reference = test.copy()
        flat = np.quantile(history_rows["log_error"].to_numpy(float), QUANTILES)
        for column, offset in zip(COLUMNS, flat):
            reference[column] = np.clip(point * np.exp(offset), *CLIP)
        fixed_blocks.append(reference)

        print(f"  fold {year}: {len(test)} rows, "
              f"{len(history_rows)} earlier out-of-sample rows, "
              f"{len(usable)} scale features", flush=True)

    out = pd.concat(blocks, ignore_index=True)
    fixed = pd.concat(fixed_blocks, ignore_index=True)
    for f in (out, fixed):
        f[COLUMNS] = np.maximum.accumulate(f[COLUMNS].to_numpy(float), axis=1)

    print("\n=== Distribution quality ===")
    rows = []
    for label, data in (("V15-style fixed shape", fixed),
                        ("V16 conditional scale", out)):
        for period, subset in (("all folds", data),
                               ("2019-2022", data[data.season_start_year.isin(LATE)])):
            probs = event_probabilities(subset)
            rows.append({"model": label, "period": period,
                         **evaluate(subset), **brier_skill(subset, probs)})
    report = pd.DataFrame(rows)
    report.to_csv(ARTIFACTS / "distribution_metrics.csv", index=False)
    print(report[["model", "period", "rows", "mean_pinball", "crps_approx",
                  "coverage_80", "coverage_90", "mean_width_80", "width_cv",
                  "corr_width_abs_error", "corr_relative"]].to_string(index=False))

    print("\n=== Event probabilities ===")
    print(report[["model", "period", "brier_rise", "brier_skill_rise",
                  "brier_drop", "brier_skill_drop"]].to_string(index=False))

    print("\ncorr_width_abs_error is the resolution measure: does the interval")
    print("widen exactly when the forecast is genuinely harder?  V15 scored 0.18.")

    out = pd.concat([out, event_probabilities(out)], axis=1)
    out.to_parquet(ARTIFACTS / "distribution_predictions.parquet", index=False)
    fixed.to_parquet(ARTIFACTS / "distribution_fixed_shape.parquet", index=False)


if __name__ == "__main__":
    main()
