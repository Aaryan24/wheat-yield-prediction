#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    model: object


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def _predict_prob(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    score = np.asarray(model.decision_function(x), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-score))


def _in_window(valid_date: pd.Timestamp, season_year: int, window: str) -> bool:
    dec1 = pd.Timestamp(year=season_year, month=12, day=1)
    jan31 = pd.Timestamp(year=season_year + 1, month=1, day=31)
    feb1 = pd.Timestamp(year=season_year + 1, month=2, day=1)
    feb_end = pd.Timestamp(year=season_year + 1, month=3, day=1) - pd.Timedelta(days=1)
    mar1 = pd.Timestamp(year=season_year + 1, month=3, day=1)
    mar31 = pd.Timestamp(year=season_year + 1, month=3, day=31)
    apr1 = pd.Timestamp(year=season_year + 1, month=4, day=1)
    apr30 = pd.Timestamp(year=season_year + 1, month=4, day=30)
    if window == "vegetative_dec_jan":
        return bool(dec1 <= valid_date <= jan31)
    if window == "grainfill_feb_apr":
        return bool(feb1 <= valid_date <= apr30)
    if window == "feb":
        return bool(feb1 <= valid_date <= feb_end)
    if window == "mar":
        return bool(mar1 <= valid_date <= mar31)
    if window == "apr":
        return bool(apr1 <= valid_date <= apr30)
    raise ValueError(f"Unknown window: {window}")


def _daily_short_lead_frame(weather_path: Path, max_lead: int) -> pd.DataFrame:
    cols = ["district_id", "issue_date", "lead_day", "tmax_mean", "tmin_mean", "tp_mean"]
    df = pd.read_parquet(weather_path, columns=cols)
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df["lead_day"] = pd.to_numeric(df["lead_day"], errors="coerce").astype(int)
    df = df[(df["lead_day"] >= 1) & (df["lead_day"] <= int(max_lead))].copy()
    df["valid_date"] = df["issue_date"] + pd.to_timedelta(df["lead_day"], unit="D")
    df["tmax_c"] = pd.to_numeric(df["tmax_mean"], errors="coerce") - 273.15
    df["tmin_c"] = pd.to_numeric(df["tmin_mean"], errors="coerce") - 273.15
    df["tmean_c"] = 0.5 * (df["tmax_c"] + df["tmin_c"])
    df["tp_mm"] = pd.to_numeric(df["tp_mean"], errors="coerce")
    # Multiple short-lead forecasts can map to the same valid day; average them as a near-observed proxy.
    return (
        df.groupby(["district_id", "valid_date"], as_index=False)[["tmax_c", "tmin_c", "tmean_c", "tp_mm"]]
        .mean()
        .dropna()
    )


def build_observed_proxy_features(
    *,
    weather_dir: Path,
    district_ids: Sequence[str],
    season_years: Sequence[int],
    train_years: Sequence[int],
    max_lead: int,
) -> pd.DataFrame:
    daily_frames: List[pd.DataFrame] = []
    for year in sorted({int(y) for y in season_years}):
        path = weather_dir / f"s2s_district_daily_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing weather file: {path}")
        df = _daily_short_lead_frame(path, max_lead=max_lead)
        df["season_year"] = int(year)
        daily_frames.append(df)
    daily = pd.concat(daily_frames, ignore_index=True)
    daily = daily[daily["district_id"].astype(str).isin([str(d) for d in district_ids])].copy()
    daily["month_day"] = pd.to_datetime(daily["valid_date"]).dt.strftime("%m-%d")

    train_daily = daily[daily["season_year"].astype(int).isin([int(y) for y in train_years])].copy()
    clim = (
        train_daily.groupby(["district_id", "month_day"], as_index=False)
        .agg(
            clim_tmax_c=("tmax_c", "mean"),
            clim_tmin_c=("tmin_c", "mean"),
            clim_tmean_c=("tmean_c", "mean"),
            clim_tp_mm=("tp_mm", "mean"),
        )
    )
    work = daily.merge(clim, on=["district_id", "month_day"], how="left")
    for col in ["tmax_c", "tmin_c", "tmean_c", "tp_mm"]:
        c = f"clim_{col}"
        if c in work:
            work[c] = work[c].fillna(work[c].mean())
    work["tmax_anom_c"] = work["tmax_c"] - work["clim_tmax_c"]
    work["tmin_anom_c"] = work["tmin_c"] - work["clim_tmin_c"]
    work["tmean_anom_c"] = work["tmean_c"] - work["clim_tmean_c"]
    work["tp_anom_mm"] = work["tp_mm"] - work["clim_tp_mm"]

    rows: List[dict] = []
    windows = ["vegetative_dec_jan", "grainfill_feb_apr", "feb", "mar", "apr"]
    for (season_year, district_id), group in work.groupby(["season_year", "district_id"], dropna=False):
        row: Dict[str, object] = {"season_year": int(season_year), "district_id": str(district_id)}
        for window in windows:
            mask = [_in_window(pd.Timestamp(v), int(season_year), window) for v in group["valid_date"].tolist()]
            g = group[np.asarray(mask, dtype=bool)]
            prefix = f"obsproxy_{window}"
            row[f"{prefix}_n_days"] = int(len(g))
            if g.empty:
                for name in [
                    "tmax_mean_c",
                    "tmin_mean_c",
                    "tmean_mean_c",
                    "tmax_anom_mean_c",
                    "tmin_anom_mean_c",
                    "tmean_anom_mean_c",
                    "rain_sum_mm",
                    "rain_deficit_mm",
                    "heat_days_gt30",
                    "heat_days_gt32",
                    "heat_days_gt35",
                    "hot_nights_tmin_gt18",
                    "hot_nights_tmin_gt20",
                    "hdd_tmax_gt30",
                    "hdd_tmax_gt32",
                    "hdd_tmax_gt35",
                ]:
                    row[f"{prefix}_{name}"] = np.nan
                continue
            row[f"{prefix}_tmax_mean_c"] = float(g["tmax_c"].mean())
            row[f"{prefix}_tmin_mean_c"] = float(g["tmin_c"].mean())
            row[f"{prefix}_tmean_mean_c"] = float(g["tmean_c"].mean())
            row[f"{prefix}_tmax_anom_mean_c"] = float(g["tmax_anom_c"].mean())
            row[f"{prefix}_tmin_anom_mean_c"] = float(g["tmin_anom_c"].mean())
            row[f"{prefix}_tmean_anom_mean_c"] = float(g["tmean_anom_c"].mean())
            row[f"{prefix}_rain_sum_mm"] = float(g["tp_mm"].sum())
            row[f"{prefix}_rain_deficit_mm"] = float(np.maximum(0.0, -g["tp_anom_mm"]).sum())
            tmax = g["tmax_c"].to_numpy(dtype=np.float64)
            tmin = g["tmin_c"].to_numpy(dtype=np.float64)
            row[f"{prefix}_heat_days_gt30"] = int((tmax > 30.0).sum())
            row[f"{prefix}_heat_days_gt32"] = int((tmax > 32.0).sum())
            row[f"{prefix}_heat_days_gt35"] = int((tmax > 35.0).sum())
            row[f"{prefix}_hot_nights_tmin_gt18"] = int((tmin > 18.0).sum())
            row[f"{prefix}_hot_nights_tmin_gt20"] = int((tmin > 20.0).sum())
            row[f"{prefix}_hdd_tmax_gt30"] = float(np.maximum(0.0, tmax - 30.0).sum())
            row[f"{prefix}_hdd_tmax_gt32"] = float(np.maximum(0.0, tmax - 32.0).sum())
            row[f"{prefix}_hdd_tmax_gt35"] = float(np.maximum(0.0, tmax - 35.0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _add_priors(dy: pd.DataFrame, train_years: Sequence[int]) -> pd.DataFrame:
    out = dy.copy()
    train = out[out["season_year"].astype(int).isin([int(y) for y in train_years])]
    global_rate = float(train["actual_sign"].mean())
    district_rate = train.groupby("district_id")["actual_sign"].mean().to_dict()
    state_rate = train.groupby("state_name")["actual_sign"].mean().to_dict()
    out["manual_prior_district_rise_rate"] = out["district_id"].map(district_rate).fillna(global_rate)
    out["manual_prior_state_rise_rate"] = out["state_name"].map(state_rate).fillna(global_rate)
    out["manual_prior_global_rise_rate"] = global_rate
    out["manual_year_index"] = out["season_year"].astype(float) - float(min(train_years))
    out["manual_abs_lag1_delta_kg_per_ha"] = out["lag1_baseline_delta_kg_per_ha"].abs()
    return out


def _feature_cols(df: pd.DataFrame, feature_set: str) -> List[str]:
    manual = [c for c in df.columns if c.startswith("manual_")]
    state = [c for c in df.columns if c.startswith("state_") and c != "state_name"]
    lag = ["lag1_baseline_delta_kg_per_ha", "manual_abs_lag1_delta_kg_per_ha"]
    heat = [
        c
        for c in df.columns
        if c.startswith("obsproxy_")
        and (
            "heat" in c
            or "hdd" in c
            or "hot_nights" in c
            or "tmax" in c
            or "tmin" in c
            or "tmean" in c
        )
    ]
    rain = [c for c in df.columns if c.startswith("obsproxy_") and ("rain" in c or "deficit" in c)]
    counts = [c for c in df.columns if c.startswith("obsproxy_") and c.endswith("_n_days")]
    key = feature_set.lower()
    if key == "lag_state":
        cols = lag + manual + state
    elif key == "lag_heat":
        cols = lag + manual + state + heat + counts
    elif key == "lag_rain":
        cols = lag + manual + state + rain + counts
    elif key == "lag_heat_rain":
        cols = lag + manual + state + heat + rain + counts
    elif key == "observed_proxy_only":
        cols = state + heat + rain + counts
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    numeric_cols = [
        c
        for c in dict.fromkeys(cols)
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols


def _candidates(seed: int) -> List[Candidate]:
    feature_sets = ["lag_state", "lag_heat", "lag_rain", "lag_heat_rain", "observed_proxy_only"]
    out: List[Candidate] = []
    for fs in feature_sets:
        for c in [0.1, 0.3, 1.0, 3.0]:
            out.append(
                Candidate(
                    f"logreg_c{c:g}_{fs}",
                    fs,
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed),
                    ),
                )
            )
        for depth in [2, 3, 4, None]:
            out.append(
                Candidate(
                    f"extratrees_depth{depth}_{fs}",
                    fs,
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        ExtraTreesClassifier(
                            n_estimators=600,
                            max_depth=depth,
                            min_samples_leaf=4,
                            max_features=0.75,
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                )
            )
        for depth in [3, 5, 7]:
            out.append(
                Candidate(
                    f"rf_depth{depth}_{fs}",
                    fs,
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        RandomForestClassifier(
                            n_estimators=600,
                            max_depth=depth,
                            min_samples_leaf=5,
                            max_features=0.7,
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                )
            )
        for leaf in [7, 15, 31]:
            out.append(
                Candidate(
                    f"hgb_leaf{leaf}_{fs}",
                    fs,
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        HistGradientBoostingClassifier(
                            max_iter=250,
                            learning_rate=0.035,
                            max_leaf_nodes=leaf,
                            min_samples_leaf=20,
                            l2_regularization=2.0,
                            random_state=seed,
                        ),
                    ),
                )
            )
        if XGBClassifier is not None:
            for depth, lr in [(2, 0.035), (3, 0.025)]:
                out.append(
                    Candidate(
                        f"xgb_d{depth}_lr{lr:g}_{fs}",
                        fs,
                        XGBClassifier(
                            n_estimators=350,
                            max_depth=depth,
                            learning_rate=lr,
                            subsample=0.9,
                            colsample_bytree=0.8,
                            reg_lambda=10.0,
                            reg_alpha=0.5,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    )
                )
    return out


def _thresholds() -> np.ndarray:
    return np.round(np.linspace(0.05, 0.95, 91), 4)


def _score(
    *,
    mode: str,
    model_name: str,
    feature_set: str,
    prob: np.ndarray,
    dy: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> List[Dict[str, object]]:
    actual = dy["actual_sign"].to_numpy(dtype=np.int64)
    lag = dy["lag1_sign"].to_numpy(dtype=np.int64)
    target = actual if mode == "direct_actual_sign" else dy["flip_lag1"].to_numpy(dtype=np.int64)
    train_scores = []
    for thr in _thresholds():
        if mode == "direct_actual_sign":
            pred_train = (prob[train_mask] >= thr).astype(np.int64)
        else:
            pred_train = lag[train_mask].copy()
            flip = prob[train_mask] >= thr
            pred_train[flip] = 1 - pred_train[flip]
        train_scores.append((float(accuracy_score(actual[train_mask], pred_train)), float(thr)))
    best_thr = sorted(train_scores, key=lambda x: (x[0], -abs(x[1] - 0.5)), reverse=True)[0][1]
    rows: List[Dict[str, object]] = []
    for label, thr in [("thr05", 0.5), ("thr_train_best", best_thr)]:
        if mode == "direct_actual_sign":
            pred = (prob[test_mask] >= thr).astype(np.int64)
            flip = np.zeros_like(pred, dtype=bool)
        else:
            pred = lag[test_mask].copy()
            flip = prob[test_mask] >= thr
            pred[flip] = 1 - pred[flip]
        y = actual[test_mask]
        rows.append(
            {
                "mode": mode,
                "model": model_name,
                "feature_set": feature_set,
                "threshold_variant": label,
                "threshold": float(thr),
                "train_accuracy": float(
                    sorted(train_scores, key=lambda x: abs(x[1] - thr))[0][0]
                    if label == "thr_train_best"
                    else train_scores[np.where(_thresholds() == 0.5)[0][0]][0]
                ),
                "test_accuracy": float(accuracy_score(y, pred)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "test_auc": _safe_auc(target[test_mask], prob[test_mask]),
                "drop_recall": float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
                "rise_recall": float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
                "predicted_drop_rate": float((pred == 0).mean()),
                "test_flipped_rate": float(flip.mean()) if mode == "lag1_flip_router" else float("nan"),
            }
        )
    return rows


def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_years = [int(y) for y in args.train_years]
    test_years = [int(y) for y in args.test_years]
    cfg = _load_yaml(Path(args.config_data))
    weather_dir = Path(cfg["paths"]["weather_dir"])

    dy = pd.read_csv(args.sign_table).copy()
    if "actual_sign" not in dy:
        dy["actual_sign"] = (dy["actual_delta_kg_per_ha"] >= 0.0).astype(int)
    if "lag1_sign" not in dy:
        dy["lag1_sign"] = (dy["lag1_baseline_delta_kg_per_ha"] >= 0.0).astype(int)
    dy["flip_lag1"] = (dy["actual_sign"].astype(int) != dy["lag1_sign"].astype(int)).astype(int)
    dy = _add_priors(dy, train_years)
    state_oh = pd.get_dummies(dy["state_name"].astype(str), prefix="state", dtype=float)
    dy = pd.concat([dy.reset_index(drop=True), state_oh.reset_index(drop=True)], axis=1)

    obs = build_observed_proxy_features(
        weather_dir=weather_dir,
        district_ids=sorted(dy["district_id"].astype(str).unique()),
        season_years=sorted(dy["season_year"].astype(int).unique()),
        train_years=train_years,
        max_lead=int(args.max_lead),
    )
    obs.to_csv(out_dir / "observed_weather_proxy_features.csv", index=False)
    full = dy.merge(obs, on=["season_year", "district_id"], how="left")
    full.to_csv(out_dir / "observed_proxy_model_table.csv", index=False)

    train_mask = full["season_year"].astype(int).isin(train_years).to_numpy()
    train_flip_mask = train_mask & (full["season_year"].astype(int).to_numpy() > min(train_years))
    test_mask = full["season_year"].astype(int).isin(test_years).to_numpy()

    rows: List[Dict[str, object]] = []
    actual = full["actual_sign"].to_numpy(dtype=np.int64)
    lag = full["lag1_sign"].to_numpy(dtype=np.int64)
    rows.append(
        {
            "mode": "baseline",
            "model": "lag1_sign",
            "feature_set": "lag1",
            "threshold_variant": "raw",
            "threshold": float("nan"),
            "train_accuracy": float(accuracy_score(actual[train_mask], lag[train_mask])),
            "test_accuracy": float(accuracy_score(actual[test_mask], lag[test_mask])),
            "test_balanced_accuracy": float(balanced_accuracy_score(actual[test_mask], lag[test_mask])),
            "test_auc": float("nan"),
            "drop_recall": float(
                ((lag[test_mask] == 0) & (actual[test_mask] == 0)).sum()
                / max(1, int((actual[test_mask] == 0).sum()))
            ),
            "rise_recall": float(
                ((lag[test_mask] == 1) & (actual[test_mask] == 1)).sum()
                / max(1, int((actual[test_mask] == 1).sum()))
            ),
            "predicted_drop_rate": float((lag[test_mask] == 0).mean()),
            "test_flipped_rate": float("nan"),
        }
    )

    for i, cand in enumerate(_candidates(int(args.seed)), start=1):
        cols = _feature_cols(full, cand.feature_set)
        x = full[cols].to_numpy(dtype=np.float32)
        sign_model = clone(cand.model)
        sign_model.fit(x[train_mask], actual[train_mask])
        sign_prob = _predict_prob(sign_model, x)
        rows.extend(
            _score(
                mode="direct_actual_sign",
                model_name=cand.name,
                feature_set=cand.feature_set,
                prob=sign_prob,
                dy=full,
                train_mask=train_mask,
                test_mask=test_mask,
            )
        )

        flip_model = clone(cand.model)
        flip_model.fit(x[train_flip_mask], full["flip_lag1"].to_numpy(dtype=np.int64)[train_flip_mask])
        flip_prob = _predict_prob(flip_model, x)
        rows.extend(
            _score(
                mode="lag1_flip_router",
                model_name=cand.name,
                feature_set=cand.feature_set,
                prob=flip_prob,
                dy=full,
                train_mask=train_flip_mask,
                test_mask=test_mask,
            )
        )
        if i % 15 == 0:
            pd.DataFrame(rows).sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=[False, False]).to_csv(
                out_dir / "observed_proxy_sign_metrics.csv", index=False
            )
            print(f"finished {i}", flush=True)

    metrics = pd.DataFrame(rows).sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=[False, False])
    metrics.to_csv(out_dir / "observed_proxy_sign_metrics.csv", index=False)
    full[[c for c in full.columns if c.startswith("obsproxy_")] + ["actual_sign"]].corr(numeric_only=True)[
        "actual_sign"
    ].sort_values(key=lambda s: s.abs(), ascending=False).to_csv(out_dir / "observed_proxy_feature_corr.csv")
    with (out_dir / "observed_proxy_config.json").open("w") as fh:
        json.dump(
            {
                "warning": "Uses short-lead future-weather proxy features; diagnostic/oracle-style, not operational at early forecast date.",
                "max_lead": int(args.max_lead),
                "train_years": train_years,
                "test_years": test_years,
                "sign_table": str(args.sign_table),
                "weather_dir": str(weather_dir),
            },
            fh,
            indent=2,
        )
    return {"out_dir": str(out_dir), "best": metrics.head(12).to_dict(orient="records")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sign search using short-lead observed-weather proxy features.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config-data", default="expansion_2010_workspace/configs/codex_data_v2_2010.yaml")
    parser.add_argument(
        "--sign-table",
        default="codex_v2/experiments/sign_breakthrough_2010_2018_test2019_2022/district_year_sign_table.csv",
    )
    parser.add_argument("--max-lead", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-years", nargs="+", type=int, default=list(range(2010, 2019)))
    parser.add_argument("--test-years", nargs="+", type=int, default=[2019, 2020, 2021, 2022])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
