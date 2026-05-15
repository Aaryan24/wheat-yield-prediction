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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional local dependency
    XGBClassifier = None

from codex_v2.scripts.run_tabular_residual_search import _build_feature_table, _feature_indices
from codex_v2.src.data.build_dataset_v2 import build_dataset_v2


@dataclass(frozen=True)
class SignCandidate:
    name: str
    feature_set: str
    model: object


def _sign01(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) >= 0.0).astype(np.int64)


def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def _aggregate_district_year(
    row_df: pd.DataFrame, x: np.ndarray, feature_names: Sequence[str]
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    meta = row_df.reset_index(drop=True).copy()
    meta["row_pos"] = np.arange(len(meta), dtype=np.int64)
    gcols = ["season_year", "district_id", "state_name", "district_name"]

    agg_rows: List[dict] = []
    feature_blocks: List[np.ndarray] = []
    for _, group in meta.groupby(gcols, sort=False, dropna=False):
        idx = group["row_pos"].to_numpy(dtype=np.int64)
        xg = x[idx]
        stats = np.concatenate(
            [
                np.nanmean(xg, axis=0),
                np.nanstd(xg, axis=0),
                np.nanmin(xg, axis=0),
                np.nanmax(xg, axis=0),
            ],
            axis=0,
        )
        first = group.iloc[0]
        actual_delta = float(first["actual_delta_kg_per_ha"])
        lag_delta = float(first["lag1_baseline_delta_kg_per_ha"])
        actual_sign = int(actual_delta >= 0.0)
        lag_sign = int(lag_delta >= 0.0)
        agg_rows.append(
            {
                "season_year": int(first["season_year"]),
                "district_id": str(first["district_id"]),
                "state_name": str(first["state_name"]),
                "district_name": str(first["district_name"]),
                "split": str(first["split"]),
                "actual_yield_kg_per_ha": float(first["actual_yield_kg_per_ha"]),
                "trend_baseline_yield_kg_per_ha": float(first["trend_baseline_yield_kg_per_ha"]),
                "lag1_baseline_yield_kg_per_ha": float(first["lag1_baseline_yield_kg_per_ha"]),
                "actual_delta_kg_per_ha": actual_delta,
                "lag1_baseline_delta_kg_per_ha": lag_delta,
                "actual_sign": actual_sign,
                "lag1_sign": lag_sign,
                "lag1_correct": int(actual_sign == lag_sign),
                "flip_lag1": int(actual_sign != lag_sign),
                "abs_actual_delta_kg_per_ha": abs(actual_delta),
                "abs_lag1_delta_kg_per_ha": abs(lag_delta),
                "n_opdates": int(group["operational_date"].nunique()),
            }
        )
        feature_blocks.append(stats.astype(np.float32))

    suffixes = ["mean", "std", "min", "max"]
    names = [f"{name}__op_{suffix}" for suffix in suffixes for name in feature_names]
    return pd.DataFrame(agg_rows), np.vstack(feature_blocks).astype(np.float32), names


def _district_prior_features(dy: pd.DataFrame, train_years: Sequence[int]) -> pd.DataFrame:
    out = dy.copy()
    train = out[out["season_year"].astype(int).isin([int(y) for y in train_years])]
    global_rate = float(train["actual_sign"].mean())
    district_rate = train.groupby("district_id")["actual_sign"].mean().to_dict()
    state_rate = train.groupby("state_name")["actual_sign"].mean().to_dict()
    out["prior_district_rise_rate"] = out["district_id"].map(district_rate).fillna(global_rate).astype(float)
    out["prior_state_rise_rate"] = out["state_name"].map(state_rate).fillna(global_rate).astype(float)
    out["prior_global_rise_rate"] = global_rate
    out["year_index"] = out["season_year"].astype(float) - float(min(train_years))
    return out


def _append_manual_features(dy: pd.DataFrame, x: np.ndarray, names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    manual = dy[
        [
            "prior_district_rise_rate",
            "prior_state_rise_rate",
            "prior_global_rise_rate",
            "year_index",
            "abs_lag1_delta_kg_per_ha",
        ]
    ].to_numpy(dtype=np.float32)
    manual_names = [
        "manual_prior_district_rise_rate",
        "manual_prior_state_rise_rate",
        "manual_prior_global_rise_rate",
        "manual_year_index",
        "manual_abs_lag1_delta_kg_per_ha",
    ]
    return np.concatenate([x, manual], axis=1), list(names) + manual_names


def _feature_indices_dy(feature_names: Sequence[str], feature_set: str) -> np.ndarray:
    key = str(feature_set).strip().lower()
    if key == "manual_lag":
        prefixes = ("base_lag1", "base_trend", "manual_", "state_")
    elif key == "weather_heat_lag":
        prefixes = (
            "base_lag1",
            "base_trend",
            "manual_",
            "state_",
            "weather_heat",
            "weather_hot",
            "weather_tmax",
            "weather_tmin",
            "weather_tmean",
            "weather_dry",
            "weather_tp",
            "weather_gdd",
        )
    elif key == "weather_sat_agri_no_district":
        prefixes = ("base_", "manual_", "weather_", "sat_", "agri_", "agri_mask_", "state_")
    elif key == "all_no_district":
        prefixes = ("base_", "manual_", "weather_", "sat_", "agri_", "agri_mask_", "state_")
    elif key == "all":
        prefixes = ("base_", "manual_", "weather_", "sat_", "agri_", "agri_mask_", "state_", "district_")
    else:
        prefixes = tuple()
    if not prefixes:
        raise ValueError(f"Unknown feature set: {feature_set}")
    keep = [idx for idx, name in enumerate(feature_names) if str(name).startswith(prefixes)]
    return np.asarray(keep, dtype=np.int64)


def _make_candidates(seed: int) -> List[SignCandidate]:
    feature_sets = [
        "manual_lag",
        "weather_heat_lag",
        "weather_sat_agri_no_district",
        "all",
    ]
    candidates: List[SignCandidate] = []
    for fs in feature_sets:
        for c in [0.1, 0.3, 1.0, 3.0]:
            candidates.append(
                SignCandidate(
                    name=f"logreg_c{c:g}_{fs}",
                    feature_set=fs,
                    model=make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        LogisticRegression(
                            C=c,
                            penalty="l2",
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                )
            )
        for depth in [2, 3, 4, None]:
            candidates.append(
                SignCandidate(
                    name=f"extratrees_depth{depth}_{fs}",
                    feature_set=fs,
                    model=make_pipeline(
                        SimpleImputer(strategy="median"),
                        ExtraTreesClassifier(
                            n_estimators=600,
                            max_depth=depth,
                            min_samples_leaf=4,
                            max_features=0.7,
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                )
            )
        for depth in [3, 5, 7]:
            candidates.append(
                SignCandidate(
                    name=f"rf_depth{depth}_{fs}",
                    feature_set=fs,
                    model=make_pipeline(
                        SimpleImputer(strategy="median"),
                        RandomForestClassifier(
                            n_estimators=600,
                            max_depth=depth,
                            min_samples_leaf=5,
                            max_features=0.65,
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                )
            )
        for leaf_nodes in [7, 15, 31]:
            candidates.append(
                SignCandidate(
                    name=f"hgb_leaf{leaf_nodes}_{fs}",
                    feature_set=fs,
                    model=make_pipeline(
                        SimpleImputer(strategy="median"),
                        HistGradientBoostingClassifier(
                            max_iter=250,
                            learning_rate=0.035,
                            max_leaf_nodes=leaf_nodes,
                            min_samples_leaf=20,
                            l2_regularization=2.0,
                            random_state=seed,
                        ),
                    ),
                )
            )
        if XGBClassifier is not None:
            for depth, lr in [(2, 0.035), (3, 0.025)]:
                candidates.append(
                    SignCandidate(
                        name=f"xgb_d{depth}_lr{lr:g}_{fs}",
                        feature_set=fs,
                        model=XGBClassifier(
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
    return candidates


def _predict_prob(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    decision = np.asarray(model.decision_function(x), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-decision))


def _thresholds() -> np.ndarray:
    return np.round(np.linspace(0.05, 0.95, 91), 4)


def _score_direct(
    dy: pd.DataFrame,
    prob: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    model_name: str,
    feature_set: str,
) -> List[Dict[str, object]]:
    y = dy["actual_sign"].to_numpy(dtype=np.int64)
    rows: List[Dict[str, object]] = []
    train_scores = []
    for thr in _thresholds():
        pred_train = (prob[train_mask] >= thr).astype(np.int64)
        train_scores.append((float(accuracy_score(y[train_mask], pred_train)), float(thr)))
    best_thr = sorted(train_scores, key=lambda t: (t[0], -abs(t[1] - 0.5)), reverse=True)[0][1]
    for label, thr in [("thr05", 0.5), ("thr_train_best", best_thr)]:
        pred = (prob[test_mask] >= thr).astype(np.int64)
        rows.append(
            {
                "mode": "direct_actual_sign",
                "model": model_name,
                "feature_set": feature_set,
                "threshold_variant": label,
                "threshold": float(thr),
                "train_accuracy": float(
                    accuracy_score(y[train_mask], (prob[train_mask] >= thr).astype(np.int64))
                ),
                "test_accuracy": float(accuracy_score(y[test_mask], pred)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y[test_mask], pred)),
                "test_auc": _safe_auc(y[test_mask], prob[test_mask]),
                "drop_recall": float(((pred == 0) & (y[test_mask] == 0)).sum() / max(1, (y[test_mask] == 0).sum())),
                "rise_recall": float(((pred == 1) & (y[test_mask] == 1)).sum() / max(1, (y[test_mask] == 1).sum())),
                "predicted_drop_rate": float((pred == 0).mean()),
                "n_test": int(test_mask.sum()),
            }
        )
    return rows


def _score_flip_router(
    dy: pd.DataFrame,
    flip_prob: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    model_name: str,
    feature_set: str,
) -> List[Dict[str, object]]:
    y_actual = dy["actual_sign"].to_numpy(dtype=np.int64)
    lag_sign = dy["lag1_sign"].to_numpy(dtype=np.int64)
    y_flip = dy["flip_lag1"].to_numpy(dtype=np.int64)
    rows: List[Dict[str, object]] = []
    train_scores = []
    for thr in _thresholds():
        train_final = lag_sign[train_mask].copy()
        train_final[flip_prob[train_mask] >= thr] = 1 - train_final[flip_prob[train_mask] >= thr]
        train_scores.append((float(accuracy_score(y_actual[train_mask], train_final)), float(thr)))
    best_thr = sorted(train_scores, key=lambda t: (t[0], t[1]), reverse=True)[0][1]

    for label, thr in [("thr05", 0.5), ("thr_train_best", best_thr)]:
        final = lag_sign[test_mask].copy()
        flip = flip_prob[test_mask] >= thr
        final[flip] = 1 - final[flip]
        y = y_actual[test_mask]
        rows.append(
            {
                "mode": "lag1_flip_router",
                "model": model_name,
                "feature_set": feature_set,
                "threshold_variant": label,
                "threshold": float(thr),
                "train_accuracy": float(
                    sorted(train_scores, key=lambda t: abs(t[1] - thr))[0][0]
                    if label == "thr_train_best"
                    else accuracy_score(
                        y_actual[train_mask],
                        np.where(flip_prob[train_mask] >= thr, 1 - lag_sign[train_mask], lag_sign[train_mask]),
                    )
                ),
                "test_accuracy": float(accuracy_score(y, final)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y, final)),
                "test_auc": _safe_auc(y_flip[test_mask], flip_prob[test_mask]),
                "drop_recall": float(((final == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
                "rise_recall": float(((final == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
                "predicted_drop_rate": float((final == 0).mean()),
                "n_test": int(test_mask.sum()),
                "test_flip_precision": float(
                    ((flip) & (y_flip[test_mask] == 1)).sum() / max(1, int(flip.sum()))
                ),
                "test_flip_recall": float(
                    ((flip) & (y_flip[test_mask] == 1)).sum() / max(1, int((y_flip[test_mask] == 1).sum()))
                ),
                "test_flipped_rate": float(flip.mean()),
            }
        )
    return rows


def _baseline_rows(dy: pd.DataFrame, train_years: Sequence[int], test_years: Sequence[int]) -> List[Dict[str, object]]:
    test = dy[dy["season_year"].astype(int).isin([int(y) for y in test_years])].copy()
    y = test["actual_sign"].to_numpy(dtype=np.int64)
    lag = test["lag1_sign"].to_numpy(dtype=np.int64)
    rows: List[Dict[str, object]] = []
    rows.append(
        {
            "mode": "baseline",
            "model": "lag1_sign",
            "feature_set": "lag1",
            "threshold_variant": "raw",
            "threshold": float("nan"),
            "train_accuracy": float(
                accuracy_score(
                    dy[dy["season_year"].astype(int).isin([int(y) for y in train_years])]["actual_sign"],
                    dy[dy["season_year"].astype(int).isin([int(y) for y in train_years])]["lag1_sign"],
                )
            ),
            "test_accuracy": float(accuracy_score(y, lag)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y, lag)),
            "test_auc": float("nan"),
            "drop_recall": float(((lag == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
            "rise_recall": float(((lag == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
            "predicted_drop_rate": float((lag == 0).mean()),
            "n_test": int(len(test)),
        }
    )
    majority = int(
        dy[dy["season_year"].astype(int).isin([int(yy) for yy in train_years])]["actual_sign"].mean() >= 0.5
    )
    pred_major = np.full_like(y, majority)
    rows.append(
        {
            "mode": "baseline",
            "model": "train_majority_sign",
            "feature_set": "none",
            "threshold_variant": "raw",
            "threshold": float("nan"),
            "train_accuracy": float("nan"),
            "test_accuracy": float(accuracy_score(y, pred_major)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y, pred_major)),
            "test_auc": float("nan"),
            "drop_recall": float(((pred_major == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
            "rise_recall": float(((pred_major == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
            "predicted_drop_rate": float((pred_major == 0).mean()),
            "n_test": int(len(test)),
        }
    )
    return rows


def _diagnostics(dy: pd.DataFrame, out_dir: Path) -> None:
    dy.groupby("season_year").agg(
        n=("actual_sign", "size"),
        actual_rise_rate=("actual_sign", "mean"),
        lag1_rise_rate=("lag1_sign", "mean"),
        lag1_sign_accuracy=("lag1_correct", "mean"),
        flip_rate=("flip_lag1", "mean"),
        mean_actual_delta=("actual_delta_kg_per_ha", "mean"),
        mean_lag_delta=("lag1_baseline_delta_kg_per_ha", "mean"),
    ).reset_index().to_csv(out_dir / "sign_diagnostics_by_year.csv", index=False)

    dy.groupby(["season_year", "state_name"]).agg(
        n=("actual_sign", "size"),
        actual_rise_rate=("actual_sign", "mean"),
        lag1_rise_rate=("lag1_sign", "mean"),
        lag1_sign_accuracy=("lag1_correct", "mean"),
        flip_rate=("flip_lag1", "mean"),
    ).reset_index().to_csv(out_dir / "sign_diagnostics_by_year_state.csv", index=False)

    dy.groupby("district_id").agg(
        state_name=("state_name", "first"),
        district_name=("district_name", "first"),
        n=("actual_sign", "size"),
        actual_rise_rate=("actual_sign", "mean"),
        lag1_sign_accuracy=("lag1_correct", "mean"),
        flip_rate=("flip_lag1", "mean"),
    ).reset_index().sort_values("lag1_sign_accuracy").to_csv(
        out_dir / "sign_diagnostics_by_district.csv", index=False
    )


def _feature_associations(
    dy: pd.DataFrame,
    x: np.ndarray,
    feature_names: Sequence[str],
    train_mask: np.ndarray,
    out_dir: Path,
) -> None:
    y = dy["actual_sign"].to_numpy(dtype=np.int64)
    train_x = pd.DataFrame(x[train_mask], columns=feature_names)
    train_y = y[train_mask]
    rows: List[Dict[str, object]] = []
    for name in feature_names:
        if not (
            name.startswith("weather_")
            or name.startswith("sat_")
            or name.startswith("agri_")
            or name.startswith("manual_")
            or name.startswith("base_lag1")
        ):
            continue
        vals = pd.to_numeric(train_x[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() < 20 or vals.nunique(dropna=True) < 3:
            continue
        corr = np.corrcoef(vals.fillna(vals.median()).to_numpy(dtype=float), train_y.astype(float))[0, 1]
        rise_mean = float(vals[train_y == 1].mean())
        drop_mean = float(vals[train_y == 0].mean())
        rows.append(
            {
                "feature": name,
                "corr_with_actual_rise_train": float(corr),
                "rise_mean_train": rise_mean,
                "drop_mean_train": drop_mean,
                "mean_difference_rise_minus_drop": rise_mean - drop_mean,
            }
        )
    pd.DataFrame(rows).assign(abs_corr=lambda d: d["corr_with_actual_rise_train"].abs()).sort_values(
        "abs_corr", ascending=False
    ).drop(columns=["abs_corr"]).to_csv(out_dir / "feature_associations_train.csv", index=False)


def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_years = [int(y) for y in args.train_years]
    test_years = [int(y) for y in args.test_years]

    bundle = build_dataset_v2(
        data_config_path=Path(args.config_data),
        mode="shared",
        target_mode="district_signed_log",
        horizon_days=int(args.horizon_days),
        operational_dates=None,
        opdate_profile=str(args.opdate_profile),
        allow_manual_opdates_override=False,
        apply_sat_mask_fix=True,
        use_engineered_weather=True,
        use_engineered_satellite=True,
        use_missingness_indicators=True,
        enable_e6=True,
        enable_e7=True,
        enable_token_time_features=True,
        enable_agri_economics=True,
        agri_economics_dir=Path(args.agri_economics_dir),
        agri_year_lag=int(args.agri_year_lag),
        enable_yield_history_features=True,
        residual_baseline_mode="lag1_residual_linear",
        reforecast_climatology_path=Path(args.reforecast_climatology_path),
        detrend_targets=True,
        train_years_override=train_years,
        val_years_override=[int(args.val_year)],
        test_years_override=test_years,
    )
    row_df, row_x, row_feature_names = _build_feature_table(bundle)
    dy, x, feature_names = _aggregate_district_year(row_df, row_x, row_feature_names)
    dy = _district_prior_features(dy, train_years)
    x, feature_names = _append_manual_features(dy, x, feature_names)

    dy.to_csv(out_dir / "district_year_sign_table.csv", index=False)
    pd.Series(feature_names, name="feature_name").to_csv(out_dir / "district_year_feature_names.csv", index=False)
    _diagnostics(dy, out_dir)

    train_mask = dy["season_year"].astype(int).isin(train_years).to_numpy()
    test_mask = dy["season_year"].astype(int).isin(test_years).to_numpy()
    train_flip_mask = train_mask & (dy["season_year"].astype(int).to_numpy() > min(train_years))
    _feature_associations(dy, x, feature_names, train_mask, out_dir)

    metrics: List[Dict[str, object]] = []
    metrics.extend(_baseline_rows(dy, train_years, test_years))
    y_sign = dy["actual_sign"].to_numpy(dtype=np.int64)
    y_flip = dy["flip_lag1"].to_numpy(dtype=np.int64)

    candidates = _make_candidates(int(args.seed))
    for i, cand in enumerate(candidates, start=1):
        idx = _feature_indices_dy(feature_names, cand.feature_set)
        model = cand.model
        model.fit(x[train_mask][:, idx], y_sign[train_mask])
        sign_prob = _predict_prob(model, x[:, idx])
        metrics.extend(
            _score_direct(
                dy,
                sign_prob,
                train_mask,
                test_mask,
                model_name=cand.name,
                feature_set=cand.feature_set,
            )
        )

        flip_model = cand.model
        # Recreate the estimator to avoid warm-start/shared-state surprises.
        flip_model = _make_candidates(int(args.seed))[i - 1].model
        flip_model.fit(x[train_flip_mask][:, idx], y_flip[train_flip_mask])
        flip_prob = _predict_prob(flip_model, x[:, idx])
        metrics.extend(
            _score_flip_router(
                dy,
                flip_prob,
                train_flip_mask,
                test_mask,
                model_name=cand.name,
                feature_set=cand.feature_set,
            )
        )

        if i % 12 == 0:
            pd.DataFrame(metrics).sort_values("test_accuracy", ascending=False).to_csv(
                out_dir / "sign_breakthrough_metrics.csv", index=False
            )
            print(f"finished {i}/{len(candidates)} candidates", flush=True)

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["test_accuracy", "test_balanced_accuracy"], ascending=[False, False]
    )
    metrics_df.to_csv(out_dir / "sign_breakthrough_metrics.csv", index=False)

    # A compact feature-importance pass for the best non-baseline model.
    best = metrics_df[metrics_df["mode"].isin(["direct_actual_sign", "lag1_flip_router"])].head(1)
    if len(best):
        best_name = str(best.iloc[0]["model"])
        best_mode = str(best.iloc[0]["mode"])
        match = [c for c in candidates if c.name == best_name][0]
        idx = _feature_indices_dy(feature_names, match.feature_set)
        target = y_sign if best_mode == "direct_actual_sign" else y_flip
        fit_mask = train_mask if best_mode == "direct_actual_sign" else train_flip_mask
        model = match.model
        model.fit(x[fit_mask][:, idx], target[fit_mask])
        perm = permutation_importance(
            model,
            x[test_mask][:, idx],
            target[test_mask],
            n_repeats=8,
            random_state=int(args.seed),
            scoring="accuracy",
        )
        imp = pd.DataFrame(
            {
                "feature": np.asarray(feature_names, dtype=object)[idx],
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        imp.to_csv(out_dir / "best_model_permutation_importance_test.csv", index=False)

    with (out_dir / "sign_breakthrough_config.json").open("w") as fh:
        json.dump(
            {
                "train_years": train_years,
                "test_years": test_years,
                "n_district_year_rows": int(len(dy)),
                "n_features": int(x.shape[1]),
                "config_data": str(args.config_data),
                "agri_economics_dir": str(args.agri_economics_dir),
                "reforecast_climatology_path": str(args.reforecast_climatology_path),
            },
            fh,
            indent=2,
        )
    return {"out_dir": str(out_dir), "best": metrics_df.head(12).to_dict(orient="records")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sign-focused EDA and classifier/router search.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config-data", default="expansion_2010_workspace/configs/codex_data_v2_2010.yaml")
    parser.add_argument("--horizon-days", type=int, default=25)
    parser.add_argument("--opdate-profile", default="ten_day_dec1_apr30")
    parser.add_argument("--agri-economics-dir", default="expansion_2010_workspace/data/agri_economics_latest")
    parser.add_argument(
        "--reforecast-climatology-path",
        default="expansion_2010_workspace/data/processed/s2s_district/reforecast_climatology_2010_2018_tenday_h25.parquet",
    )
    parser.add_argument("--agri-year-lag", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-years", nargs="+", type=int, default=list(range(2010, 2019)))
    parser.add_argument("--val-year", type=int, default=2018)
    parser.add_argument("--test-years", nargs="+", type=int, default=[2019, 2020, 2021, 2022])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
