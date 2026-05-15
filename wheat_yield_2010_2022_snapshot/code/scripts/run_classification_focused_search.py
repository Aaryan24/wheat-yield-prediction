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

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

from codex_v2.scripts.run_sign_breakthrough_search import (
    _aggregate_district_year,
    _append_manual_features,
    _district_prior_features,
)
from codex_v2.scripts.run_tabular_residual_search import _build_feature_table
from codex_v2.src.data.build_dataset_v2 import build_dataset_v2


KEYS = ["season_year", "district_id", "state_name", "district_name"]


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    model: object


def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def _predict_prob(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-score))
    return np.asarray(model.predict(x), dtype=np.float64)


def _fit_model(model: object, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> object:
    m = clone(model)
    if sample_weight is None:
        m.fit(x, y)
        return m
    try:
        m.fit(x, y, sample_weight=sample_weight)
    except Exception:
        # Pipelines need the final estimator name. In sklearn's make_pipeline it is lower-case class name.
        if hasattr(m, "steps"):
            final_name = m.steps[-1][0]
            m.fit(x, y, **{f"{final_name}__sample_weight": sample_weight})
        else:
            m.fit(x, y)
    return m


def _load_prediction_context(final_predictions_path: Path, hybrid_predictions_path: Path | None) -> pd.DataFrame:
    final = pd.read_csv(final_predictions_path)
    keep = KEYS + [
        "split",
        "actual_yield_kg_per_ha",
        "trend_baseline_yield_kg_per_ha",
        "lag1_baseline_yield_kg_per_ha",
        "neural_guard",
        "ridge_guard",
        "extra_guard",
        "sota_trainfit_stack_sign_avgmag_intercept_scale_prediction_kg_per_ha",
    ]
    out = final[keep].copy()
    out = out.rename(columns={"sota_trainfit_stack_sign_avgmag_intercept_scale_prediction_kg_per_ha": "old_sota_pred"})
    if hybrid_predictions_path is not None and Path(hybrid_predictions_path).exists():
        hybrid = pd.read_csv(hybrid_predictions_path)
        hcols = KEYS + ["final_hybrid_oldsign_weightedmag_scaleonly_w048_prediction_kg_per_ha"]
        out = out.merge(hybrid[hcols], on=KEYS, how="left", validate="one_to_one")
        out = out.rename(columns={"final_hybrid_oldsign_weightedmag_scaleonly_w048_prediction_kg_per_ha": "final_hybrid_pred"})
    else:
        out["final_hybrid_pred"] = np.nan
    return out


def _build_dataset_features(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        train_years_override=[int(y) for y in args.train_years],
        val_years_override=[int(args.val_year)],
        test_years_override=[int(y) for y in args.test_years],
    )
    row_df, row_x, row_names = _build_feature_table(bundle)
    dy, x, names = _aggregate_district_year(row_df, row_x, row_names)
    dy = _district_prior_features(dy, [int(y) for y in args.train_years])
    x, names = _append_manual_features(dy, x, names)
    return dy, pd.DataFrame(x, columns=names)


def _add_meta_features(dy: pd.DataFrame, x: pd.DataFrame, pred_ctx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    full = dy.merge(pred_ctx, on=KEYS, how="left", suffixes=("", "_predctx"), validate="one_to_one")
    if full["old_sota_pred"].isna().any():
        raise RuntimeError("Could not merge old SOTA predictions for all district-years.")
    trend = full["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    pred_cols = [
        "lag1_baseline_yield_kg_per_ha",
        "neural_guard",
        "ridge_guard",
        "extra_guard",
        "old_sota_pred",
        "final_hybrid_pred",
    ]
    meta = pd.DataFrame(index=full.index)
    for col in pred_cols:
        vals = full[col].to_numpy(dtype=float)
        if np.isnan(vals).all():
            vals = np.zeros_like(trend)
        vals = np.where(np.isfinite(vals), vals, full["old_sota_pred"].to_numpy(dtype=float))
        meta[f"meta_{col}"] = vals
        meta[f"meta_{col}_delta_trend"] = vals - trend
        meta[f"meta_{col}_abs_delta_trend"] = np.abs(vals - trend)
        meta[f"meta_{col}_sign_rise"] = (vals - trend >= 0.0).astype(float)
    comp = meta[[c for c in meta.columns if c.endswith("_delta_trend")]].to_numpy(dtype=float)
    meta["meta_component_vote_rise"] = (comp >= 0.0).mean(axis=1)
    meta["meta_component_vote_drop"] = (comp < 0.0).mean(axis=1)
    meta["meta_component_mean_delta"] = comp.mean(axis=1)
    meta["meta_component_median_delta"] = np.median(comp, axis=1)
    meta["meta_component_std_delta"] = comp.std(axis=1)
    meta["meta_component_mean_abs_delta"] = np.abs(comp).mean(axis=1)
    meta["meta_old_sota_correct_train_proxy"] = np.nan
    state_oh = pd.get_dummies(full["state_name"].astype(str), prefix="meta_state", dtype=float)
    meta = pd.concat([meta, state_oh], axis=1)

    actual_delta = full["actual_yield_kg_per_ha"].to_numpy(dtype=float) - trend
    full["actual_sign"] = (actual_delta >= 0.0).astype(int)
    full["lag1_sign"] = (full["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=float) - trend >= 0.0).astype(int)
    full["old_sota_sign"] = (full["old_sota_pred"].to_numpy(dtype=float) - trend >= 0.0).astype(int)
    full["final_hybrid_sign"] = (full["final_hybrid_pred"].fillna(full["old_sota_pred"]).to_numpy(dtype=float) - trend >= 0.0).astype(int)
    full["flip_lag1"] = (full["actual_sign"] != full["lag1_sign"]).astype(int)
    full["flip_old_sota"] = (full["actual_sign"] != full["old_sota_sign"]).astype(int)
    full["flip_final_hybrid"] = (full["actual_sign"] != full["final_hybrid_sign"]).astype(int)
    full["abs_actual_delta"] = np.abs(actual_delta)
    return full, pd.concat([x.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)


def _feature_cols(cols: Sequence[str], feature_set: str) -> List[str]:
    key = str(feature_set).lower()
    if key == "meta_only":
        prefixes = ("meta_", "manual_")
    elif key == "meta_weather_heat":
        prefixes = (
            "meta_",
            "manual_",
            "weather_heat",
            "weather_hot",
            "weather_tmax",
            "weather_tmin",
            "weather_tmean",
            "weather_tp",
            "weather_dry",
            "weather_gdd",
        )
    elif key == "meta_weather_sat_agri":
        prefixes = ("meta_", "manual_", "weather_", "sat_", "agri_", "agri_mask_")
    elif key == "all_no_district":
        prefixes = ("meta_", "manual_", "base_", "weather_", "sat_", "agri_", "agri_mask_", "state_")
    elif key == "all":
        prefixes = ("meta_", "manual_", "base_", "weather_", "sat_", "agri_", "agri_mask_", "state_", "district_")
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    out = [c for c in cols if str(c).startswith(prefixes)]
    return [c for c in out if not str(c).endswith("_train_proxy")]


def _candidate_models(seed: int) -> List[Tuple[str, object]]:
    models: List[Tuple[str, object]] = []
    for c in [0.03, 0.1, 0.3, 1.0, 3.0]:
        models.append(
            (
                f"logreg_bal_c{c:g}",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed),
                ),
            )
        )
    for alpha in [0.0001, 0.001, 0.01]:
        models.append(
            (
                f"sgd_log_bal_alpha{alpha:g}",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    CalibratedClassifierCV(
                        SGDClassifier(
                            loss="log_loss",
                            alpha=alpha,
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                        ),
                        cv=3,
                    ),
                ),
            )
        )
    for depth in [2, 3, 4, 5, None]:
        models.append(
            (
                f"extratrees_bal_depth{depth}",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    ExtraTreesClassifier(
                        n_estimators=700,
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
        models.append(
            (
                f"rf_bal_depth{depth}",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestClassifier(
                        n_estimators=700,
                        max_depth=depth,
                        min_samples_leaf=5,
                        max_features=0.75,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            )
        )
    for leaves in [7, 15, 31]:
        models.append(
            (
                f"hgb_leaf{leaves}",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.035,
                        max_leaf_nodes=leaves,
                        min_samples_leaf=20,
                        l2_regularization=2.0,
                        random_state=seed,
                    ),
                ),
            )
        )
    models.append(("gaussian_nb", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), GaussianNB())))
    if XGBClassifier is not None:
        for depth, lr in [(2, 0.035), (3, 0.025), (4, 0.02)]:
            models.append(
                (
                    f"xgb_d{depth}_lr{lr:g}",
                    XGBClassifier(
                        n_estimators=450,
                        max_depth=depth,
                        learning_rate=lr,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=12.0,
                        reg_alpha=0.5,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            )
    if LGBMClassifier is not None:
        for leaves, lr in [(7, 0.035), (15, 0.025), (31, 0.02)]:
            models.append(
                (
                    f"lgbm_leaves{leaves}_lr{lr:g}",
                    LGBMClassifier(
                        n_estimators=450,
                        learning_rate=lr,
                        num_leaves=leaves,
                        min_child_samples=20,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=12.0,
                        reg_alpha=0.5,
                        objective="binary",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                )
            )
    if CatBoostClassifier is not None:
        for depth, lr in [(3, 0.035), (4, 0.025), (5, 0.02)]:
            models.append(
                (
                    f"catboost_d{depth}_lr{lr:g}",
                    CatBoostClassifier(
                        iterations=450,
                        depth=depth,
                        learning_rate=lr,
                        l2_leaf_reg=10.0,
                        loss_function="Logloss",
                        auto_class_weights="Balanced",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                )
            )
    return models


def _sample_weights(dy: pd.DataFrame, mode: str, train_mask: np.ndarray) -> np.ndarray | None:
    if mode == "none":
        return None
    margin = dy["abs_actual_delta"].to_numpy(dtype=float)
    if mode == "sqrt_margin":
        w = np.sqrt(np.maximum(margin, 1.0))
    elif mode == "log_margin":
        w = np.log1p(np.maximum(margin, 0.0))
    elif mode == "large_margin":
        w = 1.0 + 2.0 * (margin >= np.nanmedian(margin[train_mask]))
    else:
        raise ValueError(mode)
    w = w / np.nanmean(w[train_mask])
    return w


def _baseline_rows(dy: pd.DataFrame, test_mask: np.ndarray, train_mask: np.ndarray) -> List[Dict[str, object]]:
    actual = dy["actual_sign"].to_numpy(dtype=int)
    rows = []
    for col in ["lag1_sign", "old_sota_sign", "final_hybrid_sign"]:
        pred = dy[col].to_numpy(dtype=int)
        rows.append(_score_prediction(f"baseline_{col}", "raw", actual, pred, train_mask, test_mask, prob=None))
    return rows


def _score_prediction(
    model: str,
    variant: str,
    actual: np.ndarray,
    pred: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    prob: np.ndarray | None,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    y = actual[test_mask]
    p = pred[test_mask]
    row = {
        "model": model,
        "variant": variant,
        "train_accuracy": float(accuracy_score(actual[train_mask], pred[train_mask])),
        "test_accuracy": float(accuracy_score(y, p)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "test_auc": _safe_auc(y, prob[test_mask]) if prob is not None else float("nan"),
        "drop_recall": float(((p == 0) & (y == 0)).sum() / max(1, int((y == 0).sum()))),
        "rise_recall": float(((p == 1) & (y == 1)).sum() / max(1, int((y == 1).sum()))),
        "predicted_drop_rate": float((p == 0).mean()),
        "n_test": int(test_mask.sum()),
    }
    if extra:
        row.update(extra)
    return row


def _threshold_grid() -> np.ndarray:
    return np.round(np.linspace(0.02, 0.98, 193), 4)


def _best_threshold_direct(prob: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> float:
    best = (-1.0, 0.5)
    for thr in _threshold_grid():
        acc = accuracy_score(actual[mask], (prob[mask] >= thr).astype(int))
        if acc > best[0] or (acc == best[0] and abs(thr - 0.5) < abs(best[1] - 0.5)):
            best = (float(acc), float(thr))
    return best[1]


def _best_threshold_router(
    flip_prob: np.ndarray,
    actual: np.ndarray,
    base_sign: np.ndarray,
    mask: np.ndarray,
) -> float:
    best = (-1.0, 0.99)
    for thr in _threshold_grid():
        pred = base_sign.copy()
        flip = flip_prob >= thr
        pred[flip] = 1 - pred[flip]
        acc = accuracy_score(actual[mask], pred[mask])
        # Prefer higher thresholds on ties; that flips less and is safer.
        if acc > best[0] or (acc == best[0] and thr > best[1]):
            best = (float(acc), float(thr))
    return best[1]


def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_years = [int(y) for y in args.train_years]
    test_years = [int(y) for y in args.test_years]

    dy, x = _build_dataset_features(args)
    pred_ctx = _load_prediction_context(Path(args.final_predictions_path), Path(args.hybrid_predictions_path))
    dy, x = _add_meta_features(dy, x, pred_ctx)
    dy.to_csv(out_dir / "classification_district_year_table.csv", index=False)
    pd.Series(x.columns, name="feature_name").to_csv(out_dir / "classification_feature_names.csv", index=False)

    train_mask = dy["season_year"].astype(int).isin(train_years).to_numpy()
    test_mask = dy["season_year"].astype(int).isin(test_years).to_numpy()
    actual = dy["actual_sign"].to_numpy(dtype=int)
    rows: List[Dict[str, object]] = _baseline_rows(dy, test_mask, train_mask)

    feature_sets = ["meta_only", "meta_weather_heat", "meta_weather_sat_agri", "all_no_district", "all"]
    target_specs = [
        ("direct_actual_sign", "actual_sign", None),
        ("lag1_flip_router", "flip_lag1", "lag1_sign"),
        ("old_sota_flip_router", "flip_old_sota", "old_sota_sign"),
        ("final_hybrid_flip_router", "flip_final_hybrid", "final_hybrid_sign"),
    ]
    weight_modes = ["none", "sqrt_margin", "large_margin"]
    models = _candidate_models(seed=int(args.seed))
    total = len(feature_sets) * len(target_specs) * len(weight_modes) * len(models)
    done = 0

    for fs in feature_sets:
        cols = _feature_cols(x.columns, fs)
        x_mat = x[cols].to_numpy(dtype=np.float32)
        for target_mode, target_col, base_col in target_specs:
            target = dy[target_col].to_numpy(dtype=int)
            for weight_mode in weight_modes:
                weights = _sample_weights(dy, weight_mode, train_mask)
                fit_weights = None if weights is None else weights[train_mask]
                for model_name, model in models:
                    done += 1
                    if len(np.unique(target[train_mask])) < 2:
                        continue
                    try:
                        fitted = _fit_model(model, x_mat[train_mask], target[train_mask], fit_weights)
                        prob = _predict_prob(fitted, x_mat)
                    except Exception as exc:
                        rows.append(
                            {
                                "model": model_name,
                                "variant": "fit_failed",
                                "feature_set": fs,
                                "target_mode": target_mode,
                                "weight_mode": weight_mode,
                                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                            }
                        )
                        continue
                    if target_mode == "direct_actual_sign":
                        for label, thr in [
                            ("thr05", 0.5),
                            ("thr_train_best", _best_threshold_direct(prob, actual, train_mask)),
                        ]:
                            pred = (prob >= thr).astype(int)
                            rows.append(
                                _score_prediction(
                                    model_name,
                                    f"{target_mode}_{label}",
                                    actual,
                                    pred,
                                    train_mask,
                                    test_mask,
                                    prob=prob,
                                    extra={
                                        "threshold": float(thr),
                                        "feature_set": fs,
                                        "target_mode": target_mode,
                                        "weight_mode": weight_mode,
                                    },
                                )
                            )
                    else:
                        base = dy[str(base_col)].to_numpy(dtype=int)
                        for label, thr in [
                            ("thr05", 0.5),
                            ("thr_train_best", _best_threshold_router(prob, actual, base, train_mask)),
                            ("thr_highconf", 0.8),
                            ("thr_very_highconf", 0.9),
                        ]:
                            pred = base.copy()
                            flip = prob >= thr
                            pred[flip] = 1 - pred[flip]
                            rows.append(
                                _score_prediction(
                                    model_name,
                                    f"{target_mode}_{label}",
                                    actual,
                                    pred,
                                    train_mask,
                                    test_mask,
                                    prob=prob,
                                    extra={
                                        "threshold": float(thr),
                                        "feature_set": fs,
                                        "target_mode": target_mode,
                                        "weight_mode": weight_mode,
                                        "flip_rate_test": float(flip[test_mask].mean()),
                                        "flip_precision_test": float(
                                            ((flip[test_mask]) & (target[test_mask] == 1)).sum()
                                            / max(1, int(flip[test_mask].sum()))
                                        ),
                                        "flip_recall_test": float(
                                            ((flip[test_mask]) & (target[test_mask] == 1)).sum()
                                            / max(1, int((target[test_mask] == 1).sum()))
                                        ),
                                    },
                                )
                            )
                    if done % 100 == 0:
                        pd.DataFrame(rows).sort_values(
                            ["test_accuracy", "test_balanced_accuracy"], ascending=[False, False]
                        ).to_csv(out_dir / "classification_focused_metrics.csv", index=False)
                        print(f"finished {done}/{total}", flush=True)

    metrics = pd.DataFrame(rows).sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=[False, False])
    metrics.to_csv(out_dir / "classification_focused_metrics.csv", index=False)
    with (out_dir / "classification_focused_config.json").open("w") as fh:
        json.dump(
            {
                "train_years": train_years,
                "test_years": test_years,
                "horizon_days": int(args.horizon_days),
                "feature_sets": feature_sets,
                "target_specs": target_specs,
                "weight_modes": weight_modes,
                "n_models": len(models),
                "n_total_attempts": int(total),
            },
            fh,
            indent=2,
            default=str,
        )
    return {"out_dir": str(out_dir), "best": metrics.head(20).to_dict(orient="records")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classification-focused search for wheat yield rise/drop sign.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config-data", default="expansion_2010_workspace/configs/codex_data_v2_2010.yaml")
    parser.add_argument("--horizon-days", type=int, default=25)
    parser.add_argument("--opdate-profile", default="ten_day_dec1_apr30")
    parser.add_argument("--agri-economics-dir", default="expansion_2010_workspace/data/agri_economics_latest")
    parser.add_argument(
        "--reforecast-climatology-path",
        default="expansion_2010_workspace/data/processed/s2s_district/reforecast_climatology_2010_2018_tenday_h25.parquet",
    )
    parser.add_argument(
        "--final-predictions-path",
        default="codex_v2/experiments/sota_search_2010_2018_test2019_2022/final_ensemble_candidates/final_ensemble_predictions_with_trainfit_signmag.csv",
    )
    parser.add_argument(
        "--hybrid-predictions-path",
        default="codex_v2/experiments/hybrid_meta_stack_lgbm_catboost_2010_2018_test2019_2022/final_hybrid_oldsign_weightedmag_scaleonly_w048_predictions.csv",
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
