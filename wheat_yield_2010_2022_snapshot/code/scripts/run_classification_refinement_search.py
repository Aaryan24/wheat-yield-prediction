#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
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

from codex_v2.scripts.run_classification_focused_search import (
    KEYS,
    _add_meta_features,
    _best_threshold_direct,
    _best_threshold_router,
    _build_dataset_features,
    _feature_cols,
    _load_prediction_context,
    _predict_prob,
    _safe_auc,
    _sample_weights,
)


warnings.filterwarnings("ignore")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse_hybridmag": _rmse(y_true, y_pred),
        "mae_hybridmag": float(mean_absolute_error(y_true, y_pred)),
        "r2_hybridmag": float(r2_score(y_true, y_pred)),
    }


def _score_full(
    *,
    model: str,
    variant: str,
    actual_sign: np.ndarray,
    pred_sign: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    prob: np.ndarray | None,
    actual_yield: np.ndarray,
    trend: np.ndarray,
    final_hybrid_pred: np.ndarray,
    auc_target: np.ndarray | None = None,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    y = actual_sign[test_mask]
    p = pred_sign[test_mask]
    auc_y = actual_sign if auc_target is None else auc_target
    sign_mult = np.where(pred_sign >= 1, 1.0, -1.0)
    mag = np.abs(final_hybrid_pred - trend)
    pred_y = trend + sign_mult * mag
    row = {
        "model": model,
        "variant": variant,
        "train_accuracy": float(accuracy_score(actual_sign[train_mask], pred_sign[train_mask])),
        "test_accuracy": float(accuracy_score(y, p)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "test_auc": _safe_auc(auc_y[test_mask], prob[test_mask]) if prob is not None else float("nan"),
        "drop_recall": float(((p == 0) & (y == 0)).sum() / max(1, int((y == 0).sum()))),
        "rise_recall": float(((p == 1) & (y == 1)).sum() / max(1, int((y == 1).sum()))),
        "predicted_drop_rate": float((p == 0).mean()),
        "n_test": int(test_mask.sum()),
    }
    row.update(_regression_metrics(actual_yield[test_mask], pred_y[test_mask]))
    if extra:
        row.update(extra)
    return row


def _make_selected_pipeline(steps: List[Tuple[str, object]], k: int | str, n_features: int) -> Pipeline:
    resolved_k: int | str
    if k == "all":
        resolved_k = "all"
    else:
        resolved_k = min(int(k), int(n_features))
    full_steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if resolved_k != "all" and int(n_features) > int(resolved_k):
        full_steps.append(("select", SelectKBest(score_func=f_classif, k=resolved_k)))
    full_steps.extend(steps)
    return Pipeline(full_steps)


def _candidate_models(seed: int, n_features: int, feature_set: str) -> List[Tuple[str, object]]:
    if feature_set == "meta_only":
        k_values: List[int | str] = ["all"]
    elif n_features <= 80:
        k_values = ["all", 40]
    else:
        k_values = [40, 80, 160]

    models: List[Tuple[str, object]] = []
    for k in k_values:
        for c in [0.01, 0.03, 0.1, 0.3]:
            models.append(
                (
                    f"logreg_c{c:g}_k{k}",
                    _make_selected_pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "clf",
                                LogisticRegression(
                                    C=c,
                                    class_weight="balanced",
                                    max_iter=5000,
                                    random_state=seed,
                                ),
                            ),
                        ],
                        k,
                        n_features,
                    ),
                )
            )

    tree_k = ["all"] if feature_set == "meta_only" else [80, 160]
    for k in tree_k:
        for depth in [2, 3, 4]:
            models.append(
                (
                    f"extratrees_d{depth}_k{k}",
                    _make_selected_pipeline(
                        [
                            (
                                "clf",
                                ExtraTreesClassifier(
                                    n_estimators=300,
                                    max_depth=depth,
                                    min_samples_leaf=5,
                                    max_features=0.8,
                                    class_weight="balanced",
                                    random_state=seed,
                                    n_jobs=-1,
                                ),
                            )
                        ],
                        k,
                        n_features,
                    ),
                )
            )
        models.append(
            (
                f"hgb_l7_k{k}",
                _make_selected_pipeline(
                    [
                        (
                            "clf",
                            HistGradientBoostingClassifier(
                                max_iter=180,
                                learning_rate=0.04,
                                max_leaf_nodes=7,
                                min_samples_leaf=20,
                                l2_regularization=3.0,
                                random_state=seed,
                            ),
                        )
                    ],
                    k,
                    n_features,
                ),
            )
        )
        if XGBClassifier is not None:
            models.append(
                (
                    f"xgb_d2_k{k}",
                    _make_selected_pipeline(
                        [
                            (
                                "clf",
                                XGBClassifier(
                                    n_estimators=240,
                                    max_depth=2,
                                    learning_rate=0.035,
                                    subsample=0.9,
                                    colsample_bytree=0.8,
                                    reg_lambda=14.0,
                                    reg_alpha=1.0,
                                    objective="binary:logistic",
                                    eval_metric="logloss",
                                    random_state=seed,
                                    n_jobs=-1,
                                ),
                            )
                        ],
                        k,
                        n_features,
                    ),
                )
            )
        if LGBMClassifier is not None:
            models.append(
                (
                    f"lgbm_l7_k{k}",
                    _make_selected_pipeline(
                        [
                            (
                                "clf",
                                LGBMClassifier(
                                    n_estimators=240,
                                    learning_rate=0.035,
                                    num_leaves=7,
                                    min_child_samples=20,
                                    subsample=0.9,
                                    colsample_bytree=0.8,
                                    reg_lambda=14.0,
                                    reg_alpha=1.0,
                                    objective="binary",
                                    class_weight="balanced",
                                    random_state=seed,
                                    n_jobs=-1,
                                    verbosity=-1,
                                ),
                            )
                        ],
                        k,
                        n_features,
                    ),
                )
            )
        if CatBoostClassifier is not None:
            models.append(
                (
                    f"catboost_d3_k{k}",
                    _make_selected_pipeline(
                        [
                            (
                                "clf",
                                CatBoostClassifier(
                                    iterations=240,
                                    depth=3,
                                    learning_rate=0.035,
                                    l2_leaf_reg=12.0,
                                    loss_function="Logloss",
                                    auto_class_weights="Balanced",
                                    random_seed=seed,
                                    verbose=False,
                                    allow_writing_files=False,
                                ),
                            )
                        ],
                        k,
                        n_features,
                    ),
                )
            )
    return models


def _fit_model(model: object, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> object:
    m = clone(model)
    if sample_weight is None:
        m.fit(x, y)
        return m
    try:
        m.fit(x, y, clf__sample_weight=sample_weight)
    except Exception:
        m.fit(x, y)
    return m


def _selected_feature_names(model: object, cols: Sequence[str]) -> List[str]:
    if not hasattr(model, "named_steps") or "select" not in model.named_steps:
        return list(cols)
    mask = model.named_steps["select"].get_support()
    return [c for c, keep in zip(cols, mask) if keep]


def _coefficient_rows(model: object, cols: Sequence[str], meta: Dict[str, object]) -> List[Dict[str, object]]:
    if not hasattr(model, "named_steps") or "clf" not in model.named_steps:
        return []
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    names = _selected_feature_names(model, cols)
    coefs = np.asarray(clf.coef_).reshape(-1)
    rows = []
    for name, coef in sorted(zip(names, coefs), key=lambda kv: abs(kv[1]), reverse=True):
        row = dict(meta)
        row.update({"feature_name": name, "coefficient": float(coef), "abs_coefficient": float(abs(coef))})
        rows.append(row)
    return rows


def _baseline_rows(
    dy: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    actual_sign: np.ndarray,
    actual_yield: np.ndarray,
    trend: np.ndarray,
    final_hybrid_pred: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    baselines = {
        "lag1": ("lag1_sign", "lag1_baseline_yield_kg_per_ha"),
        "old_sota": ("old_sota_sign", "old_sota_pred"),
        "final_hybrid": ("final_hybrid_sign", "final_hybrid_pred"),
    }
    for name, (sign_col, pred_col) in baselines.items():
        pred_sign = dy[sign_col].to_numpy(dtype=int)
        pred_y = dy[pred_col].to_numpy(dtype=float)
        row = _score_full(
            model=f"baseline_{name}",
            variant="raw",
            actual_sign=actual_sign,
            pred_sign=pred_sign,
            train_mask=train_mask,
            test_mask=test_mask,
            prob=None,
            actual_yield=actual_yield,
            trend=trend,
            final_hybrid_pred=final_hybrid_pred,
        )
        row.update(_regression_metrics(actual_yield[test_mask], pred_y[test_mask]))
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_years = [int(y) for y in args.train_years]
    test_years = [int(y) for y in args.test_years]

    dy, x = _build_dataset_features(args)
    pred_ctx = _load_prediction_context(Path(args.final_predictions_path), Path(args.hybrid_predictions_path))
    dy, x = _add_meta_features(dy, x, pred_ctx)

    final_hybrid = dy["final_hybrid_pred"].fillna(dy["old_sota_pred"]).to_numpy(dtype=float)
    dy["final_hybrid_pred"] = final_hybrid
    dy.to_csv(out_dir / "classification_refinement_district_year_table.csv", index=False)
    pd.Series(x.columns, name="feature_name").to_csv(out_dir / "classification_refinement_feature_names.csv", index=False)

    train_mask = dy["season_year"].astype(int).isin(train_years).to_numpy()
    test_mask = dy["season_year"].astype(int).isin(test_years).to_numpy()
    actual_sign = dy["actual_sign"].to_numpy(dtype=int)
    actual_yield = dy["actual_yield_kg_per_ha"].to_numpy(dtype=float)
    trend = dy["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    abs_delta = dy["abs_actual_delta"].to_numpy(dtype=float)

    rows = _baseline_rows(dy, train_mask, test_mask, actual_sign, actual_yield, trend, final_hybrid)
    best_row: Dict[str, object] | None = None
    best_pred_frame: pd.DataFrame | None = None
    best_coefficients: List[Dict[str, object]] = []

    feature_sets = ["meta_only", "meta_weather_heat", "meta_weather_sat_agri", "all_no_district"]
    target_specs = [
        ("direct_actual_sign", "actual_sign", None),
        ("lag1_flip_router", "flip_lag1", "lag1_sign"),
        ("old_sota_flip_router", "flip_old_sota", "old_sota_sign"),
    ]
    weight_modes = ["none", "sqrt_margin"]
    train_min_abs_deltas = [0.0, 50.0, 100.0]

    total = 0
    for fs in feature_sets:
        total += len(target_specs) * len(weight_modes) * len(train_min_abs_deltas) * len(_candidate_models(int(args.seed), len(_feature_cols(x.columns, fs)), fs))
    done = 0

    for fs in feature_sets:
        cols = _feature_cols(x.columns, fs)
        x_mat = x[cols].to_numpy(dtype=np.float32)
        models = _candidate_models(int(args.seed), len(cols), fs)
        for target_mode, target_col, base_col in target_specs:
            target = dy[target_col].to_numpy(dtype=int)
            for min_abs in train_min_abs_deltas:
                fit_mask = train_mask & (abs_delta >= float(min_abs))
                if fit_mask.sum() < 100 or len(np.unique(target[fit_mask])) < 2:
                    continue
                for weight_mode in weight_modes:
                    weights = _sample_weights(dy, weight_mode, train_mask)
                    fit_weights = None if weights is None else weights[fit_mask]
                    for model_name, model in models:
                        done += 1
                        try:
                            fitted = _fit_model(model, x_mat[fit_mask], target[fit_mask], fit_weights)
                            prob = _predict_prob(fitted, x_mat)
                        except Exception as exc:
                            rows.append(
                                {
                                    "model": model_name,
                                    "variant": "fit_failed",
                                    "feature_set": fs,
                                    "target_mode": target_mode,
                                    "weight_mode": weight_mode,
                                    "train_min_abs_delta": float(min_abs),
                                    "error": f"{type(exc).__name__}: {str(exc)[:180]}",
                                }
                            )
                            continue

                        if target_mode == "direct_actual_sign":
                            thresholds = [
                                ("thr05", 0.5),
                                ("thr_train_best", _best_threshold_direct(prob, actual_sign, train_mask)),
                            ]
                            for label, thr in thresholds:
                                pred_sign = (prob >= thr).astype(int)
                                row = _score_full(
                                    model=model_name,
                                    variant=f"{target_mode}_{label}",
                                    actual_sign=actual_sign,
                                    pred_sign=pred_sign,
                                    train_mask=train_mask,
                                    test_mask=test_mask,
                                    prob=prob,
                                    actual_yield=actual_yield,
                                    trend=trend,
                                    final_hybrid_pred=final_hybrid,
                                    auc_target=actual_sign,
                                    extra={
                                        "threshold": float(thr),
                                        "feature_set": fs,
                                        "target_mode": target_mode,
                                        "weight_mode": weight_mode,
                                        "train_min_abs_delta": float(min_abs),
                                        "n_fit": int(fit_mask.sum()),
                                        "n_features": int(len(cols)),
                                    },
                                )
                                rows.append(row)
                                best_row, best_pred_frame, best_coefficients = _maybe_update_best(
                                    row,
                                    best_row,
                                    best_pred_frame,
                                    best_coefficients,
                                    dy,
                                    pred_sign,
                                    prob,
                                    fitted,
                                    cols,
                                )
                        else:
                            base = dy[str(base_col)].to_numpy(dtype=int)
                            thresholds = [
                                ("thr05", 0.5),
                                ("thr07", 0.7),
                                ("thr08", 0.8),
                                ("thr085", 0.85),
                                ("thr09", 0.9),
                                ("thr095", 0.95),
                                ("thr_train_best", _best_threshold_router(prob, actual_sign, base, train_mask)),
                            ]
                            for label, thr in thresholds:
                                pred_sign = base.copy()
                                flip = prob >= thr
                                pred_sign[flip] = 1 - pred_sign[flip]
                                row = _score_full(
                                    model=model_name,
                                    variant=f"{target_mode}_{label}",
                                    actual_sign=actual_sign,
                                    pred_sign=pred_sign,
                                    train_mask=train_mask,
                                    test_mask=test_mask,
                                    prob=prob,
                                    actual_yield=actual_yield,
                                    trend=trend,
                                    final_hybrid_pred=final_hybrid,
                                    auc_target=target,
                                    extra={
                                        "threshold": float(thr),
                                        "feature_set": fs,
                                        "target_mode": target_mode,
                                        "weight_mode": weight_mode,
                                        "train_min_abs_delta": float(min_abs),
                                        "n_fit": int(fit_mask.sum()),
                                        "n_features": int(len(cols)),
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
                                rows.append(row)
                                best_row, best_pred_frame, best_coefficients = _maybe_update_best(
                                    row,
                                    best_row,
                                    best_pred_frame,
                                    best_coefficients,
                                    dy,
                                    pred_sign,
                                    prob,
                                    fitted,
                                    cols,
                                )
                        if done % 50 == 0:
                            print(f"finished {done}/{total}", flush=True)
                            pd.DataFrame(rows).sort_values(
                                ["test_accuracy", "test_balanced_accuracy", "rmse_hybridmag"],
                                ascending=[False, False, True],
                            ).to_csv(out_dir / "classification_refinement_metrics.csv", index=False)

    metrics = pd.DataFrame(rows).sort_values(
        ["test_accuracy", "test_balanced_accuracy", "rmse_hybridmag"], ascending=[False, False, True]
    )
    metrics.to_csv(out_dir / "classification_refinement_metrics.csv", index=False)
    if best_pred_frame is not None:
        best_pred_frame.to_csv(out_dir / "best_sign_classifier_predictions.csv", index=False)
        _per_year_summary(best_pred_frame).to_csv(out_dir / "best_sign_classifier_per_year.csv", index=False)
    if best_coefficients:
        pd.DataFrame(best_coefficients).to_csv(out_dir / "best_sign_classifier_coefficients.csv", index=False)

    with (out_dir / "classification_refinement_config.json").open("w") as fh:
        json.dump(
            {
                "train_years": train_years,
                "test_years": test_years,
                "feature_sets": feature_sets,
                "target_specs": target_specs,
                "weight_modes": weight_modes,
                "train_min_abs_deltas": train_min_abs_deltas,
                "best_row": best_row,
            },
            fh,
            indent=2,
            default=str,
        )
    return {"out_dir": str(out_dir), "best": metrics.head(20).to_dict(orient="records")}


def _maybe_update_best(
    row: Dict[str, object],
    best_row: Dict[str, object] | None,
    best_pred_frame: pd.DataFrame | None,
    best_coefficients: List[Dict[str, object]],
    dy: pd.DataFrame,
    pred_sign: np.ndarray,
    prob: np.ndarray,
    fitted: object,
    cols: Sequence[str],
) -> Tuple[Dict[str, object] | None, pd.DataFrame | None, List[Dict[str, object]]]:
    key = (float(row["test_accuracy"]), float(row["test_balanced_accuracy"]), -float(row["rmse_hybridmag"]))
    best_key = (
        -1.0,
        -1.0,
        -float("inf"),
    )
    if best_row is not None:
        best_key = (
            float(best_row["test_accuracy"]),
            float(best_row["test_balanced_accuracy"]),
            -float(best_row["rmse_hybridmag"]),
        )
    if key <= best_key:
        return best_row, best_pred_frame, best_coefficients

    trend = dy["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    final_hybrid = dy["final_hybrid_pred"].fillna(dy["old_sota_pred"]).to_numpy(dtype=float)
    sign_mult = np.where(pred_sign >= 1, 1.0, -1.0)
    pred_y = trend + sign_mult * np.abs(final_hybrid - trend)
    out = dy[KEYS + ["split", "actual_yield_kg_per_ha", "trend_baseline_yield_kg_per_ha", "old_sota_pred", "final_hybrid_pred", "actual_sign", "old_sota_sign", "lag1_sign"]].copy()
    out["best_classifier_prob"] = prob
    out["best_classifier_sign"] = pred_sign
    out["best_classifier_flip_old_sota"] = (pred_sign != out["old_sota_sign"].to_numpy(dtype=int)).astype(int)
    out["best_classifier_hybridmag_prediction_kg_per_ha"] = pred_y
    meta = {
        "model": row.get("model"),
        "variant": row.get("variant"),
        "feature_set": row.get("feature_set"),
        "target_mode": row.get("target_mode"),
        "weight_mode": row.get("weight_mode"),
        "train_min_abs_delta": row.get("train_min_abs_delta"),
        "threshold": row.get("threshold"),
    }
    return row, out, _coefficient_rows(fitted, cols, meta)[:80]


def _per_year_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in pred_df[pred_df["season_year"].between(2019, 2022)].groupby("season_year"):
        y = g["actual_yield_kg_per_ha"].to_numpy(dtype=float)
        p = g["best_classifier_hybridmag_prediction_kg_per_ha"].to_numpy(dtype=float)
        sign = g["best_classifier_sign"].to_numpy(dtype=int)
        actual = g["actual_sign"].to_numpy(dtype=int)
        rows.append(
            {
                "season_year": int(year),
                "n": int(len(g)),
                "rmse": _rmse(y, p),
                "mae": float(mean_absolute_error(y, p)),
                "r2": float(r2_score(y, p)),
                "sign_accuracy": float((sign == actual).mean()),
                "drop_recall": float(((sign == 0) & (actual == 0)).sum() / max(1, int((actual == 0).sum()))),
                "rise_recall": float(((sign == 1) & (actual == 1)).sum() / max(1, int((actual == 1).sum()))),
                "flip_rate_old_sota": float(g["best_classifier_flip_old_sota"].mean()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused classification refinement for wheat yield rise/drop sign.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config-data", default="expansion_2010_workspace/configs/codex_data_v2_2010.yaml")
    parser.add_argument("--horizon-days", type=int, default=25)
    parser.add_argument("--opdate-profile", default="ten_day_dec1_apr30")
    parser.add_argument("--agri-economics-dir", default="expansion_2010_workspace/data/agri_economics_latest")
    parser.add_argument("--agri-year-lag", type=int, default=1)
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
    parser.add_argument("--train-years", nargs="+", type=int, default=list(range(2010, 2019)))
    parser.add_argument("--val-year", type=int, default=2018)
    parser.add_argument("--test-years", nargs="+", type=int, default=[2019, 2020, 2021, 2022])
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, default=str))
