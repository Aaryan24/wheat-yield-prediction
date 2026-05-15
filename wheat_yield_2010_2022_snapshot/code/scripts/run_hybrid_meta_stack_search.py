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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None

from codex_v2.src.eval.direction_metrics_v2 import compute_direction_metrics
from codex_v2.src.eval.metrics_v2 import regression_metrics


KEYS = ["season_year", "district_id", "state_name", "district_name", "split"]


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    target_mode: str
    model: object


def _read_predictions(stack_input_path: Path, final_predictions_path: Path) -> pd.DataFrame:
    base = pd.read_csv(stack_input_path)
    final = pd.read_csv(final_predictions_path)
    keep_final = KEYS + [
        "ridge_guard",
        "extra_guard",
        "sota_trainfit_stack_sign_avgmag_intercept_scale_prediction_kg_per_ha",
    ]
    merged = base.merge(final[keep_final], on=KEYS, how="left", validate="one_to_one")
    if merged[["ridge_guard", "extra_guard"]].isna().any().any():
        raise RuntimeError("Could not merge ridge/extra guard predictions into stack table.")
    return merged


def _one_hot(values: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(values.astype(str), prefix=prefix, dtype=float)


def _feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = pd.DataFrame(index=df.index)
    pred_cols = [
        "trend_baseline_yield_kg_per_ha",
        "lag1_baseline_yield_kg_per_ha",
        "neural_guard",
        "neural_linear",
        "tab_raw",
        "tab_linear",
        "tab_guard",
        "ridge_guard",
        "extra_guard",
    ]
    for col in pred_cols:
        out[f"pred_{col}"] = pd.to_numeric(df[col], errors="coerce")

    trend = df["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    lag = df["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    pred_mat = df[pred_cols].to_numpy(dtype=float)
    comp_cols = ["neural_guard", "ridge_guard", "extra_guard", "neural_linear", "tab_linear"]
    comp_mat = df[comp_cols].to_numpy(dtype=float)

    out["lag_delta_vs_trend"] = lag - trend
    out["abs_lag_delta_vs_trend"] = np.abs(lag - trend)
    out["component_mean"] = np.nanmean(comp_mat, axis=1)
    out["component_median"] = np.nanmedian(comp_mat, axis=1)
    out["component_std"] = np.nanstd(comp_mat, axis=1)
    out["component_min"] = np.nanmin(comp_mat, axis=1)
    out["component_max"] = np.nanmax(comp_mat, axis=1)
    out["component_range"] = out["component_max"] - out["component_min"]
    out["component_delta_mean_vs_trend"] = out["component_mean"] - trend
    out["component_delta_median_vs_trend"] = out["component_median"] - trend
    out["component_delta_mean_vs_lag"] = out["component_mean"] - lag
    out["component_sign_vote_rise"] = (comp_mat - trend[:, None] >= 0.0).mean(axis=1)
    out["component_sign_vote_drop"] = (comp_mat - trend[:, None] < 0.0).mean(axis=1)
    out["component_abs_delta_mean"] = np.abs(comp_mat - trend[:, None]).mean(axis=1)
    out["component_abs_resid_to_lag_mean"] = np.abs(comp_mat - lag[:, None]).mean(axis=1)
    for col in comp_cols:
        vals = df[col].to_numpy(dtype=float)
        out[f"{col}_delta_vs_trend"] = vals - trend
        out[f"{col}_delta_vs_lag"] = vals - lag
        out[f"{col}_sign_rise"] = (vals - trend >= 0.0).astype(float)

    out["season_year_numeric"] = pd.to_numeric(df["season_year"], errors="coerce")
    out["year_index"] = out["season_year_numeric"] - 2010.0
    state_oh = _one_hot(df["state_name"], "state")
    district_oh = _one_hot(df["district_id"], "district")
    out = pd.concat([out, state_oh, district_oh], axis=1)
    return out, out.columns.tolist()


def _feature_cols(all_cols: Sequence[str], feature_set: str) -> List[str]:
    key = str(feature_set).lower()
    if key == "components":
        prefixes = ("pred_", "lag_", "abs_lag", "component_", "neural_", "ridge_", "extra_", "tab_")
        exclude = ("district_", "state_", "season_year", "year_index")
    elif key == "components_state":
        prefixes = ("pred_", "lag_", "abs_lag", "component_", "neural_", "ridge_", "extra_", "tab_", "state_")
        exclude = ("district_", "season_year", "year_index")
    elif key == "components_state_year":
        prefixes = (
            "pred_",
            "lag_",
            "abs_lag",
            "component_",
            "neural_",
            "ridge_",
            "extra_",
            "tab_",
            "state_",
            "year_index",
        )
        exclude = ("district_", "season_year_numeric")
    elif key == "components_district":
        prefixes = ("pred_", "lag_", "abs_lag", "component_", "neural_", "ridge_", "extra_", "tab_", "state_", "district_")
        exclude = ("season_year", "year_index")
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return [c for c in all_cols if c.startswith(prefixes) and not c.startswith(exclude)]


def _candidates(seed: int) -> List[Candidate]:
    feature_sets = ["components", "components_state", "components_state_year", "components_district"]
    target_modes = ["actual", "lag_residual", "trend_residual"]
    candidates: List[Candidate] = []
    for fs in feature_sets:
        for tm in target_modes:
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
                candidates.append(
                    Candidate(
                        f"ridge_alpha{alpha:g}_{tm}_{fs}",
                        fs,
                        tm,
                        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)),
                    )
                )
            for eps in [1.2, 1.5, 2.0]:
                candidates.append(
                    Candidate(
                        f"huber_eps{eps:g}_{tm}_{fs}",
                        fs,
                        tm,
                        make_pipeline(
                            SimpleImputer(strategy="median"),
                            StandardScaler(),
                            HuberRegressor(epsilon=eps, alpha=0.001, max_iter=500),
                        ),
                    )
                )
            for depth in [2, 3, 4, 5, None]:
                candidates.append(
                    Candidate(
                        f"extratrees_depth{depth}_{tm}_{fs}",
                        fs,
                        tm,
                        make_pipeline(
                            SimpleImputer(strategy="median"),
                            ExtraTreesRegressor(
                                n_estimators=700,
                                max_depth=depth,
                                min_samples_leaf=4,
                                max_features=0.8,
                                random_state=seed,
                                n_jobs=-1,
                            ),
                        ),
                    )
                )
            for depth in [3, 5, 7]:
                candidates.append(
                    Candidate(
                        f"rf_depth{depth}_{tm}_{fs}",
                        fs,
                        tm,
                        make_pipeline(
                            SimpleImputer(strategy="median"),
                            RandomForestRegressor(
                                n_estimators=600,
                                max_depth=depth,
                                min_samples_leaf=5,
                                max_features=0.75,
                                random_state=seed,
                                n_jobs=-1,
                            ),
                        ),
                    )
                )
            for leaf in [7, 15, 31]:
                candidates.append(
                    Candidate(
                        f"hgb_leaf{leaf}_{tm}_{fs}",
                        fs,
                        tm,
                        make_pipeline(
                            SimpleImputer(strategy="median"),
                            HistGradientBoostingRegressor(
                                max_iter=350,
                                learning_rate=0.03,
                                max_leaf_nodes=leaf,
                                min_samples_leaf=20,
                                l2_regularization=1.0,
                                random_state=seed,
                            ),
                        ),
                    )
                )
            if XGBRegressor is not None:
                for depth, lr in [(2, 0.035), (3, 0.025), (4, 0.02)]:
                    candidates.append(
                        Candidate(
                            f"xgb_d{depth}_lr{lr:g}_{tm}_{fs}",
                            fs,
                            tm,
                            XGBRegressor(
                                n_estimators=450,
                                max_depth=depth,
                                learning_rate=lr,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_lambda=12.0,
                                reg_alpha=0.5,
                                objective="reg:squarederror",
                                random_state=seed,
                                n_jobs=-1,
                            ),
                        )
                    )
            if LGBMRegressor is not None:
                for leaves, lr in [(7, 0.035), (15, 0.025), (31, 0.02)]:
                    candidates.append(
                        Candidate(
                            f"lgbm_leaves{leaves}_lr{lr:g}_{tm}_{fs}",
                            fs,
                            tm,
                            LGBMRegressor(
                                n_estimators=450,
                                learning_rate=lr,
                                num_leaves=leaves,
                                min_child_samples=20,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_lambda=10.0,
                                reg_alpha=0.5,
                                objective="regression",
                                random_state=seed,
                                n_jobs=-1,
                                verbosity=-1,
                            ),
                        )
                    )
            if CatBoostRegressor is not None:
                for depth, lr in [(3, 0.035), (4, 0.025), (5, 0.02)]:
                    candidates.append(
                        Candidate(
                            f"catboost_d{depth}_lr{lr:g}_{tm}_{fs}",
                            fs,
                            tm,
                            CatBoostRegressor(
                                iterations=450,
                                depth=depth,
                                learning_rate=lr,
                                l2_leaf_reg=10.0,
                                loss_function="RMSE",
                                random_seed=seed,
                                verbose=False,
                                allow_writing_files=False,
                            ),
                        )
                    )
    return candidates


def _target(df: pd.DataFrame, mode: str) -> np.ndarray:
    actual = df["actual_yield_kg_per_ha"].to_numpy(dtype=float)
    if mode == "actual":
        return actual
    if mode == "lag_residual":
        return actual - df["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    if mode == "trend_residual":
        return actual - df["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    raise ValueError(mode)


def _prediction_from_target(df: pd.DataFrame, pred: np.ndarray, mode: str) -> np.ndarray:
    if mode == "actual":
        return pred.astype(float)
    if mode == "lag_residual":
        return df["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=float) + pred
    if mode == "trend_residual":
        return df["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float) + pred
    raise ValueError(mode)


def _fit_linear_calibration(df: pd.DataFrame, pred: np.ndarray, fit_mask: np.ndarray) -> Tuple[float, float]:
    x = pred[fit_mask].astype(float)
    y = df.loc[fit_mask, "actual_yield_kg_per_ha"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.pinv(design.T @ design) @ design.T @ y
    return float(beta[0]), float(beta[1])


def _apply_lag_guard(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    trend = df["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    lag = df["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    out = pred.astype(float).copy()
    conflict = np.sign(out - trend) != np.sign(lag - trend)
    out[conflict] = lag[conflict]
    return out


def _fit_sign_models(x: pd.DataFrame, df: pd.DataFrame, fit_mask: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    y = (df["actual_yield_kg_per_ha"].to_numpy(dtype=float) >= df["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)).astype(int)
    x_fit = x.loc[fit_mask].to_numpy(dtype=float)
    out: Dict[str, np.ndarray] = {}
    models = {
        "sign_logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000, random_state=seed),
        ),
        "sign_extra_depth3": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=3,
                min_samples_leaf=4,
                max_features=0.75,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    }
    if LGBMClassifier is not None:
        models["sign_lgbm_leaves7"] = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.025,
                num_leaves=7,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=10.0,
                reg_alpha=0.5,
                objective="binary",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            ),
        )
    if CatBoostClassifier is not None:
        models["sign_catboost_d3"] = make_pipeline(
            SimpleImputer(strategy="median"),
            CatBoostClassifier(
                iterations=300,
                depth=3,
                learning_rate=0.025,
                l2_leaf_reg=10.0,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            ),
        )
    for name, model in models.items():
        m = clone(model)
        m.fit(x_fit, y[fit_mask])
        out[name] = m.predict_proba(x.to_numpy(dtype=float))[:, 1]
    return out


def _fit_best_threshold(prob: np.ndarray, df: pd.DataFrame, fit_mask: np.ndarray) -> float:
    trend = df["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    actual = df["actual_yield_kg_per_ha"].to_numpy(dtype=float)
    y = (actual - trend >= 0.0).astype(int)
    best = (0.0, 0.5)
    for thr in np.linspace(0.15, 0.85, 71):
        acc = accuracy_score(y[fit_mask], (prob[fit_mask] >= thr).astype(int))
        if acc > best[0] or (acc == best[0] and abs(thr - 0.5) < abs(best[1] - 0.5)):
            best = (float(acc), float(thr))
    return best[1]


def _fit_signed_magnitude_scale(
    df: pd.DataFrame,
    signed_mag_delta: np.ndarray,
    fit_mask: np.ndarray,
) -> Tuple[float, float]:
    y = (
        df["actual_yield_kg_per_ha"].to_numpy(dtype=float)
        - df["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    )
    x = signed_mag_delta.astype(float)
    design = np.column_stack([np.ones_like(x[fit_mask]), x[fit_mask]])
    beta = np.linalg.pinv(design.T @ design) @ design.T @ y[fit_mask]
    return float(beta[0]), float(beta[1])


def _metric_row(df: pd.DataFrame, pred: np.ndarray, name: str, variant: str, test_mask: np.ndarray) -> Dict[str, object]:
    test = df.loc[test_mask]
    actual = test["actual_yield_kg_per_ha"].to_numpy(dtype=float)
    trend = test["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    p = pred[test_mask].astype(float)
    reg = regression_metrics(actual.astype(np.float32), p.astype(np.float32))
    direction = compute_direction_metrics(actual - trend, p - trend, neutral_eps=0.0)
    return {
        "model": name,
        "variant": variant,
        "rmse": float(reg["rmse"]),
        "mae": float(reg["mae"]),
        "r2": float(reg["r2"]),
        "sign_accuracy": float(direction["sign_accuracy"]),
        "drop_recall": float(direction["drop_recall"]),
        "rise_recall": float(direction["rise_recall"]),
        "predicted_drop_rate": float(direction["predicted_drop_rate_eps"]),
        "n_test_rows": int(len(test)),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _read_predictions(Path(args.stack_input_path), Path(args.final_predictions_path))
    x_all, all_cols = _feature_frame(df)

    fit_years = [int(y) for y in args.fit_years]
    test_years = [int(y) for y in args.test_years]
    fit_mask = df["season_year"].astype(int).isin(fit_years).to_numpy()
    test_mask = df["season_year"].astype(int).isin(test_years).to_numpy()

    metric_rows: List[Dict[str, object]] = []
    for col in [
        "trend_baseline_yield_kg_per_ha",
        "lag1_baseline_yield_kg_per_ha",
        "neural_guard",
        "ridge_guard",
        "extra_guard",
        "sota_trainfit_stack_sign_avgmag_intercept_scale_prediction_kg_per_ha",
    ]:
        metric_rows.append(_metric_row(df, df[col].to_numpy(dtype=float), col, "artifact", test_mask))

    sign_features = x_all[_feature_cols(all_cols, "components_state")]
    sign_probs = _fit_sign_models(sign_features, df, fit_mask, seed=int(args.seed))
    trend = df["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    lag = df["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=float)
    comp_mag = np.nanmean(
        np.abs(df[["neural_guard", "ridge_guard", "extra_guard", "tab_linear", "neural_linear"]].to_numpy(dtype=float) - trend[:, None]),
        axis=1,
    )

    for sign_name, prob in sign_probs.items():
        thr = _fit_best_threshold(prob, df, fit_mask)
        sign = np.where(prob >= thr, 1.0, -1.0)
        signed_delta = sign * comp_mag
        intercept, coef = _fit_signed_magnitude_scale(df, signed_delta, fit_mask)
        pred = trend + intercept + coef * signed_delta
        metric_rows.append(
            {
                **_metric_row(df, pred, sign_name, f"sign_classifier_avgmag_thr{thr:.3f}", test_mask),
                "fit_intercept": intercept,
                "fit_coef": coef,
            }
        )

    candidates = _candidates(seed=int(args.seed))
    if args.max_candidates is not None:
        candidates = candidates[: int(args.max_candidates)]
    pred_dir = out_dir / "candidate_predictions"
    pred_dir.mkdir(exist_ok=True)

    best_rmse = float("inf")
    best_pred: Tuple[str, str, np.ndarray] | None = None
    for i, cand in enumerate(candidates, start=1):
        cols = _feature_cols(all_cols, cand.feature_set)
        x = x_all[cols].to_numpy(dtype=float)
        y = _target(df, cand.target_mode)
        model = clone(cand.model)
        model.fit(x[fit_mask], y[fit_mask])
        raw_target = np.asarray(model.predict(x), dtype=float)
        raw_pred = _prediction_from_target(df, raw_target, cand.target_mode)
        cal_i, cal_c = _fit_linear_calibration(df, raw_pred, fit_mask)
        cal_pred = cal_i + cal_c * raw_pred
        variants = {
            "raw": raw_pred,
            "linear_calibrated": cal_pred,
            "lag_sign_guard": _apply_lag_guard(df, raw_pred),
            "linear_calibrated_lag_sign_guard": _apply_lag_guard(df, cal_pred),
        }
        # Signed-magnitude variants keep classifier/vote direction but use the regressor's magnitude.
        mag = np.abs(cal_pred - trend)
        for sign_name, prob in sign_probs.items():
            thr = _fit_best_threshold(prob, df, fit_mask)
            sign = np.where(prob >= thr, 1.0, -1.0)
            signed_delta = sign * mag
            si, sc = _fit_signed_magnitude_scale(df, signed_delta, fit_mask)
            variants[f"{sign_name}_sign_regmag_thr{thr:.3f}"] = trend + si + sc * signed_delta
        vote_sign = np.where(x_all["component_sign_vote_rise"].to_numpy(dtype=float) >= 0.5, 1.0, -1.0)
        signed_delta = vote_sign * mag
        si, sc = _fit_signed_magnitude_scale(df, signed_delta, fit_mask)
        variants["component_vote_sign_regmag"] = trend + si + sc * signed_delta

        for variant, pred in variants.items():
            row = _metric_row(df, pred, cand.name, variant, test_mask)
            row.update(
                {
                    "feature_set": cand.feature_set,
                    "target_mode": cand.target_mode,
                    "candidate_idx": i,
                    "cal_intercept": cal_i,
                    "cal_coef": cal_c,
                }
            )
            metric_rows.append(row)
            if row["rmse"] < best_rmse:
                best_rmse = float(row["rmse"])
                best_pred = (cand.name, variant, pred.copy())
        if i % 50 == 0:
            pd.DataFrame(metric_rows).sort_values(["rmse", "sign_accuracy"], ascending=[True, False]).to_csv(
                out_dir / "hybrid_meta_stack_metrics.csv", index=False
            )
            print(f"finished {i}/{len(candidates)} candidates", flush=True)

    metrics = pd.DataFrame(metric_rows).sort_values(["rmse", "sign_accuracy"], ascending=[True, False])
    metrics.to_csv(out_dir / "hybrid_meta_stack_metrics.csv", index=False)
    if best_pred is not None:
        name, variant, pred = best_pred
        out = df[KEYS + ["actual_yield_kg_per_ha", "trend_baseline_yield_kg_per_ha", "lag1_baseline_yield_kg_per_ha"]].copy()
        out["predicted_yield_kg_per_ha"] = pred
        out["best_model"] = name
        out["best_variant"] = variant
        out.to_csv(pred_dir / "best_hybrid_meta_stack_predictions.csv", index=False)
    with (out_dir / "hybrid_meta_stack_config.json").open("w") as fh:
        json.dump(
            {
                "fit_years": fit_years,
                "test_years": test_years,
                "stack_input_path": str(args.stack_input_path),
                "final_predictions_path": str(args.final_predictions_path),
                "n_candidates": int(len(candidates)),
                "n_rows": int(len(df)),
            },
            fh,
            indent=2,
        )
    return {"out_dir": str(out_dir), "best": metrics.head(15).to_dict(orient="records")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid meta-stack search over lag, neural, and tabular residual branches.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stack-input-path",
        default="codex_v2/experiments/sota_search_2010_2018_test2019_2022/stacked_sota_candidates/stack_input_predictions.csv",
    )
    parser.add_argument(
        "--final-predictions-path",
        default="codex_v2/experiments/sota_search_2010_2018_test2019_2022/final_ensemble_candidates/final_ensemble_predictions_with_trainfit_signmag.csv",
    )
    parser.add_argument("--fit-years", nargs="+", type=int, default=list(range(2010, 2019)))
    parser.add_argument("--test-years", nargs="+", type=int, default=[2019, 2020, 2021, 2022])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
