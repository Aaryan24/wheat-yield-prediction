#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional local dependency
    XGBRegressor = None

from codex_v2.src.data.build_dataset_v2 import DatasetBundle, build_dataset_v2
from codex_v2.src.eval.direction_metrics_v2 import compute_direction_metrics
from codex_v2.src.eval.metrics_v2 import regression_metrics


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    model: object


def _masked_stats(
    x: np.ndarray,
    mask: np.ndarray,
    feature_names: Sequence[str],
    prefix: str,
    *,
    include_segments: bool,
) -> Tuple[np.ndarray, List[str]]:
    valid = (mask > 0.5)[..., None].astype(np.float32)
    count = valid.sum(axis=2).clip(min=1.0)
    mean = (x * valid).sum(axis=2) / count
    centered = (x - mean[:, :, None, :]) * valid
    std = np.sqrt((centered**2).sum(axis=2) / count)

    x_min = np.where(valid > 0.0, x, np.inf).min(axis=2)
    x_max = np.where(valid > 0.0, x, -np.inf).max(axis=2)
    no_valid = (count[..., 0] <= 0.0)
    x_min[no_valid] = 0.0
    x_max[no_valid] = 0.0

    arrays = [mean, std, x_min, x_max]
    suffixes = ["mean", "std", "min", "max"]

    if include_segments:
        t_len = int(x.shape[2])
        edges = np.linspace(0, t_len, 4, dtype=int)
        for seg_idx in range(3):
            lo, hi = int(edges[seg_idx]), int(edges[seg_idx + 1])
            seg_valid = valid[:, :, lo:hi, :]
            seg_count = seg_valid.sum(axis=2).clip(min=1.0)
            seg_mean = (x[:, :, lo:hi, :] * seg_valid).sum(axis=2) / seg_count
            arrays.append(seg_mean)
            suffixes.append(f"seg{seg_idx + 1}_mean")

    out = np.concatenate(arrays, axis=-1).astype(np.float32)
    names = [f"{prefix}_{name}_{suffix}" for suffix in suffixes for name in feature_names]
    return out, names


def _one_hot(values: Sequence[str], prefix: str) -> Tuple[np.ndarray, List[str]]:
    vals = np.asarray([str(v) for v in values], dtype=object)
    cats = sorted(set(vals.tolist()))
    mat = np.zeros((len(vals), len(cats)), dtype=np.float32)
    cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
    for row, val in enumerate(vals.tolist()):
        mat[row, cat_to_idx[val]] = 1.0
    return mat, [f"{prefix}_{cat}" for cat in cats]


def _build_feature_table(bundle: DatasetBundle) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    weather_stats, weather_names = _masked_stats(
        bundle.weather_x,
        bundle.weather_mask,
        bundle.weather_feature_names,
        "weather",
        include_segments=True,
    )
    sat_stats, sat_names = _masked_stats(
        bundle.sat_x,
        bundle.sat_mask,
        bundle.sat_feature_names,
        "sat",
        include_segments=False,
    )

    s_count, n_nodes = int(bundle.y_raw.shape[0]), int(bundle.y_raw.shape[1])
    district_ids = bundle.district_df["district_id"].astype(str).to_numpy()
    state_names = bundle.district_df["state_name"].astype(str).to_numpy()
    district_names = bundle.district_df["district_name"].astype(str).to_numpy()
    state_oh, state_oh_names = _one_hot(state_names, "state")
    district_oh, district_oh_names = _one_hot(district_ids, "district")

    lag_delta = (
        bundle.lag1_baseline_delta
        if bundle.lag1_baseline_delta is not None
        else np.zeros_like(bundle.y_raw, dtype=np.float32)
    )
    lag_mask = (
        bundle.lag1_baseline_mask
        if bundle.lag1_baseline_mask is not None
        else np.zeros_like(bundle.y_raw, dtype=np.float32)
    )
    lag_yield = (
        bundle.target_transform_mean
        if bundle.target_transform_mean is not None
        else bundle.target_mean
    ).astype(np.float32)

    base_features = np.stack(
        [
            bundle.target_mean.astype(np.float32),
            lag_yield,
            lag_delta.astype(np.float32),
            lag_mask.astype(np.float32),
            np.repeat(bundle.sample_opdate_idx[:, None], n_nodes, axis=1).astype(np.float32),
        ],
        axis=-1,
    )
    base_names = [
        "base_trend_yield",
        "base_lag1_yield",
        "base_lag1_delta",
        "base_lag1_available",
        "base_opdate_idx",
    ]

    agri_names = [f"agri_{name}" for name in bundle.agri_feature_names]
    agri_mask_names = [f"agri_mask_{name}" for name in bundle.agri_feature_names[: bundle.agri_mask.shape[-1]]]

    feature_blocks: List[np.ndarray] = [
        base_features,
        weather_stats,
        sat_stats,
        bundle.agri_x.astype(np.float32),
        bundle.agri_mask.astype(np.float32),
    ]
    feature_names = base_names + weather_names + sat_names + agri_names + agri_mask_names

    state_block = np.repeat(state_oh[None, :, :], s_count, axis=0)
    district_block = np.repeat(district_oh[None, :, :], s_count, axis=0)
    feature_blocks.extend([state_block.astype(np.float32), district_block.astype(np.float32)])
    feature_names.extend(state_oh_names + district_oh_names)

    x = np.concatenate(feature_blocks, axis=-1).astype(np.float32)
    rows: List[dict] = []
    split_name = np.array(["unknown"] * s_count, dtype=object)
    split_name[bundle.train_idx] = "train"
    split_name[bundle.val_idx] = "val"
    split_name[bundle.test_idx] = "test"

    for s in range(s_count):
        for n in range(n_nodes):
            trend = float(bundle.target_mean[s, n])
            lag_base = float(lag_yield[s, n])
            actual = float(bundle.y_raw[s, n])
            rows.append(
                {
                    "sample_idx": int(s),
                    "node_idx": int(n),
                    "split": str(split_name[s]),
                    "season_year": int(bundle.sample_years[s]),
                    "operational_date": str(bundle.sample_operational_dates[s]),
                    "district_id": str(district_ids[n]),
                    "state_name": str(state_names[n]),
                    "district_name": str(district_names[n]),
                    "actual_yield_kg_per_ha": actual,
                    "trend_baseline_yield_kg_per_ha": trend,
                    "lag1_baseline_yield_kg_per_ha": lag_base,
                    "actual_delta_kg_per_ha": actual - trend,
                    "lag1_baseline_delta_kg_per_ha": lag_base - trend,
                    "lag1_residual_error_kg_per_ha": actual - lag_base,
                }
            )

    return pd.DataFrame(rows), x.reshape(s_count * n_nodes, x.shape[-1]), feature_names


def _feature_indices(feature_names: Sequence[str], feature_set: str) -> np.ndarray:
    key = str(feature_set).strip().lower()
    prefixes = {
        "all": ("base_", "weather_", "sat_", "agri_", "agri_mask_", "state_", "district_"),
        "weather_agri": ("base_", "weather_", "agri_", "agri_mask_", "state_", "district_"),
        "weather_only": ("base_", "weather_", "state_", "district_"),
        "agri_only": ("base_", "agri_", "agri_mask_", "state_", "district_"),
        "baseline_only": ("base_", "state_", "district_"),
        "no_district": ("base_", "weather_", "sat_", "agri_", "agri_mask_", "state_"),
    }
    if key not in prefixes:
        raise ValueError(f"Unknown feature set: {feature_set}")
    keep = [idx for idx, name in enumerate(feature_names) if str(name).startswith(prefixes[key])]
    return np.asarray(keep, dtype=np.int64)


def _district_year_predictions(row_df: pd.DataFrame, row_pred: np.ndarray, label: str) -> pd.DataFrame:
    work = row_df.copy()
    work[f"{label}_yield_kg_per_ha"] = (
        work["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=np.float32) + np.asarray(row_pred, dtype=np.float32)
    )
    gcols = ["season_year", "district_id", "state_name", "district_name"]
    return (
        work.groupby(gcols, dropna=False)
        .agg(
            split=("split", "first"),
            actual_yield_kg_per_ha=("actual_yield_kg_per_ha", "first"),
            trend_baseline_yield_kg_per_ha=("trend_baseline_yield_kg_per_ha", "first"),
            lag1_baseline_yield_kg_per_ha=("lag1_baseline_yield_kg_per_ha", "first"),
            predicted_yield_kg_per_ha=(f"{label}_yield_kg_per_ha", "mean"),
            n_opdates=("operational_date", "nunique"),
        )
        .reset_index()
    )


def _fit_linear_residual_calibration(dy: pd.DataFrame, fit_years: Sequence[int]) -> Dict[str, float]:
    fit_df = dy[dy["season_year"].astype(int).isin([int(y) for y in fit_years])]
    x = (
        fit_df["predicted_yield_kg_per_ha"].to_numpy(dtype=np.float64)
        - fit_df["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=np.float64)
    )
    y = (
        fit_df["actual_yield_kg_per_ha"].to_numpy(dtype=np.float64)
        - fit_df["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=np.float64)
    )
    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.pinv(design.T @ design) @ design.T @ y
    return {"intercept": float(beta[0]), "coef": float(beta[1]), "fit_rows": int(len(fit_df))}


def _apply_calibration_and_guard(dy: pd.DataFrame, fit: Dict[str, float]) -> pd.DataFrame:
    out = dy.copy()
    raw_resid = (
        out["predicted_yield_kg_per_ha"].to_numpy(dtype=np.float32)
        - out["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=np.float32)
    )
    calibrated = (
        out["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=np.float32)
        + float(fit["intercept"])
        + (float(fit["coef"]) * raw_resid)
    ).astype(np.float32)
    trend = out["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=np.float32)
    lag = out["lag1_baseline_yield_kg_per_ha"].to_numpy(dtype=np.float32)
    conflict = np.sign(calibrated - trend) != np.sign(lag - trend)
    guarded = calibrated.copy()
    guarded[conflict] = lag[conflict]
    out["predicted_yield_calibrated_kg_per_ha"] = calibrated
    out["predicted_yield_guarded_kg_per_ha"] = guarded
    out["guard_fallback"] = conflict
    return out


def _metric_row(dy: pd.DataFrame, pred_col: str, model: str, variant: str) -> Dict[str, object]:
    test = dy[dy["split"] == "test"].copy()
    actual = test["actual_yield_kg_per_ha"].to_numpy(dtype=np.float32)
    trend = test["trend_baseline_yield_kg_per_ha"].to_numpy(dtype=np.float32)
    pred = test[pred_col].to_numpy(dtype=np.float32)
    reg = regression_metrics(actual, pred)
    direction = compute_direction_metrics(actual - trend, pred - trend, neutral_eps=0.0)
    return {
        "model": str(model),
        "variant": str(variant),
        "rmse": float(reg["rmse"]),
        "mae": float(reg["mae"]),
        "r2": float(reg["r2"]),
        "sign_accuracy": float(direction["sign_accuracy"]),
        "drop_recall": float(direction["drop_recall"]),
        "rise_recall": float(direction["rise_recall"]),
        "predicted_drop_rate": float(direction["predicted_drop_rate_eps"]),
        "n_test_rows": int(len(test)),
    }


def _baseline_metric_rows(row_df: pd.DataFrame) -> List[Dict[str, object]]:
    dy = (
        row_df.groupby(["season_year", "district_id", "state_name", "district_name"], dropna=False)
        .agg(
            split=("split", "first"),
            actual_yield_kg_per_ha=("actual_yield_kg_per_ha", "first"),
            trend_baseline_yield_kg_per_ha=("trend_baseline_yield_kg_per_ha", "first"),
            lag1_baseline_yield_kg_per_ha=("lag1_baseline_yield_kg_per_ha", "first"),
        )
        .reset_index()
    )
    rows = []
    for col, model in [
        ("trend_baseline_yield_kg_per_ha", "trend_baseline"),
        ("lag1_baseline_yield_kg_per_ha", "lag1_baseline"),
    ]:
        scored = dy.copy()
        scored["pred"] = scored[col].astype(np.float32)
        rows.append(_metric_row(scored, "pred", model, "raw"))
    return rows


def _make_candidates(seed: int) -> List[Candidate]:
    feature_sets = ["baseline_only", "agri_only", "weather_agri", "all", "no_district"]
    candidates: List[Candidate] = []
    for fs in feature_sets:
        for alpha in [1.0, 10.0, 100.0, 1000.0]:
            candidates.append(
                Candidate(
                    name=f"ridge_alpha{alpha:g}_{fs}",
                    feature_set=fs,
                    model=make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)),
                )
            )
        for depth in [4, 8, None]:
            candidates.append(
                Candidate(
                    name=f"extra_trees_depth{depth}_{fs}",
                    feature_set=fs,
                    model=ExtraTreesRegressor(
                        n_estimators=450,
                        max_depth=depth,
                        min_samples_leaf=4,
                        max_features=0.65,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            )
        for leaf_nodes in [15, 31]:
            candidates.append(
                Candidate(
                    name=f"hgb_leaf{leaf_nodes}_{fs}",
                    feature_set=fs,
                    model=make_pipeline(
                        SimpleImputer(strategy="median"),
                        HistGradientBoostingRegressor(
                            max_iter=350,
                            learning_rate=0.035,
                            max_leaf_nodes=leaf_nodes,
                            min_samples_leaf=20,
                            l2_regularization=1.0,
                            random_state=seed,
                        ),
                    ),
                )
            )
        candidates.append(
            Candidate(
                name=f"rf_depth8_{fs}",
                feature_set=fs,
                model=RandomForestRegressor(
                    n_estimators=450,
                    max_depth=8,
                    min_samples_leaf=5,
                    max_features=0.6,
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
        if XGBRegressor is not None:
            for depth, lr in [(2, 0.035), (3, 0.025), (4, 0.02)]:
                candidates.append(
                    Candidate(
                        name=f"xgb_d{depth}_lr{lr:g}_{fs}",
                        feature_set=fs,
                        model=XGBRegressor(
                            n_estimators=500,
                            max_depth=depth,
                            learning_rate=lr,
                            subsample=0.85,
                            colsample_bytree=0.75,
                            reg_lambda=12.0,
                            reg_alpha=0.5,
                            objective="reg:squarederror",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    )
                )
    return candidates


def _fit_predict_candidate(
    cand: Candidate,
    row_df: pd.DataFrame,
    x: np.ndarray,
    feature_names: Sequence[str],
    train_years: Sequence[int],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    idx = _feature_indices(feature_names, cand.feature_set)
    train_mask = row_df["season_year"].astype(int).isin([int(y) for y in train_years]).to_numpy()
    y = row_df["lag1_residual_error_kg_per_ha"].to_numpy(dtype=np.float32)
    cand.model.fit(x[train_mask][:, idx], y[train_mask])
    pred_resid = np.asarray(cand.model.predict(x[:, idx]), dtype=np.float32)
    dy = _district_year_predictions(row_df, pred_resid, label="tabular")
    fit = _fit_linear_residual_calibration(dy, fit_years=train_years)
    dy = _apply_calibration_and_guard(dy, fit)
    return dy, fit


def run_search(args: argparse.Namespace) -> Dict[str, object]:
    train_years = [int(y) for y in args.train_years]
    test_years = [int(y) for y in args.test_years]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    row_df, x, feature_names = _build_feature_table(bundle)
    row_df.to_parquet(out_dir / "tabular_feature_rows_meta.parquet", index=False)
    pd.Series(feature_names, name="feature_name").to_csv(out_dir / "tabular_feature_names.csv", index=False)

    metric_rows: List[Dict[str, object]] = _baseline_metric_rows(row_df)
    prediction_dir = out_dir / "district_year_predictions"
    prediction_dir.mkdir(exist_ok=True)

    candidates = _make_candidates(seed=int(args.seed))
    if args.max_candidates is not None:
        candidates = candidates[: int(args.max_candidates)]

    for i, cand in enumerate(candidates, start=1):
        dy, fit = _fit_predict_candidate(cand, row_df, x, feature_names, train_years)
        for pred_col, variant in [
            ("predicted_yield_kg_per_ha", "raw"),
            ("predicted_yield_calibrated_kg_per_ha", "linear_calibrated"),
            ("predicted_yield_guarded_kg_per_ha", "linear_calibrated_lag_sign_guard"),
        ]:
            row = _metric_row(dy, pred_col, cand.name, variant)
            row.update(
                {
                    "feature_set": cand.feature_set,
                    "cal_intercept": fit["intercept"],
                    "cal_coef": fit["coef"],
                    "cal_fit_rows": fit["fit_rows"],
                    "candidate_idx": i,
                }
            )
            metric_rows.append(row)
        if i <= int(args.save_top_candidate_predictions):
            dy.to_csv(prediction_dir / f"{i:03d}_{cand.name}.csv", index=False)
        if i % 10 == 0:
            pd.DataFrame(metric_rows).sort_values(["rmse", "sign_accuracy"], ascending=[True, False]).to_csv(
                out_dir / "tabular_search_metrics.csv",
                index=False,
            )
            print(f"finished {i}/{len(candidates)} candidates", flush=True)

    metrics = pd.DataFrame(metric_rows).sort_values(["rmse", "sign_accuracy"], ascending=[True, False])
    metrics.to_csv(out_dir / "tabular_search_metrics.csv", index=False)
    with (out_dir / "tabular_search_config.json").open("w") as fh:
        json.dump(
            {
                "config_data": str(args.config_data),
                "horizon_days": int(args.horizon_days),
                "opdate_profile": str(args.opdate_profile),
                "train_years": train_years,
                "test_years": test_years,
                "seed": int(args.seed),
                "n_candidates": int(len(candidates)),
                "n_rows": int(len(row_df)),
                "n_features": int(x.shape[1]),
                "lag1_baseline": bundle.config_resolved.get("residual_baseline", {}),
            },
            fh,
            indent=2,
        )
    return {
        "out_dir": str(out_dir),
        "n_candidates": int(len(candidates)),
        "best": metrics.head(10).to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search tabular residual models over lag-1 wheat yield baseline.")
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
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--save-top-candidate-predictions", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    result = run_search(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
