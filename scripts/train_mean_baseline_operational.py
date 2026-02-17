#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
import yaml

# Allow running the script directly from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DISTRICT_ALIASES = {
    # Haryana
    "gurgaon": "gurugram",
    "mewat": "nuh",
    "hisar": "hissar",
    "sonipat": "sonepat",
    "yamunanagar": "yamuna nagar",
    # Punjab
    "ferozepur": "firozpur",
    "sas nagar": "mohali",
    "s a s nagar": "mohali",
    "s a s nagar sahibzada ajit singh nagar": "mohali",
    "shaheed bhagat singh nagar": "nawan shehar",
    "shahid bhagat singh nagar": "nawan shehar",
    # Uttar Pradesh
    "budaun": "badaun",
    "kanpur nagar": "kanpur",
    "kheri": "lakhimpur kheri",
    "kushi nagar": "kushinagar",
    "mau": "maunathbhanjan",
    "siddharthnagar": "siddharth nagar",
    "bhadohi": "sant ravi das nagar",
    "sant ravidas nagar": "sant ravi das nagar",
    "sant kabeer nagar": "sant kabir nagar",
}

STATE_ALIASES = {
    "uttar pradesh": "Uttar Pradesh",
    "uttar_pradesh": "Uttar Pradesh",
    "haryana": "Haryana",
    "punjab": "Punjab",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _fmt_seconds(sec: float) -> str:
    sec_i = int(max(0, round(sec)))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _norm(text: str) -> str:
    x = str(text).strip().lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _canon_state(text: str) -> str:
    n = _norm(text)
    return STATE_ALIASES.get(n, str(text).strip())


def _canon_district(text: str) -> str:
    n = _norm(text)
    return DISTRICT_ALIASES.get(n, n)


def _load_district_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["district_id", "state_name", "district_name", "district_index"]].copy()
    df = df.sort_values("district_index").reset_index(drop=True)
    df["state_norm"] = df["state_name"].map(_norm)
    df["district_norm"] = df["district_name"].map(_canon_district)
    return df


def _load_yield_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["district_id", "season_start_year", "yield_kg_per_ha", "area_ha"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Yield file missing required columns: {missing}")
    return df[needed].copy()


def _season_split(
    years: Sequence[int],
    mode: str,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    years = sorted(years)
    if mode == "fixed":
        train = [y for y in years if y <= 2020]
        val = [y for y in years if y == 2021]
        test = [y for y in years if y == 2022]
        if not train or not val or not test:
            raise RuntimeError("Fixed split requires seasons including 2017-2022.")
        return train, val, test

    if len(years) < 4:
        raise RuntimeError("Random split requires at least 4 labeled seasons.")
    rng = random.Random(seed)
    shuffled = years.copy()
    rng.shuffle(shuffled)
    val = [shuffled[0]]
    test = [shuffled[1]]
    train = sorted(shuffled[2:])
    return train, val, test


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    e = y_pred - y_true
    rmse = float(np.sqrt(np.mean(e**2)))
    mae = float(np.mean(np.abs(e)))
    denom = np.abs(y_true)
    valid = denom > 1e-6
    if np.any(valid):
        mape = float(np.mean(np.abs(e[valid]) / denom[valid]) * 100.0)
    else:
        mape = float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _analyze_errors(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> Dict[str, int]:
    e = y_pred - y_true
    denom = np.abs(y_true)
    # MAPE in percentage points
    mape = np.zeros_like(e)
    valid = denom > 1e-6
    mape[valid] = (np.abs(e[valid]) / denom[valid]) * 100.0

    accurate = int(np.sum(mape < 2.0))
    moderate = int(np.sum((mape >= 2.0) & (mape <= 5.0)))
    serious = int(np.sum((mape > 5.0) & (mape <= 10.0)))
    extreme = int(np.sum(mape > 10.0))

    _log(f"[{label} Analysis] Total: {len(y_true)}")
    _log(f"  Accurate guess (< 2%): {accurate}")
    _log(f"  Moderate error (2-5%): {moderate}")
    _log(f"  Serious Error (5-10%): {serious}")
    _log(f"  Extreme error (> 10%): {extreme}")
    
    return {
        "count_accurate": accurate,
        "count_moderate": moderate,
        "count_serious": serious,
        "count_extreme": extreme,
    }


def _prediction_rows(
    split_name: str,
    years: Sequence[int],
    district_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    operational_label: str,
) -> pd.DataFrame:
    # y_true/y_pred: [S_split, N]
    rows: List[dict] = []
    district_id = district_df["district_id"].to_numpy()
    state_name = district_df["state_name"].to_numpy()
    district_name = district_df["district_name"].to_numpy()
    for yi, year in enumerate(years):
        for di in range(len(district_df)):
            rows.append(
                {
                    "operational_date": operational_label,
                    "split": split_name,
                    "season_year": int(year),
                    "district_id": str(district_id[di]),
                    "state_name": str(state_name[di]),
                    "district_name": str(district_name[di]),
                    "actual_yield_kg_per_ha": float(y_true[yi, di]),
                    "predicted_yield_kg_per_ha": float(y_pred[yi, di]),
                    "error_kg_per_ha": float(y_pred[yi, di] - y_true[yi, di]),
                    "abs_error_kg_per_ha": float(abs(y_pred[yi, di] - y_true[yi, di])),
                }
            )
    return pd.DataFrame(rows)


def _train_one_operational_day(
    operational_label: str,
    district_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    seasons: List[int],
    split_mode: str,
    split_seed: int,
    pred_out_dir: Optional[Path] = None,
) -> Dict[str, object]:
    
    district_ids = district_df["district_id"].tolist()
    
    # Organize yield data: [S, N]
    y_list: List[np.ndarray] = []
    for year in seasons:
        yy = (
            yield_df[yield_df["season_start_year"] == year]
            .set_index("district_id")
            .reindex(district_ids)["yield_kg_per_ha"]
            .to_numpy(dtype=np.float32)
        )
        if np.isnan(yy).any():
            _log(f"Warning: Missing yields for season {year}. Filling with 0 for alignment.")
            yy = np.nan_to_num(yy, nan=0.0)
        y_list.append(yy)
    
    y_arr = np.stack(y_list, axis=0) # [S, N]

    train_years, val_years, test_years = _season_split(
        years=seasons,
        mode=split_mode,
        seed=split_seed,
    )
    year_to_idx = {y: i for i, y in enumerate(seasons)}
    idx_train = np.array([year_to_idx[y] for y in train_years], dtype=int)
    
    train_y = y_arr[idx_train] # [S_train, N]
    
    # Calculate Mean Yield per District from Training Set
    # Handle zeros/NaNs if any were filled (though _load_yield_panel check usually prevents this)
    # We use masked array to ignore 0s if they represent missing data, but here valid yields > 0.
    mean_yields = np.zeros(len(district_ids), dtype=np.float32)
    for i in range(len(district_ids)):
        dist_data = train_y[:, i]
        valid_data = dist_data[dist_data > 0]
        if len(valid_data) > 0:
            mean_yields[i] = np.mean(valid_data)
        else:
            # Fallback to global mean if a district has no history in training set
            mean_yields[i] = np.mean(train_y[train_y > 0]) if np.any(train_y > 0) else 0.0

    # Predictions are just the broadcasted means
    # train preds: [S_train, N]
    train_pred = np.tile(mean_yields, (len(train_years), 1))
    
    # val preds
    idx_val = np.array([year_to_idx[y] for y in val_years], dtype=int)
    val_y = y_arr[idx_val]
    val_pred = np.tile(mean_yields, (len(val_years), 1))
    
    # test preds
    idx_test = np.array([year_to_idx[y] for y in test_years], dtype=int)
    test_y = y_arr[idx_test]
    test_pred = np.tile(mean_yields, (len(test_years), 1))

    # Flatten for metrics
    train_y_flat = train_y.ravel()
    train_pred_flat = train_pred.ravel()
    val_y_flat = val_y.ravel()
    val_pred_flat = val_pred.ravel()
    test_y_flat = test_y.ravel()
    test_pred_flat = test_pred.ravel()

    train_metrics = _metrics(train_y_flat, train_pred_flat)
    val_metrics = _metrics(val_y_flat, val_pred_flat)
    test_metrics = _metrics(test_y_flat, test_pred_flat)
    
    _log(f"Analyze: Test Set (Baseline Mean)")
    bucket_counts = _analyze_errors(test_y_flat, test_pred_flat, "Test Set (Baseline)")

    pred_path: Optional[Path] = None
    if pred_out_dir is not None:
        pred_out_dir.mkdir(parents=True, exist_ok=True)
        # Reshape back to [S, N] for row generation logic
        n_districts = len(district_df)
        
        # Helper to reshape 1D -> 2D
        def _reshape_pred(flat_pred, years_len):
             return flat_pred.reshape(years_len, n_districts)
             
        pred_frames = [
            _prediction_rows(
                split_name="train",
                years=train_years,
                district_df=district_df,
                y_true=_reshape_pred(train_y_flat, len(train_years)),
                y_pred=_reshape_pred(train_pred_flat, len(train_years)),
                operational_label=operational_label,
            ),
            _prediction_rows(
                split_name="val",
                years=val_years,
                district_df=district_df,
                y_true=_reshape_pred(val_y_flat, len(val_years)),
                y_pred=_reshape_pred(val_pred_flat, len(val_years)),
                operational_label=operational_label,
            ),
            _prediction_rows(
                split_name="test",
                years=test_years,
                district_df=district_df,
                y_true=_reshape_pred(test_y_flat, len(test_years)),
                y_pred=_reshape_pred(test_pred_flat, len(test_years)),
                operational_label=operational_label,
            ),
        ]
        pred_df = pd.concat(pred_frames, ignore_index=True)
        pred_key = operational_label.replace("/", "-")
        pred_path = pred_out_dir / f"predictions_baseline_opdate_{pred_key}.csv"
        pred_df.to_csv(pred_path, index=False)

    results = {
        "operational_date": operational_label,
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years,
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_mape": train_metrics["mape"],
        "train_r2": train_metrics["r2"],
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_mape": val_metrics["mape"],
        "val_r2": val_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_mape": test_metrics["mape"],
        "test_r2": test_metrics["r2"],
        "prediction_file": str(pred_path) if pred_path is not None else "",
    }
    results.update(bucket_counts)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Mean Yield Baseline model."
    )
    parser.add_argument("--districts", type=str, default="data/processed/s2s_district/districts.parquet")
    parser.add_argument("--yield-file", type=str, default="data/yields/apy_query_report_model_ready_119.csv")
    parser.add_argument("--split-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--years", type=int, nargs="*", default=[2017, 2018, 2019, 2020, 2021, 2022])
    # Operational dates argument is kept for compatibility/looping but logic is essentially same for all dates
    # unless we wanted to simulate 'available data' up to a date, but mean yield is usually year-based.
    # For simplicity, we treat it as static per season.
    parser.add_argument(
        "--operational-dates",
        type=str,
        nargs="*",
        default=["02-15"],
        help="Operational prediction dates (dummy for baseline as it is static).",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--out-dir", type=str, default="experiments/mean_baseline")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions" if args.save_predictions else None

    district_df = _load_district_table(Path(args.districts))
    yield_df = _load_yield_panel(Path(args.yield_file))

    seasons = sorted([int(y) for y in args.years])
    
    results: List[dict] = []
    all_t0 = time.perf_counter()
    
    # Even though baseline is static, we loop to output metrics for each requested 'op date' 
    # to match the output format of other scripts for easy comparison.
    for op_label in args.operational_dates:
        _log(f"\n=== Operational date: {op_label} ===")
        op_t0 = time.perf_counter()
        res = _train_one_operational_day(
            operational_label=op_label,
            district_df=district_df,
            yield_df=yield_df,
            seasons=seasons,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            pred_out_dir=pred_dir,
        )
        results.append(res)
        op_dt = time.perf_counter() - op_t0
        _log(
            f"op-{op_label}: val_rmse={res['val_rmse']:.3f}, val_mape={res['val_mape']:.2f}%, "
            f"test_rmse={res['test_rmse']:.3f}, test_mape={res['test_mape']:.2f}%, "
            f"time={_fmt_seconds(op_dt)}"
        )

    res_df = pd.DataFrame(results)
    csv_path = out_dir / f"operational_date_metrics_{int(time.time())}.csv"
    json_path = out_dir / f"operational_date_metrics_{int(time.time())}.json"
    res_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=2))
    _log(f"\nSaved metrics: {csv_path}")
    _log(f"Total runtime: {_fmt_seconds(time.perf_counter() - all_t0)}")


if __name__ == "__main__":
    main()
