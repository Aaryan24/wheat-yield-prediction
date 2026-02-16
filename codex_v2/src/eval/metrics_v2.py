from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true)
    valid = denom > 1e-6
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / denom[valid])) * 100.0)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = _safe_mape(y_true, y_pred)
    r2 = _safe_r2(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
    }
