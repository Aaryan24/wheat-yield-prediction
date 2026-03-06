from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def month_group_from_opdate(op_label: str) -> str:
    label = str(op_label)
    month = int(label.split("-")[0])
    if month in {12, 1}:
        return "dec_jan"
    if month == 2:
        return "feb"
    return "mar_apr"


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return 1.0, 0.0
    mat = np.column_stack([x, np.ones_like(x)])
    coeff, *_ = np.linalg.lstsq(mat, y, rcond=None)
    a = float(coeff[0])
    b = float(coeff[1])
    if not np.isfinite(a):
        a = 1.0
    if not np.isfinite(b):
        b = 0.0
    return a, b


def fit_rise_bias_calibrator(
    pred_raw: np.ndarray,
    actual_raw: np.ndarray,
    sample_opdates: Sequence[str],
    sample_splits: Sequence[str],
    target_mean: np.ndarray,
    a_min: float = 1.0,
    a_max: float = 1.6,
    b_min: float = 0.0,
    b_max: float = 120.0,
) -> Dict[str, object]:
    pred_raw = np.asarray(pred_raw, dtype=np.float32)
    actual_raw = np.asarray(actual_raw, dtype=np.float32)
    mean = np.asarray(target_mean, dtype=np.float32)

    groups = ["dec_jan", "feb", "mar_apr"]
    out: Dict[str, Dict[str, float]] = {}

    for grp in groups:
        xs = []
        ys = []
        for s in range(pred_raw.shape[0]):
            if str(sample_splits[s]) != "train":
                continue
            if month_group_from_opdate(str(sample_opdates[s])) != grp:
                continue
            pred_delta = pred_raw[s] - mean
            actual_delta = actual_raw[s] - mean
            mask = np.isfinite(pred_delta) & np.isfinite(actual_delta) & (pred_delta > 0.0)
            if np.any(mask):
                xs.append(pred_delta[mask])
                ys.append(actual_delta[mask])

        if xs:
            x = np.concatenate(xs).astype(np.float32)
            y = np.concatenate(ys).astype(np.float32)
            a, b = _fit_affine(x=x, y=y)
            n = int(len(x))
        else:
            a, b, n = 1.0, 0.0, 0

        a = float(np.clip(a, a_min, a_max))
        b = float(np.clip(b, b_min, b_max))
        out[grp] = {
            "a": a,
            "b": b,
            "n": int(n),
        }

    return {
        "groups": out,
        "a_min": float(a_min),
        "a_max": float(a_max),
        "b_min": float(b_min),
        "b_max": float(b_max),
    }


def apply_rise_bias_calibrator(
    pred_raw: np.ndarray,
    sample_opdates: Sequence[str],
    target_mean: np.ndarray,
    calibrator: Dict[str, object],
) -> np.ndarray:
    pred_raw = np.asarray(pred_raw, dtype=np.float32)
    out = pred_raw.copy().astype(np.float32)
    mean = np.asarray(target_mean, dtype=np.float32)
    groups = dict(calibrator.get("groups", {})) if isinstance(calibrator, dict) else {}

    for s in range(out.shape[0]):
        grp = month_group_from_opdate(str(sample_opdates[s]))
        coeff = groups.get(grp, {"a": 1.0, "b": 0.0})
        a = float(coeff.get("a", 1.0))
        b = float(coeff.get("b", 0.0))

        pred_delta = out[s] - mean
        mask = pred_delta > 0.0
        if not np.any(mask):
            continue
        adjusted = np.maximum(a * pred_delta[mask] + b, 0.0)
        out[s, mask] = mean[mask] + adjusted

    return out.astype(np.float32)
