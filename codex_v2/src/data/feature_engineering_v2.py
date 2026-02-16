from __future__ import annotations

from typing import List, Tuple

import numpy as np

WEATHER_BASE_FEATURES = [
    "tmax_mean",
    "tmax_std",
    "tmin_mean",
    "tmin_std",
    "tp_mean",
    "tp_std",
    "ssrd_mean",
    "ssrd_std",
    "wind_speed_mean",
    "wind_speed_std",
]

SAT_BASE_FEATURES = ["B7", "B8", "B8A", "B12"]


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return a / (b + eps)


def _cumulative_masked_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.cumsum(values * mask, axis=2)


def _mean_window(x: np.ndarray, start: int, end: int) -> np.ndarray:
    t = x.shape[2]
    lo = max(0, min(start, t - 1))
    hi = max(lo + 1, min(end, t))
    return x[:, :, lo:hi].mean(axis=2)


def engineer_weather_features(weather_x: np.ndarray, weather_mask: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    # weather_x: [S, N, T, F=10], weather_mask: [S, N, T]
    mask = weather_mask.astype(np.float32)
    tmax_c = weather_x[..., 0] - 273.15
    tmin_c = weather_x[..., 2] - 273.15
    tp = weather_x[..., 4]

    tmean_c = 0.5 * (tmax_c + tmin_c)
    gdd_daily = np.clip(tmean_c - 5.0, 0.0, None) * mask
    gdd_cum = np.cumsum(gdd_daily, axis=2)
    heat_stress_day = ((tmax_c > 32.0) * (mask > 0)).astype(np.float32)
    heat_stress_count = np.cumsum(heat_stress_day, axis=2)

    # Consecutive low-precipitation days among valid weather entries.
    dry_flag = ((tp < 1.0) * (mask > 0)).astype(np.float32)
    dry_spell = np.zeros_like(dry_flag, dtype=np.float32)
    for ti in range(dry_flag.shape[-1]):
        if ti == 0:
            dry_spell[..., ti] = dry_flag[..., ti]
        else:
            dry_spell[..., ti] = np.where(
                dry_flag[..., ti] > 0,
                dry_spell[..., ti - 1] + 1.0,
                0.0,
            )

    extra = np.stack([gdd_cum, heat_stress_count, dry_spell], axis=-1).astype(np.float32)
    out = np.concatenate([weather_x, extra], axis=-1)
    names = WEATHER_BASE_FEATURES + ["gdd_cum", "heat_stress_count", "dry_spell_len"]
    return out, names


def engineer_satellite_features(sat_x: np.ndarray, sat_mask: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    # sat_x: [S, N, T, F=4], sat_mask: [S, N, T]
    mask = sat_mask.astype(np.float32)
    b7 = sat_x[..., 0]
    b8 = sat_x[..., 1]
    b8a = sat_x[..., 2]
    b12 = sat_x[..., 3]

    ndvi_like = _safe_div(b8 - b7, b8 + b7)
    red_edge_ratio = _safe_div(b8a - b7, b8a + b7)
    swir_nir_ratio = _safe_div(b8 - b12, b8 + b12)
    cum_b8 = _cumulative_masked_sum(b8, mask)
    cum_b8a = _cumulative_masked_sum(b8a, mask)

    # Local slope and broad-window slopes to capture phenology trend.
    slope_b8 = np.zeros_like(b8, dtype=np.float32)
    slope_b8[..., 1:] = b8[..., 1:] - b8[..., :-1]
    early = _mean_window(b8, 0, 10)
    mid = _mean_window(b8, 10, 20)
    late = _mean_window(b8, 20, 30)
    slope_early_mid = (mid - early).astype(np.float32)
    slope_mid_late = (late - mid).astype(np.float32)
    slope_early_mid = np.repeat(slope_early_mid[:, :, None], sat_x.shape[2], axis=2)
    slope_mid_late = np.repeat(slope_mid_late[:, :, None], sat_x.shape[2], axis=2)

    extra = np.stack(
        [
            ndvi_like,
            red_edge_ratio,
            swir_nir_ratio,
            cum_b8,
            cum_b8a,
            slope_b8,
            slope_early_mid,
            slope_mid_late,
        ],
        axis=-1,
    ).astype(np.float32)

    out = np.concatenate([sat_x, extra], axis=-1)
    names = SAT_BASE_FEATURES + [
        "ndvi_like",
        "red_edge_ratio",
        "swir_nir_ratio",
        "cum_b8",
        "cum_b8a",
        "slope_b8",
        "slope_early_mid",
        "slope_mid_late",
    ]
    return out, names


def add_missingness_indicators(
    weather_x: np.ndarray,
    weather_mask: np.ndarray,
    sat_x: np.ndarray,
    sat_mask: np.ndarray,
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    # Per-branch valid ratio + cumulative density as extra features.
    s, n, tw, _ = weather_x.shape
    ts = sat_x.shape[2]

    w_valid_ratio = weather_mask.mean(axis=2, keepdims=True)  # [S,N,1]
    w_cum_density = np.cumsum(weather_mask, axis=2) / np.maximum(
        np.arange(1, tw + 1, dtype=np.float32).reshape(1, 1, tw), 1.0
    )
    w_valid_ratio_rep = np.repeat(w_valid_ratio, tw, axis=2)
    w_extra = np.stack([w_valid_ratio_rep, w_cum_density], axis=-1).astype(np.float32)

    s_valid_ratio = sat_mask.mean(axis=2, keepdims=True)
    s_cum_density = np.cumsum(sat_mask, axis=2) / np.maximum(
        np.arange(1, ts + 1, dtype=np.float32).reshape(1, 1, ts), 1.0
    )
    s_valid_ratio_rep = np.repeat(s_valid_ratio, ts, axis=2)
    s_extra = np.stack([s_valid_ratio_rep, s_cum_density], axis=-1).astype(np.float32)

    weather_out = np.concatenate([weather_x, w_extra], axis=-1)
    sat_out = np.concatenate([sat_x, s_extra], axis=-1)

    weather_extra_names = ["weather_valid_ratio", "weather_mask_density"]
    sat_extra_names = ["sat_valid_ratio", "sat_mask_density"]
    return weather_out, weather_extra_names, sat_out, sat_extra_names
