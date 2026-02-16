from __future__ import annotations

import numpy as np

from codex_v2.src.data.build_dataset_v2 import apply_target_transform, compute_target_stats, fit_masked_scaler


def test_target_stats_use_train_only_indices() -> None:
    # S=3 samples, N=2 districts
    y_raw = np.array(
        [
            [100.0, 200.0],  # train
            [300.0, 500.0],  # train
            [9000.0, 9000.0],  # val/test outlier: must not affect stats
        ],
        dtype=np.float32,
    )
    train_idx = np.array([0, 1], dtype=np.int64)

    mean, std = compute_target_stats(y_raw=y_raw, train_idx=train_idx)

    np.testing.assert_allclose(mean, np.array([200.0, 350.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(std, np.array([100.0, 150.0], dtype=np.float32), atol=1e-6)

    z = apply_target_transform(y_raw=y_raw, target_mode="district_zscore", target_mean=mean, target_std=std)
    # Train rows become centered around 0; outlier remains large if transform is train-based.
    np.testing.assert_allclose(z[:2].mean(axis=0), np.array([0.0, 0.0], dtype=np.float32), atol=1e-6)
    assert z[2, 0] > 50.0


def test_feature_scaler_use_train_only_indices() -> None:
    # values shape [S, N, T, F] with one feature, one valid point per sample.
    values = np.array(
        [
            [[[1.0]]],   # train
            [[[3.0]]],   # train
            [[[100.0]]], # val/test outlier, should be excluded from fit
        ],
        dtype=np.float32,
    )
    mask = np.ones((3, 1, 1), dtype=np.float32)
    train_idx = np.array([0, 1], dtype=np.int64)

    mean, std = fit_masked_scaler(values=values, mask=mask, sample_idx=train_idx)
    np.testing.assert_allclose(mean, np.array([2.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(std, np.array([1.0], dtype=np.float32), atol=1e-6)
