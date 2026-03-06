from __future__ import annotations

import numpy as np
import torch

from codex_v2.src.data.build_dataset_v2 import apply_target_transform
from codex_v2.src.eval.calibration_v2 import apply_rise_bias_calibrator, fit_rise_bias_calibrator
from codex_v2.src.training.losses_v2 import AsymmetricHuberLoss


def test_asymmetric_huber_applies_directional_weights() -> None:
    pred = torch.tensor([[0.0, -1.0, 1.0, 0.5]], dtype=torch.float32)
    target = torch.tensor([[1.0, -0.5, -1.0, 0.2]], dtype=torch.float32)
    # positions:
    # 0: rise-under (target>0 and pred<target) => weighted
    # 1: drop-miss false
    # 2: drop-miss true (target<0 and pred>=0) => weighted
    # 3: none
    loss_fn = AsymmetricHuberLoss(delta=1.0, rise_under_w=1.0, drop_miss_w=1.0)
    loss = float(loss_fn(pred, target).item())

    err = (pred - target).numpy()
    abs_err = np.abs(err)
    huber = np.where(abs_err <= 1.0, 0.5 * (err ** 2), abs_err - 0.5)
    w = np.ones_like(huber)
    w[0, 0] += 1.0
    w[0, 2] += 1.0
    expected = float(np.mean(w * huber))
    np.testing.assert_allclose(loss, expected, atol=1e-6)


def test_signed_log_asym_transform_behaves_as_expected() -> None:
    y_raw = np.array([[100.0, 200.0], [130.0, 220.0]], dtype=np.float32)
    mean = np.array([110.0, 210.0], dtype=np.float32)
    std = np.array([1.0, 1.0], dtype=np.float32)

    yt = apply_target_transform(
        y_raw=y_raw,
        target_mode="district_signed_log_asym",
        target_mean=mean,
        target_std=std,
        signed_log_pos_gain=1.15,
        signed_log_neg_gain=1.0,
    )

    inv = np.where(
        yt >= 0.0,
        np.expm1(yt / 1.15),
        -np.expm1(np.abs(yt) / 1.0),
    ) + mean[None, :]
    np.testing.assert_allclose(inv.astype(np.float32), y_raw, atol=1e-5)


def test_rise_calibrator_noop_when_no_positive_predictions() -> None:
    pred_raw = np.array([[[100.0, 110.0]]], dtype=np.float32).reshape(1, 2)
    actual_raw = np.array([[[95.0, 105.0]]], dtype=np.float32).reshape(1, 2)
    target_mean = np.array([120.0, 130.0], dtype=np.float32)
    sample_opdates = ["01-05"]
    sample_splits = ["train"]

    calib = fit_rise_bias_calibrator(
        pred_raw=pred_raw,
        actual_raw=actual_raw,
        sample_opdates=sample_opdates,
        sample_splits=sample_splits,
        target_mean=target_mean,
    )
    out = apply_rise_bias_calibrator(
        pred_raw=pred_raw,
        sample_opdates=sample_opdates,
        target_mean=target_mean,
        calibrator=calib,
    )
    np.testing.assert_allclose(out, pred_raw, atol=1e-6)
