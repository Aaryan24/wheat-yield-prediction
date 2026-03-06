from __future__ import annotations

import torch
import torch.nn as nn


class MAEMSELoss(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mae(pred, target) + (1.0 - self.alpha) * self.mse(pred, target)


class AsymmetricHuberLoss(nn.Module):
    def __init__(
        self,
        delta: float = 1.0,
        rise_under_w: float = 0.8,
        drop_miss_w: float = 0.2,
    ) -> None:
        super().__init__()
        self.delta = float(delta)
        self.rise_under_w = float(rise_under_w)
        self.drop_miss_w = float(drop_miss_w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * (err ** 2),
            self.delta * (abs_err - 0.5 * self.delta),
        )

        w = torch.ones_like(huber)
        rise_under_mask = (target > 0.0) & (pred < target)
        drop_miss_mask = (target < 0.0) & (pred >= 0.0)
        if self.rise_under_w != 0.0:
            w = w + self.rise_under_w * rise_under_mask.to(w.dtype)
        if self.drop_miss_w != 0.0:
            w = w + self.drop_miss_w * drop_miss_mask.to(w.dtype)
        return torch.mean(w * huber)


def build_loss(
    loss_name: str,
    *,
    huber_delta: float = 1.0,
    rise_under_w: float = 0.8,
    drop_miss_w: float = 0.2,
) -> nn.Module:
    key = str(loss_name).strip().lower()
    if key == "mse":
        return nn.MSELoss()
    if key == "huber":
        return nn.HuberLoss(delta=float(huber_delta))
    if key == "mae_mse":
        return MAEMSELoss(alpha=0.5)
    if key == "asym_huber":
        return AsymmetricHuberLoss(
            delta=float(huber_delta),
            rise_under_w=float(rise_under_w),
            drop_miss_w=float(drop_miss_w),
        )
    raise ValueError(f"Unsupported loss: {loss_name}")
