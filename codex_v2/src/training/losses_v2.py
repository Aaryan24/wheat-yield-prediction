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


def build_loss(loss_name: str) -> nn.Module:
    key = str(loss_name).strip().lower()
    if key == "mse":
        return nn.MSELoss()
    if key == "huber":
        return nn.HuberLoss(delta=1.0)
    if key == "mae_mse":
        return MAEMSELoss(alpha=0.5)
    raise ValueError(f"Unsupported loss: {loss_name}")
