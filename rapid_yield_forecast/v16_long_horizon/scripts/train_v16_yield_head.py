#!/usr/bin/env python3
"""Fine-tune the pretrained encoder on yield, emitting a DISTRIBUTION.

Every previous attempt used the encoder as a feature generator: pretrain
self-supervised, freeze, hand 40 numbers to XGBoost.  That failed inside V15
because the representation encodes what makes a district-season *distinctive*,
which is not the same as what makes yield *go wrong* (correlation with V15's
residual error: 0.024).

This trains the network on the actual target instead.  Two reasons it is worth
trying where the frozen-feature route failed:

  * the expensive part -- learning crop structure from tiles, clocks and
    weather without labels -- is already validated, so fine-tuning starts from
    a representation that is not random;
  * the head predicts 19 QUANTILES under a pinball loss, not a single number.
    A distribution is an easier target than a point when labels are scarce:
    the model is allowed to say "somewhere in this range", and the loss rewards
    honest width instead of punishing every deviation quadratically.  It is
    also what the forecast is actually for.

V15 explicitly rejected direct Transformer-to-yield training because it had
"too few independent yield years" -- with four.  Here there are nineteen
rolling-origin folds and up to ~2,500 labelled district-seasons per fold, which
is what makes the attempt defensible now and was not then.

MONOTONICITY: the head emits a centre plus positive increments passed through
softplus and cumulatively summed outward, so quantiles cannot cross by
construction rather than by post-hoc sorting.

Target is the log anomaly around the three-season weighted baseline, so the
predicted interval is multiplicative and transfers across district yield levels.

Strict: for target season t only seasons < t are used, for pretraining and
fine-tuning alike, and all normalizers are fitted inside the fold.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

V16 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V16 / "scripts"))
import train_v16_unified_encoder as base  # noqa: E402
from v16_common import BASELINE, TARGET  # noqa: E402

DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"
DEVICE = base.DEVICE
QUANTILES = [round(0.05 * i, 2) for i in range(1, 20)]
CENTRE_INDEX = QUANTILES.index(0.50)
PRETRAIN_EPOCHS = int(os.environ.get("V16_Y_PRETRAIN_EPOCHS", "30"))
FINETUNE_EPOCHS = int(os.environ.get("V16_Y_FINETUNE_EPOCHS", "40"))
SEEDS = (42, 73)
CLIP_LOG = 0.60


class QuantileHead(nn.Module):
    """Centre plus outward increments -- quantiles cannot cross."""

    def __init__(self, hidden: int, levels: int):
        super().__init__()
        self.centre_index = CENTRE_INDEX
        self.levels = levels
        self.body = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(0.10))
        self.centre = nn.Linear(hidden, 1)
        self.upper = nn.Linear(hidden, levels - 1 - CENTRE_INDEX)
        self.lower = nn.Linear(hidden, CENTRE_INDEX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        centre = self.centre(h)
        up = torch.cumsum(nn.functional.softplus(self.upper(h)) + 1e-4, dim=1)
        down = torch.cumsum(nn.functional.softplus(self.lower(h)) + 1e-4, dim=1)
        return torch.cat([centre - down.flip(1), centre, centre + up], dim=1)


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor,
                 levels: torch.Tensor) -> torch.Tensor:
    difference = target.unsqueeze(1) - prediction
    return torch.mean(torch.maximum(levels * difference,
                                    (levels - 1.0) * difference))


def train_fold(fold: dict, train_index: np.ndarray, years: np.ndarray,
               target: np.ndarray, seed: int, anomaly_dim: int):
    """Self-supervised pretraining, then supervised distributional fine-tuning."""
    encoder = base.train_encoder(fold, train_index, years, seed, anomaly_dim)
    head = QuantileHead(base.HIDDEN * 2, len(QUANTILES)).to(DEVICE)

    tensors = [torch.tensor(fold[k][train_index]) for k in
               ("tiles", "tile_mask", "clocks_n", "weather_n", "weather_mask",
                "climate_n")]
    tensors.append(torch.tensor(target, dtype=torch.float32))
    loader = DataLoader(TensorDataset(*tensors), batch_size=64, shuffle=True)
    levels = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)

    # the encoder is fine-tuned at a tenth of the head's learning rate: enough
    # to adapt the representation, slow enough not to destroy what pretraining
    # found on far more examples than there are yield labels
    optimizer = torch.optim.AdamW(
        [{"params": encoder.parameters(), "lr": 6e-5},
         {"params": head.parameters(), "lr": 6e-4}], weight_decay=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS)
    encoder.train()
    head.train()
    loss = torch.tensor(float("nan"))
    for _ in range(FINETUNE_EPOCHS):
        for ti, tv, cl, wx, wv, cm, yb in loader:
            ti, tv, cl, wx, wv, cm, yb = (t.to(DEVICE) for t in
                                          (ti, tv, cl, wx, wv, cm, yb))
            combined, _, _ = encoder(ti, tv, cl, wx, wv, cm)
            loss = pinball_loss(head(combined), yb, levels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), 2.0)
            optimizer.step()
        schedule.step()
    encoder.eval()
    head.eval()
    return encoder, head, float(loss.detach().cpu())


@torch.no_grad()
def predict(encoder, head, fold: dict, index: np.ndarray) -> np.ndarray:
    tensors = [torch.tensor(fold[k][index]).to(DEVICE) for k in
               ("tiles", "tile_mask", "clocks_n", "weather_n", "weather_mask",
                "climate_n")]
    combined, _, _ = encoder(*tensors)
    return head(combined).cpu().numpy()


def main() -> None:
    data = base.assemble()
    meta = data["meta"]
    years = meta.season_start_year.to_numpy(int)
    anomaly_dim = data["n_anomaly"]

    panel = pd.read_parquet(DATA / "v16_panel.parquet")[
        ["district_id", "season_start_year", TARGET, BASELINE, "lag_1_yield",
         "state_name"]]
    labelled = meta.merge(panel, on=["district_id", "season_start_year"],
                          how="left")
    anomaly = np.log(np.clip(labelled[TARGET].to_numpy(float)
                             / labelled[BASELINE].to_numpy(float), 0.3, 3.0))
    has_label = np.isfinite(anomaly)
    print(f"{int(has_label.sum())} labelled district-seasons available")

    rows = []
    for test_year in range(2004, 2023):
        train = (years < test_year) & has_label
        test = years == test_year
        if train.sum() < 300 or test.sum() == 0:
            continue
        fold = {"tile_mask": data["tile_mask"], "weather_mask": data["weather_mask"],
                "tiles": base.normalize(data["tiles"], train, (0, 1)),
                "clocks_n": base.normalize(data["clocks"], train, (0, 1)),
                "weather_n": base.normalize(data["weather"], train, (0, 1)),
                "climate_n": base.normalize(data["climate"], train, (0,))}
        target = np.clip(anomaly[train], -CLIP_LOG, CLIP_LOG).astype(np.float32)

        seed_predictions = []
        for seed in SEEDS:
            encoder, head, final_loss = train_fold(
                fold, train, years[train], target, seed, anomaly_dim)
            seed_predictions.append(predict(encoder, head, fold, test))
        quantiles = np.mean(seed_predictions, axis=0)

        block = labelled.loc[test, ["district_id", "state_name",
                                    "season_start_year", TARGET, BASELINE,
                                    "lag_1_yield"]].copy()
        baseline_values = block[BASELINE].to_numpy(float)[:, None]
        yields = np.clip(baseline_values * np.exp(
            np.clip(quantiles, -CLIP_LOG, CLIP_LOG)), 500, 7000)
        for i, level in enumerate(QUANTILES):
            block[f"q{int(round(level * 100)):02d}"] = yields[:, i]
        block["point"] = yields[:, CENTRE_INDEX]
        rows.append(block)
        print(f"  fold {test_year}: {int(train.sum())} labelled train rows, "
              f"pinball {final_loss:.4f}", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(ARTIFACTS / "yield_head_predictions.parquet", index=False)
    print(f"\nwrote {len(out)} rows to yield_head_predictions.parquet")


if __name__ == "__main__":
    main()
