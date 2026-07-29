#!/usr/bin/env python3
"""V16 crop encoder: three changes to V15's, each forced by a measurement.

1.  ANOMALY TARGET.  V15's encoder predicted the raw crop-state change.  That
    target is two-thirds explained by climatology -- wheat simply grows between
    January and March -- so most of the network's capacity reproduced a seasonal
    mean that a lookup table already knows.  Here the target is the change MINUS
    the climatological change for that transition, so every unit of capacity is
    spent on the part that varies between seasons.

2.  CONTRASTIVE PRETRAINING (MMST-ViT Eq. 1-2) instead of next-token
    regression.  V15's MODIS pretraining did not improve even its own pretext
    task (0.04163 pretrained vs 0.04179 from scratch).  A contrastive objective
    asks a different and more useful question: what makes THIS district-season
    distinguishable from others?  That is exactly the anomaly signal yield
    depends on, and it needs no yield labels.

    One deliberate departure from SimCLR: rows from the SAME season are removed
    from the negatives.  Districts in one season share a large common shock
    (sd ~354 kg/ha, larger than all district-level variation), and standard
    in-batch negatives would train the encoder to discard precisely that.

3.  LONG-TERM CLIMATE BIAS (MMST-ViT Eq. 6).  A projection of the district's
    multi-year context is added to the attention logits, so slow-moving
    district character reweights which clock tokens matter rather than being
    concatenated as one more feature.

Trained on the MODIS panel (2000-2022, 2,737 district-seasons) rather than
Sentinel (2017-2022), because the whole point of V16 is that choices must be
made where there are enough independent seasons to resolve them.

Strictness: for target season t the encoder sees only seasons < t, and features
for training rows are produced by district-cross-fitted encoders that never saw
that district group.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

V16 = Path(__file__).resolve().parents[1]
RAPID = V16.parent
sys.path.insert(0, str(V16 / "scripts"))

V15_DATA = RAPID / "v15_complete_hierarchy" / "data"
DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"
MODELS = V16 / "models"

DEVICE = torch.device(os.environ.get(
    "V16_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
HIDDEN = 32
HEADS = 4
CROSSFIT_GROUPS = 3
PRETRAIN_EPOCHS = int(os.environ.get("V16_PRETRAIN_EPOCHS", "40"))
FINETUNE_EPOCHS = int(os.environ.get("V16_FINETUNE_EPOCHS", "40"))
TEMPERATURE = 0.2
SEED = 42

CLIMATE_COLUMNS = [
    "roll_5_mean", "roll_10_mean", "roll_10_std", "roll_20_mean",
    "roll_5_slope", "roll_10_slope", "lag_1_over_roll_10",
    "roll_3_over_roll_10", "state_lag_1_mean_yield",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_group(district_id: str) -> int:
    return sum(ord(c) for c in district_id) % CROSSFIT_GROUPS


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class ClimateBiasedEncoderLayer(nn.Module):
    """Pre-norm Transformer layer whose attention logits carry a climate bias.

    The bias is a linear function of the district's multi-year context and is
    added per head and per key position, so long-run district character changes
    WHICH clock the model attends to, rather than merely shifting its features.
    """

    def __init__(self, hidden: int, heads: int, climate_dim: int, tokens: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.project = nn.Linear(hidden, hidden)
        self.climate_bias = nn.Linear(climate_dim, heads * tokens)
        self.tokens = tokens
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.dropout = nn.Dropout(0.08)

    def forward(self, x: torch.Tensor, climate: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.climate_bias(climate).reshape(batch, self.heads, 1, self.tokens)
        attention = torch.softmax(logits + bias[..., :tokens], dim=-1)
        context = (self.dropout(attention) @ v).transpose(1, 2).reshape(
            batch, tokens, hidden)
        x = x + self.dropout(self.project(context))
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class CropEncoder(nn.Module):
    def __init__(self, features: int, climate_dim: int, tokens: int = 3):
        super().__init__()
        self.adapter = nn.Linear(features, HIDDEN)
        self.position = nn.Parameter(torch.zeros(1, tokens, HIDDEN))
        self.layers = nn.ModuleList([
            ClimateBiasedEncoderLayer(HIDDEN, HEADS, climate_dim, tokens)
            for _ in range(2)])
        self.norm = nn.LayerNorm(HIDDEN)
        # projection head used only by the contrastive loss, then discarded
        self.projection = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        # predicts the change beyond climatology for the next clock
        self.anomaly_head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN * 2), nn.GELU(), nn.Linear(HIDDEN * 2, features))

    def encode(self, x: torch.Tensor, climate: torch.Tensor) -> torch.Tensor:
        h = self.adapter(x) + self.position[:, :x.shape[1]]
        for layer in self.layers:
            h = layer(h, climate)
        return self.norm(h)

    def forward(self, x: torch.Tensor, climate: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x, climate)
        pooled = h.mean(dim=1)
        return pooled, self.projection(pooled), self.anomaly_head(h[:, -1])


# ---------------------------------------------------------------------------
# augmentation and losses
# ---------------------------------------------------------------------------
def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Nuisance transformations the representation should be invariant to."""
    # draw on CPU so one seeded generator gives identical augmentations on any
    # device (MPS generators are not interchangeable with CPU ones)
    shape = tuple(x.shape)
    noise = torch.randn(shape, generator=generator).to(x.device) * 0.15
    out = x + noise
    keep_feature = (torch.rand(shape, generator=generator) > 0.15).float().to(x.device)
    out = out * keep_feature
    keep_token = (torch.rand(shape[:2] + (1,), generator=generator)
                  > 0.20).float().to(x.device)
    # never drop every clock
    keep_token[:, -1] = 1.0
    return out * keep_token


def info_nce(a: torch.Tensor, b: torch.Tensor, years: torch.Tensor,
             temperature: float = TEMPERATURE) -> torch.Tensor:
    """Contrastive loss with same-season rows excluded from the negatives.

    Two districts observed in the same season share a large common shock.
    Treating them as negatives would teach the encoder to throw that shock
    away, which is the opposite of what the yield model needs.
    """
    a = nn.functional.normalize(a, dim=1)
    b = nn.functional.normalize(b, dim=1)
    logits = (a @ b.t()) / temperature
    positive = torch.arange(len(a), device=a.device)
    same_year = years[:, None].eq(years[None, :])
    block = same_year & ~torch.eye(len(a), dtype=torch.bool, device=a.device)
    logits = logits.masked_fill(block, float("-inf"))
    return nn.functional.cross_entropy(logits, positive)


def train_encoder(sequence: np.ndarray, climate: np.ndarray, years: np.ndarray,
                  anomaly_target: np.ndarray, seed: int) -> CropEncoder:
    set_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = CropEncoder(sequence.shape[2], climate.shape[1]).to(DEVICE)

    x = torch.tensor(sequence, dtype=torch.float32)
    c = torch.tensor(climate, dtype=torch.float32)
    y = torch.tensor(years, dtype=torch.long)
    t = torch.tensor(anomaly_target, dtype=torch.float32)

    # --- stage 1: contrastive pretraining, no labels of any kind ---
    loader = DataLoader(TensorDataset(x, c, y), batch_size=128, shuffle=True,
                        drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=7e-4)
    model.train()
    for _ in range(PRETRAIN_EPOCHS):
        for xb, cb, yb in loader:
            xb, cb, yb = xb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
            _, pa, _ = model(augment(xb, generator), cb)
            _, pb, _ = model(augment(xb, generator), cb)
            loss = info_nce(pa, pb, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    pretrain_loss = float(loss.detach().cpu())

    # --- stage 2: predict the change BEYOND climatology ---
    finite = torch.tensor(np.isfinite(anomaly_target), dtype=torch.float32)
    t = torch.nan_to_num(t)
    loader = DataLoader(TensorDataset(x, c, t, finite), batch_size=128, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=9e-4)
    for _ in range(FINETUNE_EPOCHS):
        for xb, cb, tb, mb in loader:
            xb, cb, tb, mb = (xb.to(DEVICE), cb.to(DEVICE),
                              tb.to(DEVICE), mb.to(DEVICE))
            _, _, prediction = model(xb, cb)
            error = nn.functional.smooth_l1_loss(
                prediction * mb, tb * mb, beta=0.5, reduction="sum")
            loss = error / mb.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    model.eval()
    model.pretrain_loss = pretrain_loss
    model.finetune_loss = float(loss.detach().cpu())
    return model


@torch.no_grad()
def representation(model: CropEncoder, sequence: np.ndarray,
                   climate: np.ndarray) -> np.ndarray:
    x = torch.tensor(sequence, dtype=torch.float32).to(DEVICE)
    c = torch.tensor(climate, dtype=torch.float32).to(DEVICE)
    pooled, _, anomaly = model(x, c)
    pooled = pooled.cpu().numpy()
    anomaly = anomaly.cpu().numpy()
    return np.concatenate([
        pooled[:, :16],
        anomaly.mean(axis=1, keepdims=True),
        np.abs(anomaly).mean(axis=1, keepdims=True),
        anomaly.std(axis=1, keepdims=True),
    ], axis=1)


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(DATA / "v16_panel.parquet")
    manifest = json.loads((V15_DATA / "data_manifest.json").read_text())
    names = manifest["modis_features"]
    npz = np.load(V15_DATA / "modis_sequences_2000_2022.npz", allow_pickle=True)
    meta = pd.read_parquet(V15_DATA / "modis_metadata.parquet")

    sequence = npz["sequence"].astype(np.float64).copy()
    sequence[~npz["mask"]] = np.nan
    meta = meta.reset_index(drop=True)
    meta["group"] = [stable_group(d) for d in meta.district_id]

    climate = (meta[["district_id", "season_start_year"]]
               .merge(panel[["district_id", "season_start_year"] + CLIMATE_COLUMNS],
                      on=["district_id", "season_start_year"], how="left"))
    climate_raw = climate[CLIMATE_COLUMNS].to_numpy(float)

    years = meta.season_start_year.to_numpy(int)
    test_years = list(range(2004, 2023))

    rows, audit = [], []
    for test_year in test_years:
        train_mask = years < test_year
        if train_mask.sum() < 300:
            continue

        # normalizers fitted inside the fold only
        mu = np.nanmean(sequence[train_mask], axis=(0, 1))
        sd = np.nanstd(sequence[train_mask], axis=(0, 1))
        sd = np.where(sd > 1e-8, sd, 1.0)
        x_all = np.nan_to_num((sequence - mu) / sd).astype(np.float32)

        cmu = np.nanmean(climate_raw[train_mask], axis=0)
        csd = np.nanstd(climate_raw[train_mask], axis=0)
        csd = np.where(csd > 1e-8, csd, 1.0)
        c_all = np.nan_to_num((climate_raw - cmu) / csd).astype(np.float32)

        # climatological Feb->Mar change, from training seasons only
        change = sequence[:, 2, :] - sequence[:, 1, :]
        climatology = np.nanmean(change[train_mask], axis=0)
        spread = np.nanstd(change[train_mask], axis=0)
        spread = np.where(spread > 1e-8, spread, 1.0)
        anomaly_target = ((change - climatology) / spread).astype(np.float32)

        # encoders that produced a row's features never saw that row's group
        for group in range(CROSSFIT_GROUPS):
            fit = train_mask & (meta.group.to_numpy() != group)
            model = train_encoder(x_all[fit], c_all[fit], years[fit],
                                  anomaly_target[fit], SEED + group)
            target = train_mask & (meta.group.to_numpy() == group)
            block = meta.loc[target, ["district_id", "season_start_year"]].copy()
            block[[f"enc__{i:02d}" for i in range(19)]] = representation(
                model, x_all[target], c_all[target])
            block["representation_train_end"] = test_year - 1
            block["feature_role"] = "train_crossfit"
            rows.append(block)

        model = train_encoder(x_all[train_mask], c_all[train_mask],
                              years[train_mask], anomaly_target[train_mask], SEED)
        target = years == test_year
        block = meta.loc[target, ["district_id", "season_start_year"]].copy()
        block[[f"enc__{i:02d}" for i in range(19)]] = representation(
            model, x_all[target], c_all[target])
        block["representation_train_end"] = test_year - 1
        block["feature_role"] = "test_full"
        rows.append(block)

        audit.append({"test_year": test_year, "train_rows": int(train_mask.sum()),
                      "pretrain_loss": model.pretrain_loss,
                      "finetune_loss": model.finetune_loss,
                      "parameters": sum(p.numel() for p in model.parameters())})
        print(f"  fold {test_year}: {int(train_mask.sum())} train rows, "
              f"contrastive {model.pretrain_loss:.3f}, "
              f"anomaly {model.finetune_loss:.3f}", flush=True)

    features = pd.concat(rows, ignore_index=True)
    features.to_parquet(DATA / "v16_encoder_features.parquet", index=False)
    pd.DataFrame(audit).to_csv(ARTIFACTS / "encoder_training_audit.csv", index=False)
    print(f"\nwrote {len(features)} encoder rows, "
          f"{sum(p.numel() for p in model.parameters())} parameters, device {DEVICE}")


if __name__ == "__main__":
    main()
