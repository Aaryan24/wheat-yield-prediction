#!/usr/bin/env python3
"""Spatial Transformer over sub-district tiles -- the two pieces joined.

Earlier V16 work built two things that were never connected:

  * a rewritten Transformer, which was fed the same 35 district-average MODIS
    numbers V15 used, and duly failed to beat tabular features;
  * ~44 tiles per district, which were collapsed into fixed summary statistics
    (mean, sd, percentiles) and handed to XGBoost, where they worked.

This script gives the Transformer the tiles themselves.  That is the experiment
implied by the finding that resolution, not architecture, was the constraint --
and it is the only configuration in which MMST-ViT's Spatial Transformer
(Eq. 3-4) is operating on data rich enough to justify it.

Why attention over tiles can beat fixed summaries: a percentile treats every
tile as interchangeable.  Attention can weight tiles by how anomalous they are
relative to their own history, so "20% of the area collapsed while the rest
held up" produces a different representation from "everything sagged a little",
even when both share a mean and a spread.

Tiles are treated as an unordered SET.  `tile_index` comes from Earth Engine's
covering grid and carries no meaningful ordering, so no positional embedding is
used and the encoder is permutation invariant by construction -- a deliberate
departure from MMST-ViT, which has true image-grid positions.

Self-supervised, as before: contrastive pretraining with same-season rows
excluded from the negatives, then prediction of the district's crop-state
anomaly.  No yield label is used to fit the encoder.
"""
from __future__ import annotations

import glob
import json
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
sys.path.insert(0, str(V16 / "scripts"))
DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"

DEVICE = torch.device(os.environ.get(
    "V16_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
HIDDEN = 32
HEADS = 4
MAX_TILES = 96
CROSSFIT_GROUPS = 3
PRETRAIN_EPOCHS = int(os.environ.get("V16_SP_PRETRAIN_EPOCHS", "30"))
FINETUNE_EPOCHS = int(os.environ.get("V16_SP_FINETUNE_EPOCHS", "30"))
TEMPERATURE = 0.2
SEED = 42
EMBED_DIMS = 16

INDICES = ["ndvi_season_mean", "ndvi_recent_mean", "ndvi_season_max",
           "ndvi_season_sd", "evi_season_mean", "ndwi_recent_mean"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_group(district_id: str) -> int:
    return sum(ord(c) for c in district_id) % CROSSFIT_GROUPS


# ---------------------------------------------------------------------------
# tile tensor
# ---------------------------------------------------------------------------
def build_tile_tensor() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(district-seasons, MAX_TILES, features) plus an availability mask.

    Each tile carries its raw indices AND its own anomaly against its own
    history.  The anomaly is what makes a tile comparable to other tiles:
    absolute greenness differs between locations for reasons unrelated to
    yield, so a raw-only tensor would mostly encode geography.
    """
    files = sorted(glob.glob(str(DATA / "tiles" / "*_mar05.parquet")))
    tiles = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tiles = tiles.sort_values(["district_id", "tile_index", "season_start_year"])

    grouped = tiles.groupby(["district_id", "tile_index"])
    for index in INDICES:
        prior = grouped[index].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean())
        prior_sd = grouped[index].transform(
            lambda s: s.shift(1).expanding(min_periods=4).std())
        tiles[f"z_{index}"] = (tiles[index] - prior) / prior_sd.replace(0, np.nan)

    columns = INDICES + [f"z_{i}" for i in INDICES]
    keys = ["district_id", "season_start_year"]
    meta = (tiles[keys].drop_duplicates().sort_values(keys).reset_index(drop=True))
    lookup = {(d, y): i for i, (d, y) in enumerate(
        zip(meta.district_id, meta.season_start_year))}

    tensor = np.full((len(meta), MAX_TILES, len(columns)), np.nan, dtype=np.float32)
    mask = np.zeros((len(meta), MAX_TILES), dtype=bool)
    for (district, year), block in tiles.groupby(keys, sort=False):
        row = lookup[(district, year)]
        values = block[columns].to_numpy(np.float32)[:MAX_TILES]
        tensor[row, :len(values)] = values
        mask[row, :len(values)] = True
    return tensor, mask, meta


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class SetAttentionLayer(nn.Module):
    """Masked self-attention across tiles; no positional embedding."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0.08,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.dropout = nn.Dropout(0.08)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attended, _ = self.attention(h, h, h, key_padding_mask=pad,
                                     need_weights=False)
        x = x + self.dropout(attended)
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class SpatialTileEncoder(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(features, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.layers = nn.ModuleList([SetAttentionLayer(HIDDEN, HEADS)
                                     for _ in range(2)])
        self.norm = nn.LayerNorm(HIDDEN)
        # attention pooling: the model chooses which tiles matter, instead of a
        # percentile deciding in advance that every tile is interchangeable
        self.pool_score = nn.Linear(HIDDEN, 1)
        self.projection = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.anomaly_head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN * 2), nn.GELU(),
            nn.Linear(HIDDEN * 2, len(INDICES)))

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        pad = ~valid
        h = self.adapter(x)
        for layer in self.layers:
            h = layer(h, pad)
        h = self.norm(h)
        score = self.pool_score(h).squeeze(-1).masked_fill(pad, float("-inf"))
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (h * weight).sum(dim=1)
        # a plain masked mean alongside the learned pool, so the representation
        # never loses the simple summary that already worked
        count = valid.sum(dim=1, keepdim=True).clamp(min=1)
        average = (h * valid.unsqueeze(-1)).sum(dim=1) / count
        combined = torch.cat([pooled, average], dim=1)
        return combined, self.projection(pooled), self.anomaly_head(pooled)


def augment(x: torch.Tensor, valid: torch.Tensor, generator: torch.Generator):
    """Drop tiles and jitter features -- nuisances the summary should survive."""
    shape = tuple(x.shape)
    noise = torch.randn(shape, generator=generator).to(x.device) * 0.15
    out = x + noise
    keep_feature = (torch.rand(shape, generator=generator) > 0.10).float().to(x.device)
    out = out * keep_feature
    drop = (torch.rand(tuple(valid.shape), generator=generator) > 0.25).to(x.device)
    kept = valid & drop
    # never drop every tile
    empty = ~kept.any(dim=1)
    kept[empty] = valid[empty]
    return out, kept


def info_nce(a: torch.Tensor, b: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
    a = nn.functional.normalize(a, dim=1)
    b = nn.functional.normalize(b, dim=1)
    logits = (a @ b.t()) / TEMPERATURE
    positive = torch.arange(len(a), device=a.device)
    same_year = years[:, None].eq(years[None, :])
    block = same_year & ~torch.eye(len(a), dtype=torch.bool, device=a.device)
    return nn.functional.cross_entropy(logits.masked_fill(block, float("-inf")),
                                       positive)


def train_encoder(x: np.ndarray, valid: np.ndarray, years: np.ndarray,
                  target: np.ndarray, seed: int) -> SpatialTileEncoder:
    set_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = SpatialTileEncoder(x.shape[2]).to(DEVICE)

    xt = torch.tensor(x, dtype=torch.float32)
    vt = torch.tensor(valid, dtype=torch.bool)
    yt = torch.tensor(years, dtype=torch.long)
    tt = torch.tensor(np.nan_to_num(target), dtype=torch.float32)
    ft = torch.tensor(np.isfinite(target), dtype=torch.float32)

    loader = DataLoader(TensorDataset(xt, vt, yt), batch_size=64, shuffle=True,
                        drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=7e-4)
    model.train()
    loss = torch.tensor(float("nan"))
    for _ in range(PRETRAIN_EPOCHS):
        for xb, vb, yb in loader:
            xb, vb, yb = xb.to(DEVICE), vb.to(DEVICE), yb.to(DEVICE)
            _, pa, _ = model(*augment(xb, vb, generator))
            _, pb, _ = model(*augment(xb, vb, generator))
            loss = info_nce(pa, pb, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    model.pretrain_loss = float(loss.detach().cpu())

    loader = DataLoader(TensorDataset(xt, vt, tt, ft), batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=9e-4)
    for _ in range(FINETUNE_EPOCHS):
        for xb, vb, tb, fb in loader:
            xb, vb, tb, fb = (xb.to(DEVICE), vb.to(DEVICE),
                              tb.to(DEVICE), fb.to(DEVICE))
            _, _, prediction = model(xb, vb)
            error = nn.functional.smooth_l1_loss(
                prediction * fb, tb * fb, beta=0.5, reduction="sum")
            loss = error / fb.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    model.eval()
    model.finetune_loss = float(loss.detach().cpu())
    return model


@torch.no_grad()
def representation(model: SpatialTileEncoder, x: np.ndarray,
                   valid: np.ndarray) -> np.ndarray:
    combined, _, anomaly = model(
        torch.tensor(x, dtype=torch.float32).to(DEVICE),
        torch.tensor(valid, dtype=torch.bool).to(DEVICE))
    combined = combined.cpu().numpy()
    anomaly = anomaly.cpu().numpy()
    return np.concatenate([combined[:, :EMBED_DIMS],
                           combined[:, HIDDEN:HIDDEN + EMBED_DIMS],
                           anomaly], axis=1)


def main() -> None:
    tensor, mask, meta = build_tile_tensor()
    print(f"tile tensor {tensor.shape}, "
          f"tiles per district-season: median {mask.sum(1).mean():.0f}")

    years = meta.season_start_year.to_numpy(int)
    groups = np.array([stable_group(d) for d in meta.district_id])
    columns = [f"sp__{i:02d}" for i in range(2 * EMBED_DIMS + len(INDICES))]

    rows, audit = [], []
    for test_year in range(2004, 2023):
        train = years < test_year
        if train.sum() < 300:
            continue

        centre = np.nanmean(tensor[train], axis=(0, 1))
        spread = np.nanstd(tensor[train], axis=(0, 1))
        spread = np.where(spread > 1e-8, spread, 1.0)
        x_all = np.nan_to_num((tensor - centre) / spread).astype(np.float32)

        # district-level crop anomaly for this season, from tile means
        district_mean = np.nansum(
            tensor[:, :, :len(INDICES)] * mask[:, :, None], axis=1
        ) / np.maximum(mask.sum(1, keepdims=True), 1)
        target_centre = np.nanmean(district_mean[train], axis=0)
        target_spread = np.nanstd(district_mean[train], axis=0)
        target_spread = np.where(target_spread > 1e-8, target_spread, 1.0)
        target = ((district_mean - target_centre) / target_spread).astype(np.float32)

        for group in range(CROSSFIT_GROUPS):
            fit = train & (groups != group)
            model = train_encoder(x_all[fit], mask[fit], years[fit],
                                  target[fit], SEED + group)
            pick = train & (groups == group)
            block = meta.loc[pick, ["district_id", "season_start_year"]].copy()
            block[columns] = representation(model, x_all[pick], mask[pick])
            block["representation_train_end"] = test_year - 1
            rows.append(block)

        model = train_encoder(x_all[train], mask[train], years[train],
                              target[train], SEED)
        pick = years == test_year
        block = meta.loc[pick, ["district_id", "season_start_year"]].copy()
        block[columns] = representation(model, x_all[pick], mask[pick])
        block["representation_train_end"] = test_year - 1
        rows.append(block)

        audit.append({"test_year": test_year, "train_rows": int(train.sum()),
                      "pretrain_loss": model.pretrain_loss,
                      "finetune_loss": model.finetune_loss,
                      "parameters": sum(p.numel() for p in model.parameters())})
        print(f"  fold {test_year}: {int(train.sum())} train rows, "
              f"contrastive {model.pretrain_loss:.3f}, "
              f"anomaly {model.finetune_loss:.3f}", flush=True)

    features = pd.concat(rows, ignore_index=True)
    features.to_parquet(DATA / "v16_spatial_features.parquet", index=False)
    pd.DataFrame(audit).to_csv(ARTIFACTS / "spatial_encoder_audit.csv", index=False)
    print(f"\nwrote {len(features)} rows, "
          f"{sum(p.numel() for p in model.parameters())} parameters, {DEVICE}")


if __name__ == "__main__":
    main()
