#!/usr/bin/env python3
"""One encoder over every source of data available.

Everything built before this saw a thin slice, and each failed or barely helped
for the same reason:

  V15 crop Transformer      Sentinel district means, 2017-2022   +0.93 (noise)
  V16 Transformer rewrite   MODIS district means                 -2.19 over 19 folds
  V16 tile features         tiles, collapsed to percentiles      +3.4 to +7.4

This encoder takes all of it at once -- MMST-ViT's Multi-Modal / Spatial /
Temporal decomposition adapted to the data that actually exists here.

DESIGN DECISIONS, each forced by a measurement:

1.  PRETEXT TASK IS MASKED TILE MODELLING.  An earlier version asked the model
    to predict the district-mean crop anomaly from the tiles.  That task is
    degenerate -- the district mean IS the average of the tiles being fed in,
    so the model can solve it by averaging its own input and learn nothing.
    Here 30% of a district's tiles are replaced by a learned [MASK] embedding
    and their own-history anomalies must be recovered from the surviving tiles.
    That forces the encoder to learn how crop damage is spatially organised,
    which is precisely the structure a percentile cannot express.

2.  TILES ARE NOT POOLED BEFORE FUSION.  An earlier version collapsed ~44 tiles
    into one token and only then let weather interact with it, discarding the
    resolution the tiles exist to provide.  Here every tile cross-attends to
    the weather and clock tokens directly (MMST-ViT Eq. 1: query from the
    visual stream, keys and values from meteorology), so each location asks
    what weather explains ITS state.

3.  ANOMALIES LEAD, LEVELS FOLLOW.  Measured over 19 folds: MODIS levels gain
    -1.08 kg/ha, MODIS anomalies gain +17.8.  Absolute greenness mostly encodes
    geography.  Per-tile own-history anomalies are therefore the primary
    channel and the reconstruction target; raw levels are retained only as
    secondary context.

4.  CLIMATE BIAS ON THE CROSS-ATTENTION LOGITS (MMST-ViT Eq. 6), so a
    district's multi-year character decides which weather window it attends
    to, rather than arriving as one more concatenated feature.

Self-supervised throughout: no yield label touches the encoder, and for target
season t only seasons < t are seen.  Features for training rows come from
district-cross-fitted encoders that never saw that district group.
"""
from __future__ import annotations

import glob
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

DEVICE = torch.device(os.environ.get(
    "V16_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
HIDDEN = 48
HEADS = 4
MAX_TILES = 96
CROSSFIT_GROUPS = 3
EPOCHS = int(os.environ.get("V16_UNI_EPOCHS", "45"))
TEMPERATURE = 0.2
MASK_RATE = 0.30
CONTRASTIVE_WEIGHT = 0.3
SEED = 42
EMBED_DIMS = 20

TILE_INDICES = ["ndvi_season_mean", "ndvi_recent_mean", "ndvi_season_max",
                "ndvi_season_sd", "evi_season_mean", "ndwi_recent_mean"]
WEATHER_WINDOWS = ["dec_feb", "jan_feb", "feb_mar05", "full_preclock"]
WEATHER_VARS = ["tmax_mean", "tmax_max", "tmin_mean", "precip_sum", "solar_mean",
                "rh_mean", "wind_mean", "hot30_days", "hot32_days", "gdd_sum"]
CLIMATE_COLUMNS = ["roll_5_mean", "roll_10_mean", "roll_10_std", "roll_20_mean",
                   "roll_5_slope", "roll_10_slope", "lag_1_over_roll_10",
                   "roll_3_over_roll_10", "state_lag_1_mean_yield"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_group(district_id: str) -> int:
    return sum(ord(c) for c in district_id) % CROSSFIT_GROUPS


def assemble() -> dict:
    """All modalities aligned to one district-season index."""
    files = sorted(glob.glob(str(DATA / "tiles" / "*_mar05.parquet")))
    tiles = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tiles = tiles.sort_values(["district_id", "tile_index", "season_start_year"])
    grouped = tiles.groupby(["district_id", "tile_index"])
    for index in TILE_INDICES:
        prior = grouped[index].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean())
        prior_sd = grouped[index].transform(
            lambda s: s.shift(1).expanding(min_periods=4).std())
        tiles[f"z_{index}"] = (tiles[index] - prior) / prior_sd.replace(0, np.nan)

    # anomalies first: they are both the primary channel and the target
    anomaly_columns = [f"z_{i}" for i in TILE_INDICES]
    tile_columns = anomaly_columns + TILE_INDICES

    keys = ["district_id", "season_start_year"]
    meta = tiles[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    lookup = {(d, y): i for i, (d, y) in enumerate(
        zip(meta.district_id, meta.season_start_year))}

    tile_tensor = np.full((len(meta), MAX_TILES, len(tile_columns)), np.nan,
                          dtype=np.float32)
    tile_mask = np.zeros((len(meta), MAX_TILES), dtype=bool)
    for (district, year), block in tiles.groupby(keys, sort=False):
        row = lookup[(district, year)]
        values = block[tile_columns].to_numpy(np.float32)[:MAX_TILES]
        tile_tensor[row, :len(values)] = values
        tile_mask[row, :len(values)] = True

    npz = np.load(V15_DATA / "modis_sequences_2000_2022.npz", allow_pickle=True)
    modis_meta = pd.read_parquet(V15_DATA / "modis_metadata.parquet")
    sequence = npz["sequence"].astype(np.float32).copy()
    sequence[~npz["mask"]] = np.nan
    clocks = np.full((len(meta), 3, sequence.shape[2]), np.nan, dtype=np.float32)
    for i, (district, year) in enumerate(zip(modis_meta.district_id,
                                             modis_meta.season_start_year)):
        row = lookup.get((district, int(year)))
        if row is not None:
            clocks[row] = sequence[i]

    weather = pd.read_parquet(DATA / "v16_weather_district.parquet")
    weather_tensor = np.full((len(meta), len(WEATHER_WINDOWS), len(WEATHER_VARS)),
                             np.nan, dtype=np.float32)
    weather_mask = np.zeros((len(meta), len(WEATHER_WINDOWS)), dtype=bool)
    for _, row in weather.iterrows():
        target = lookup.get((row.district_id, int(row.season_start_year)))
        if target is None:
            continue
        for w, window in enumerate(WEATHER_WINDOWS):
            values = [row.get(f"wx_{window}_{v}", np.nan) for v in WEATHER_VARS]
            weather_tensor[target, w] = values
            weather_mask[target, w] = bool(np.isfinite(values).any())

    panel = pd.read_parquet(DATA / "v16_panel.parquet")
    climate = meta.merge(panel[keys + CLIMATE_COLUMNS], on=keys, how="left")
    return {"meta": meta, "tiles": tile_tensor, "tile_mask": tile_mask,
            "clocks": clocks, "weather": weather_tensor,
            "weather_mask": weather_mask,
            "climate": climate[CLIMATE_COLUMNS].to_numpy(np.float32),
            "n_anomaly": len(anomaly_columns)}


class SpatialCrossBlock(nn.Module):
    """Tiles talk to each other, then each tile queries weather and clocks."""

    def __init__(self, hidden: int, heads: int, climate_dim: int, context: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden, heads, dropout=0.08,
                                                    batch_first=True)
        self.heads, self.head_dim, self.context = heads, hidden // heads, context
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.project = nn.Linear(hidden, hidden)
        self.climate_bias = nn.Linear(climate_dim, heads * context)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.dropout = nn.Dropout(0.08)

    def forward(self, tiles, tile_pad, context, context_pad, climate):
        h = self.norm1(tiles)
        attended, _ = self.self_attention(h, h, h, key_padding_mask=tile_pad,
                                          need_weights=False)
        tiles = tiles + self.dropout(attended)

        batch, count, hidden = tiles.shape
        h = self.norm2(tiles)
        q = self.query(h).reshape(batch, count, self.heads, self.head_dim
                                  ).transpose(1, 2)
        k = self.key(context).reshape(batch, -1, self.heads, self.head_dim
                                      ).transpose(1, 2)
        v = self.value(context).reshape(batch, -1, self.heads, self.head_dim
                                        ).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        bias = self.climate_bias(climate).reshape(batch, self.heads, 1,
                                                  self.context)
        logits = logits + bias[..., :k.shape[2]]
        logits = logits.masked_fill(context_pad[:, None, None, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        fused = (self.dropout(weights) @ v).transpose(1, 2).reshape(
            batch, count, hidden)
        tiles = tiles + self.dropout(self.project(fused))
        return tiles + self.dropout(self.feed_forward(self.norm3(tiles)))


class UnifiedEncoder(nn.Module):
    def __init__(self, tile_features: int, clock_features: int,
                 weather_features: int, climate_dim: int, anomaly_dim: int):
        super().__init__()
        self.anomaly_dim = anomaly_dim
        self.tile_adapter = nn.Sequential(
            nn.Linear(tile_features, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.mask_token = nn.Parameter(torch.zeros(HIDDEN))
        self.clock_adapter = nn.Linear(clock_features, HIDDEN)
        self.weather_adapter = nn.Linear(weather_features, HIDDEN)
        self.clock_position = nn.Parameter(torch.zeros(1, 3, HIDDEN))
        self.weather_position = nn.Parameter(torch.zeros(1, 4, HIDDEN))
        self.modality = nn.Parameter(torch.zeros(2, HIDDEN))

        context = 3 + 4
        self.blocks = nn.ModuleList([
            SpatialCrossBlock(HIDDEN, HEADS, climate_dim, context)
            for _ in range(2)])
        self.norm = nn.LayerNorm(HIDDEN)
        self.pool_score = nn.Linear(HIDDEN, 1)
        self.projection = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        # reconstructs the own-history anomaly of masked tiles
        self.reconstruct = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, anomaly_dim))

    def forward(self, tiles, tile_valid, clocks, weather, weather_valid,
                climate, masked=None):
        h = self.tile_adapter(tiles)
        if masked is not None:
            h = torch.where(masked.unsqueeze(-1), self.mask_token.expand_as(h), h)
        context = torch.cat([
            self.clock_adapter(clocks) + self.clock_position + self.modality[0],
            self.weather_adapter(weather) + self.weather_position + self.modality[1],
        ], dim=1)
        context_valid = torch.cat([
            torch.ones(len(tiles), 3, dtype=torch.bool, device=tiles.device),
            weather_valid], dim=1)
        tile_pad = ~tile_valid
        for block in self.blocks:
            h = block(h, tile_pad, context, ~context_valid, climate)
        h = self.norm(h)

        score = self.pool_score(h).squeeze(-1).masked_fill(tile_pad, float("-inf"))
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (h * weight).sum(dim=1)
        count = tile_valid.sum(1, keepdim=True).clamp(min=1)
        average = (h * tile_valid.unsqueeze(-1)).sum(1) / count
        return (torch.cat([pooled, average], dim=1), self.projection(pooled),
                self.reconstruct(h))


def info_nce(a, b, years):
    a, b = nn.functional.normalize(a, dim=1), nn.functional.normalize(b, dim=1)
    logits = (a @ b.t()) / TEMPERATURE
    same = years[:, None].eq(years[None, :])
    block = same & ~torch.eye(len(a), dtype=torch.bool, device=a.device)
    return nn.functional.cross_entropy(
        logits.masked_fill(block, float("-inf")),
        torch.arange(len(a), device=a.device))


def train_encoder(fold, index, years, seed, anomaly_dim):
    set_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = [torch.tensor(fold[k][index]) for k in
               ("tiles", "tile_mask", "clocks_n", "weather_n", "weather_mask",
                "climate_n")]
    tensors.append(torch.tensor(years, dtype=torch.long))
    model = UnifiedEncoder(tensors[0].shape[2], tensors[2].shape[2],
                           tensors[3].shape[2], tensors[5].shape[1],
                           anomaly_dim).to(DEVICE)
    loader = DataLoader(TensorDataset(*tensors), batch_size=64, shuffle=True,
                        drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=7e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    model.train()
    loss = torch.tensor(float("nan"))
    recon_value = float("nan")
    for _ in range(EPOCHS):
        for ti, tv, cl, wx, wv, cm, yb in loader:
            ti, tv, cl, wx, wv, cm, yb = (t.to(DEVICE) for t in
                                          (ti, tv, cl, wx, wv, cm, yb))
            # --- masked tile modelling ---
            draw = torch.rand(tuple(tv.shape), generator=generator).to(DEVICE)
            masked = tv & (draw < MASK_RATE)
            empty = masked.all(dim=1)            # never mask every tile
            masked[empty] = False
            _, _, reconstruction = model(ti, tv, cl, wx, wv, cm, masked=masked)
            target = ti[:, :, :anomaly_dim]
            finite = torch.isfinite(target) & masked.unsqueeze(-1)
            reconstruction_loss = nn.functional.smooth_l1_loss(
                torch.where(finite, reconstruction, torch.zeros_like(target)),
                torch.where(finite, target, torch.zeros_like(target)),
                beta=0.5, reduction="sum") / finite.sum().clamp(min=1)

            # --- contrastive, same-season rows excluded from the negatives ---
            noise = torch.randn(tuple(ti.shape), generator=generator).to(DEVICE)
            drop_a = (torch.rand(tuple(tv.shape), generator=generator).to(DEVICE)
                      > 0.25) & tv
            drop_b = (torch.rand(tuple(tv.shape), generator=generator).to(DEVICE)
                      > 0.25) & tv
            drop_a[~drop_a.any(1)] = tv[~drop_a.any(1)]
            drop_b[~drop_b.any(1)] = tv[~drop_b.any(1)]
            _, pa, _ = model(ti + 0.15 * noise, drop_a, cl, wx, wv, cm)
            _, pb, _ = model(ti - 0.15 * noise, drop_b, cl, wx, wv, cm)
            contrastive = info_nce(pa, pb, yb)

            loss = reconstruction_loss + CONTRASTIVE_WEIGHT * contrastive
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            recon_value = float(reconstruction_loss.detach().cpu())
        schedule.step()
    model.eval()
    model.reconstruction_loss = recon_value
    model.total_loss = float(loss.detach().cpu())
    return model


@torch.no_grad()
def representation(model, fold, index):
    tensors = [torch.tensor(fold[k][index]).to(DEVICE) for k in
               ("tiles", "tile_mask", "clocks_n", "weather_n", "weather_mask",
                "climate_n")]
    combined, _, _ = model(*tensors)
    combined = combined.cpu().numpy()
    return np.concatenate([combined[:, :EMBED_DIMS],
                           combined[:, HIDDEN:HIDDEN + EMBED_DIMS]], axis=1)


def normalize(values, train, axes):
    centre = np.nanmean(values[train], axis=axes)
    spread = np.nanstd(values[train], axis=axes)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    spread = np.where(np.isfinite(spread) & (spread > 1e-8), spread, 1.0)
    return np.nan_to_num((values - centre) / spread).astype(np.float32)


def main() -> None:
    data = assemble()
    meta = data["meta"]
    years = meta.season_start_year.to_numpy(int)
    groups = np.array([stable_group(d) for d in meta.district_id])
    anomaly_dim = data["n_anomaly"]
    print(f"assembled {len(meta)} district-seasons | tiles {data['tiles'].shape} "
          f"| clocks {data['clocks'].shape} | weather {data['weather'].shape}")

    columns = [f"uni__{i:02d}" for i in range(2 * EMBED_DIMS)]
    rows, audit = [], []
    for test_year in range(2004, 2023):
        train = years < test_year
        if train.sum() < 300:
            continue
        fold = {"tile_mask": data["tile_mask"], "weather_mask": data["weather_mask"],
                "tiles": normalize(data["tiles"], train, (0, 1)),
                "clocks_n": normalize(data["clocks"], train, (0, 1)),
                "weather_n": normalize(data["weather"], train, (0, 1)),
                "climate_n": normalize(data["climate"], train, (0,))}

        for group in range(CROSSFIT_GROUPS):
            fit = train & (groups != group)
            model = train_encoder(fold, fit, years[fit], SEED + group, anomaly_dim)
            pick = train & (groups == group)
            block = meta.loc[pick, ["district_id", "season_start_year"]].copy()
            block[columns] = representation(model, fold, pick)
            block["representation_train_end"] = test_year - 1
            rows.append(block)

        model = train_encoder(fold, train, years[train], SEED, anomaly_dim)
        pick = years == test_year
        block = meta.loc[pick, ["district_id", "season_start_year"]].copy()
        block[columns] = representation(model, fold, pick)
        block["representation_train_end"] = test_year - 1
        rows.append(block)

        audit.append({"test_year": test_year, "train_rows": int(train.sum()),
                      "masked_reconstruction_loss": model.reconstruction_loss,
                      "total_loss": model.total_loss,
                      "parameters": sum(p.numel() for p in model.parameters())})
        print(f"  fold {test_year}: {int(train.sum())} rows, "
              f"masked-tile loss {model.reconstruction_loss:.4f}, "
              f"total {model.total_loss:.4f}", flush=True)

    features = pd.concat(rows, ignore_index=True)
    features.to_parquet(DATA / "v16_unified_features.parquet", index=False)
    pd.DataFrame(audit).to_csv(ARTIFACTS / "unified_encoder_audit.csv", index=False)
    print(f"\nwrote {len(features)} rows, "
          f"{sum(p.numel() for p in model.parameters())} parameters, {DEVICE}")


if __name__ == "__main__":
    main()
