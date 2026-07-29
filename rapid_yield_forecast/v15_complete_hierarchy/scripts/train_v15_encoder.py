#!/usr/bin/env python3
"""Strict MODIS pretraining and Sentinel/weather fine-tuning for V15."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.set_num_threads(int(os.environ.get("V15_TORCH_THREADS", "6")))
V15 = Path(__file__).resolve().parents[1]
RAPID = V15.parent
DATA = V15 / "data"
ARTIFACTS = V15 / "artifacts"
MODELS = V15 / "models"
V12_DATA = RAPID / "v12_cross_attention_yield" / "data" / "v12_dataset.npz"
V12_META = RAPID / "v12_cross_attention_yield" / "data" / "metadata.parquet"

DEVICE_NAME = os.environ.get(
    "V15_DEVICE",
    "mps" if torch.backends.mps.is_available() else "cpu",
)
DEVICE = torch.device(DEVICE_NAME)
SEEDS = tuple(int(value) for value in os.environ.get("V15_SEEDS", "42,73").split(","))
PRETRAIN_EPOCHS = int(os.environ.get("V15_MODIS_EPOCHS", "24"))
FINETUNE_EPOCHS = int(os.environ.get("V15_SENTINEL_EPOCHS", "45"))
CLOCKS = ("jan15", "feb15", "mar05")
CLOCK_INDEX = {clock: index for index, clock in enumerate(CLOCKS)}
FOLDS = ((2018, [2019]), (2019, [2020]), (2020, [2021, 2022]))
VARIANTS = ("scratch", "modis_pretrained")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_group(district_id: str) -> int:
    return sum(ord(character) for character in district_id) % 3


def finite_scale(
    values: np.ndarray,
    axes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axes)
    sd = np.nanstd(values, axis=axes)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd) & (sd > 1e-5), sd, 1.0).astype(np.float32)
    return mean, sd


def robust_scale(
    values: np.ndarray,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=axis)
    mad = 1.4826 * np.nanmedian(
        np.abs(values - np.expand_dims(center, axis)), axis=axis
    )
    fallback = np.nanstd(values, axis=axis)
    scale = np.where(np.isfinite(mad) & (mad > 1e-5), mad, fallback)
    return (
        np.where(np.isfinite(center), center, 0.0).astype(np.float32),
        np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.0).astype(np.float32),
    )


@dataclass
class EncoderScale:
    modis_mean: np.ndarray
    modis_sd: np.ndarray
    sentinel_mean: np.ndarray
    sentinel_sd: np.ndarray
    state_mean: np.ndarray
    state_sd: np.ndarray
    future_mean: np.ndarray
    future_sd: np.ndarray
    delta_center: np.ndarray
    delta_scale: np.ndarray


class TransferCropEncoder(nn.Module):
    """Temporal crop encoder pretrained on MODIS and adapted to Sentinel."""

    def __init__(self, modis_dim: int, hidden: int = 32):
        super().__init__()
        sentinel_dim = 6 * 21
        self.modis_dim = modis_dim
        self.hidden = hidden
        self.modis_adapter = nn.Linear(modis_dim, hidden)
        self.sentinel_adapter = nn.Linear(sentinel_dim, hidden)
        self.crop_pos = nn.Parameter(torch.randn(1, 3, hidden) * 0.02)
        crop_layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 2, dropout=0.08,
            batch_first=True, norm_first=True, activation="gelu",
        )
        weather_layer = lambda: nn.TransformerEncoderLayer(
            hidden, 4, hidden * 2, dropout=0.08,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.crop_encoder = nn.TransformerEncoder(crop_layer, 2)
        self.state_adapter = nn.Linear(16, hidden)
        self.future_adapter = nn.Linear(16, hidden)
        self.state_pos = nn.Parameter(torch.randn(1, 6, hidden) * 0.02)
        self.future_pos = nn.Parameter(torch.randn(1, 10, hidden) * 0.02)
        self.state_encoder = nn.TransformerEncoder(weather_layer(), 1)
        self.future_encoder = nn.TransformerEncoder(weather_layer(), 1)
        self.crop_to_state = nn.MultiheadAttention(
            hidden, 4, dropout=0.06, batch_first=True
        )
        self.crop_to_future = nn.MultiheadAttention(
            hidden, 4, dropout=0.06, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden)
        self.modis_head = nn.Sequential(
            nn.Linear(hidden, 48), nn.GELU(), nn.Linear(48, modis_dim)
        )
        self.sentinel_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, sentinel_dim)
        )
        self.sentinel_sign = nn.Linear(hidden, sentinel_dim)

    @staticmethod
    def last_token(
        encoded: torch.Tensor,
        source_position: torch.Tensor,
    ) -> torch.Tensor:
        batch = torch.arange(encoded.shape[0], device=encoded.device)
        return encoded[batch, source_position].unsqueeze(1)

    def encode_crop(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        source_position: torch.Tensor,
        modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adapter = (
            self.modis_adapter if modality == "modis"
            else self.sentinel_adapter
        )
        encoded = self.crop_encoder(
            adapter(sequence) + self.crop_pos,
            src_key_padding_mask=~mask,
        )
        return encoded, self.last_token(encoded, source_position)

    def forward_modis(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        source_position: torch.Tensor,
    ) -> torch.Tensor:
        _, current = self.encode_crop(
            sequence, mask, source_position, "modis"
        )
        return self.modis_head(current.squeeze(1))

    def forward_sentinel(
        self,
        sequence: torch.Tensor,
        crop_mask: torch.Tensor,
        source_position: torch.Tensor,
        state: torch.Tensor,
        future: torch.Tensor,
        state_mask: torch.Tensor,
        future_mask: torch.Tensor,
        use_future: bool = True,
    ) -> dict[str, torch.Tensor]:
        crop_encoded, current = self.encode_crop(
            sequence, crop_mask, source_position, "sentinel"
        )
        state_encoded = self.state_encoder(
            self.state_adapter(state) + self.state_pos,
            src_key_padding_mask=~state_mask,
        )
        state_context, _ = self.crop_to_state(
            current, state_encoded, state_encoded,
            key_padding_mask=~state_mask, need_weights=False,
        )
        future_context = torch.zeros_like(current)
        future_pool = torch.zeros(
            current.shape[0], self.hidden,
            device=current.device, dtype=current.dtype,
        )
        if use_future:
            future_encoded = self.future_encoder(
                self.future_adapter(future) + self.future_pos,
                src_key_padding_mask=~future_mask,
            )
            future_context, _ = self.crop_to_future(
                current, future_encoded, future_encoded,
                key_padding_mask=~future_mask, need_weights=False,
            )
            weight = future_mask.float().unsqueeze(-1)
            future_pool = (
                (future_encoded * weight).sum(1)
                / weight.sum(1).clamp_min(1)
            )
        fused = self.norm(current + state_context + future_context)
        crop_weight = crop_mask.float().unsqueeze(-1)
        crop_pool = (
            (crop_encoded * crop_weight).sum(1)
            / crop_weight.sum(1).clamp_min(1)
        )
        state_weight = state_mask.float().unsqueeze(-1)
        state_pool = (
            (state_encoded * state_weight).sum(1)
            / state_weight.sum(1).clamp_min(1)
        )
        return {
            "delta": self.sentinel_head(fused.squeeze(1)),
            "sign": self.sentinel_sign(fused.squeeze(1)),
            "crop_pool": crop_pool,
            "state_pool": state_pool,
            "future_pool": future_pool,
            "fused_pool": fused.squeeze(1),
            "state_context": state_context.squeeze(1),
            "future_context": future_context.squeeze(1),
        }


def load_data() -> tuple[
    np.ndarray, np.ndarray, pd.DataFrame,
    dict[str, np.ndarray], pd.DataFrame,
]:
    modis_packed = np.load(DATA / "modis_sequences_2000_2022.npz")
    modis = modis_packed["sequence"].astype(np.float32)
    modis_mask = modis_packed["mask"].astype(bool)
    modis_meta = pd.read_parquet(DATA / "modis_metadata.parquet")

    sentinel_packed = np.load(V12_DATA)
    sentinel = {name: sentinel_packed[name] for name in sentinel_packed.files}
    sentinel["crop"] = sentinel["crop"].copy()
    invalid_psri = np.abs(sentinel["crop"][:, 5, :]) > 2
    sentinel["crop"][:, 5, :][invalid_psri] = np.nan
    sentinel_meta = pd.read_parquet(V12_META).reset_index(drop=True)
    return modis, modis_mask, modis_meta, sentinel, sentinel_meta


def sentinel_sequences(
    crop: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        (row.district_id, int(row.season_start_year), row.clock): index
        for index, row in enumerate(meta.itertuples(index=False))
    }
    flattened = crop.reshape(len(crop), -1)
    sequences = np.full((len(meta), 3, flattened.shape[1]), np.nan, np.float32)
    mask = np.zeros((len(meta), 3), bool)
    for index, row in enumerate(meta.itertuples(index=False)):
        current_position = CLOCK_INDEX[row.clock]
        for position, clock in enumerate(CLOCKS):
            if position > current_position:
                continue
            source = lookup.get((row.district_id, int(row.season_start_year), clock))
            if source is not None:
                sequences[index, position] = flattened[source]
                mask[index, position] = np.isfinite(flattened[source]).any()
    return sequences, mask


def transitions(meta: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (row.district_id, int(row.season_start_year), row.clock): index
        for index, row in enumerate(meta.itertuples(index=False))
    }
    rows = []
    for district in meta["district_id"].unique():
        for year in sorted(meta["season_start_year"].unique()):
            for source_clock, target_clock in (
                ("jan15", "feb15"), ("feb15", "mar05")
            ):
                source = lookup[(district, int(year), source_clock)]
                target = lookup[(district, int(year), target_clock)]
                rows.append({
                    "source_index": source,
                    "target_index": target,
                    "district_id": district,
                    "state_name": meta.loc[source, "state_name"],
                    "season_start_year": int(year),
                    "source_clock": source_clock,
                    "target_clock": target_clock,
                    "district_group": stable_group(district),
                })
    return pd.DataFrame(rows)


def build_scale(
    modis: np.ndarray,
    modis_meta: pd.DataFrame,
    sentinel: dict[str, np.ndarray],
    sentinel_sequence: np.ndarray,
    transition: pd.DataFrame,
    train_end: int,
    excluded_group: int | None,
) -> EncoderScale:
    modis_selected = modis_meta["season_start_year"].le(train_end).to_numpy()
    if excluded_group is not None:
        modis_selected &= modis_meta["district_group"].ne(excluded_group).to_numpy()
    modis_mean, modis_sd = finite_scale(modis[modis_selected], (0, 1))

    selected = transition["season_start_year"].le(train_end).to_numpy()
    if excluded_group is not None:
        selected &= transition["district_group"].ne(excluded_group).to_numpy()
    source = transition.loc[selected, "source_index"].to_numpy(int)
    target = transition.loc[selected, "target_index"].to_numpy(int)
    sentinel_values = np.concatenate([
        sentinel_sequence[source], sentinel_sequence[target]
    ], axis=0)
    sentinel_mean, sentinel_sd = finite_scale(sentinel_values, (0, 1))
    state_mean, state_sd = finite_scale(sentinel["state"][source], (0, 1))
    future_mean, future_sd = finite_scale(sentinel["future"][source], (0, 1))
    current = sentinel["crop"][source].reshape(len(source), -1)
    following = sentinel["crop"][target].reshape(len(target), -1)
    delta_center, delta_scale = robust_scale(following - current, axis=0)
    return EncoderScale(
        modis_mean, modis_sd, sentinel_mean, sentinel_sd,
        state_mean, state_sd, future_mean, future_sd,
        delta_center, delta_scale,
    )


def modis_transition_arrays(
    modis: np.ndarray,
    mask: np.ndarray,
    meta: pd.DataFrame,
    scale: EncoderScale,
    train_end: int,
    excluded_group: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_rows = meta["season_start_year"].le(train_end).to_numpy()
    if excluded_group is not None:
        selected_rows &= meta["district_group"].ne(excluded_group).to_numpy()
    selected_indices = np.where(selected_rows)[0]
    sequence_rows = []
    mask_rows = []
    position_rows = []
    target_rows = []
    target_mask_rows = []
    normalised = (modis - scale.modis_mean) / scale.modis_sd
    for index in selected_indices:
        for source_position in (0, 1):
            target_position = source_position + 1
            prefix = normalised[index].copy()
            prefix[target_position:] = np.nan
            prefix_mask = mask[index].copy()
            prefix_mask[target_position:] = False
            target = normalised[index, target_position]
            target_mask = np.isfinite(modis[index, target_position])
            if target_mask.sum() < max(5, modis.shape[2] // 4):
                continue
            sequence_rows.append(np.nan_to_num(prefix).astype(np.float32))
            mask_rows.append(prefix_mask)
            position_rows.append(source_position)
            target_rows.append(np.nan_to_num(target).astype(np.float32))
            target_mask_rows.append(target_mask)
    return (
        np.asarray(sequence_rows),
        np.asarray(mask_rows),
        np.asarray(position_rows, dtype=np.int64),
        np.asarray(target_rows),
        np.asarray(target_mask_rows),
    )


def train_modis(
    model: TransferCropEncoder,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    seed: int,
) -> float:
    sequence, mask, position, target, target_mask = arrays
    set_seed(seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=7e-4
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(sequence))),
        batch_size=256, shuffle=True,
    )
    last = math.nan
    model.train()
    for _ in range(PRETRAIN_EPOCHS):
        losses = []
        for (index,) in loader:
            ii = index.numpy()
            prediction = model.forward_modis(
                torch.from_numpy(sequence[ii]).to(DEVICE),
                torch.from_numpy(mask[ii]).to(DEVICE),
                torch.from_numpy(position[ii]).to(DEVICE),
            )
            truth = torch.from_numpy(target[ii]).to(DEVICE)
            valid = torch.from_numpy(target_mask[ii]).to(DEVICE)
            loss = torch.nn.functional.smooth_l1_loss(
                prediction[valid], truth[valid], beta=0.5
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        last = float(np.mean(losses))
    return last


def sentinel_arrays(
    sentinel: dict[str, np.ndarray],
    sequence: np.ndarray,
    sequence_mask: np.ndarray,
    transition: pd.DataFrame,
    scale: EncoderScale,
    train_end: int,
    excluded_group: int | None,
) -> dict[str, np.ndarray]:
    selected = transition["season_start_year"].le(train_end).to_numpy()
    if excluded_group is not None:
        selected &= transition["district_group"].ne(excluded_group).to_numpy()
    source = transition.loc[selected, "source_index"].to_numpy(int)
    target = transition.loc[selected, "target_index"].to_numpy(int)
    current = sentinel["crop"][source].reshape(len(source), -1)
    following = sentinel["crop"][target].reshape(len(target), -1)
    delta_raw = following - current
    return {
        "source": source,
        "sequence": np.nan_to_num(
            (sequence[source] - scale.sentinel_mean) / scale.sentinel_sd
        ).astype(np.float32),
        "crop_mask": sequence_mask[source],
        "position": np.asarray([
            CLOCK_INDEX[clock]
            for clock in transition.loc[selected, "source_clock"]
        ], dtype=np.int64),
        "state": np.nan_to_num(
            (sentinel["state"][source] - scale.state_mean) / scale.state_sd
        ).astype(np.float32),
        "future": np.nan_to_num(
            (sentinel["future"][source] - scale.future_mean) / scale.future_sd
        ).astype(np.float32),
        "state_mask": sentinel["state_mask"][source].astype(bool),
        "future_mask": sentinel["future_mask"][source].astype(bool),
        "delta": np.nan_to_num(
            (delta_raw - scale.delta_center) / scale.delta_scale
        ).astype(np.float32),
        "delta_mask": np.isfinite(delta_raw),
        "sign": (np.nan_to_num(delta_raw) > 0).astype(np.float32),
    }


def train_sentinel(
    model: TransferCropEncoder,
    arrays: dict[str, np.ndarray],
    seed: int,
) -> float:
    set_seed(seed + 7000)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=6e-4, weight_decay=9e-4
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(arrays["source"]))),
        batch_size=128, shuffle=True,
    )
    last = math.nan
    model.train()
    for _ in range(FINETUNE_EPOCHS):
        losses = []
        for (index,) in loader:
            ii = index.numpy()
            # Future-branch dropout teaches one network both experienced-only
            # and experienced-plus-forecast representations.
            use_future = bool(np.random.random() > 0.30)
            output = model.forward_sentinel(
                torch.from_numpy(arrays["sequence"][ii]).to(DEVICE),
                torch.from_numpy(arrays["crop_mask"][ii]).to(DEVICE),
                torch.from_numpy(arrays["position"][ii]).to(DEVICE),
                torch.from_numpy(arrays["state"][ii]).to(DEVICE),
                torch.from_numpy(arrays["future"][ii]).to(DEVICE),
                torch.from_numpy(arrays["state_mask"][ii]).to(DEVICE),
                torch.from_numpy(arrays["future_mask"][ii]).to(DEVICE),
                use_future=use_future,
            )
            truth = torch.from_numpy(arrays["delta"][ii]).to(DEVICE)
            valid = torch.from_numpy(arrays["delta_mask"][ii]).to(DEVICE)
            sign = torch.from_numpy(arrays["sign"][ii]).to(DEVICE)
            loss = torch.nn.functional.smooth_l1_loss(
                output["delta"][valid], truth[valid], beta=0.5
            )
            loss = loss + 0.08 * torch.nn.functional.binary_cross_entropy_with_logits(
                output["sign"][valid], sign[valid]
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        last = float(np.mean(losses))
    return last


def transform_rows(
    sentinel: dict[str, np.ndarray],
    sequence: np.ndarray,
    sequence_mask: np.ndarray,
    meta: pd.DataFrame,
    indices: np.ndarray,
    scale: EncoderScale,
) -> dict[str, np.ndarray]:
    return {
        "sequence": np.nan_to_num(
            (sequence[indices] - scale.sentinel_mean) / scale.sentinel_sd
        ).astype(np.float32),
        "crop_mask": sequence_mask[indices],
        "position": np.asarray([
            CLOCK_INDEX[clock] for clock in meta.loc[indices, "clock"]
        ], dtype=np.int64),
        "state": np.nan_to_num(
            (sentinel["state"][indices] - scale.state_mean) / scale.state_sd
        ).astype(np.float32),
        "future": np.nan_to_num(
            (sentinel["future"][indices] - scale.future_mean) / scale.future_sd
        ).astype(np.float32),
        "state_mask": sentinel["state_mask"][indices].astype(bool),
        "future_mask": sentinel["future_mask"][indices].astype(bool),
    }


def predict_outputs(
    model: TransferCropEncoder,
    arrays: dict[str, np.ndarray],
    scale: EncoderScale,
) -> dict[str, np.ndarray]:
    outputs = {}
    model.eval()
    for use_future, label in [(False, "no_future"), (True, "full")]:
        blocks: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for start in range(0, len(arrays["sequence"]), 256):
                stop = min(start + 256, len(arrays["sequence"]))
                output = model.forward_sentinel(
                    torch.from_numpy(arrays["sequence"][start:stop]).to(DEVICE),
                    torch.from_numpy(arrays["crop_mask"][start:stop]).to(DEVICE),
                    torch.from_numpy(arrays["position"][start:stop]).to(DEVICE),
                    torch.from_numpy(arrays["state"][start:stop]).to(DEVICE),
                    torch.from_numpy(arrays["future"][start:stop]).to(DEVICE),
                    torch.from_numpy(arrays["state_mask"][start:stop]).to(DEVICE),
                    torch.from_numpy(arrays["future_mask"][start:stop]).to(DEVICE),
                    use_future=use_future,
                )
                for name, values in output.items():
                    blocks.setdefault(name, []).append(values.cpu().numpy())
        for name, values in blocks.items():
            outputs[f"{label}_{name}"] = np.concatenate(values)
        outputs[f"{label}_delta_raw"] = (
            outputs[f"{label}_delta"] * scale.delta_scale
            + scale.delta_center
        )
    return outputs


def compact_features(
    outputs: dict[str, np.ndarray],
    sentinel_raw: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    columns = []
    values = []
    for branch in ("no_future", "full"):
        for pool in (
            "crop_pool", "state_pool", "future_pool",
            "fused_pool", "state_context", "future_context",
        ):
            matrix = outputs[f"{branch}_{pool}"]
            # Keep a compact half-width view; downstream state and district
            # models have very few independent seasons.
            for index in range(16):
                columns.append(f"{branch}_{pool}_{index:02d}")
                values.append(matrix[:, index])
        delta = outputs[f"{branch}_delta_raw"].reshape(len(sentinel_raw), 6, 21)
        for index in range(6):
            columns.append(f"{branch}_delta_index_{index}_mean")
            values.append(np.nanmean(delta[:, index], axis=1))
        columns.extend([
            f"{branch}_delta_abs_mean",
            f"{branch}_delta_positive_fraction",
        ])
        values.extend([
            np.nanmean(np.abs(delta), axis=(1, 2)),
            np.nanmean(delta > 0, axis=(1, 2)),
        ])
    effect = outputs["full_fused_pool"] - outputs["no_future_fused_pool"]
    for index in range(16):
        columns.append(f"future_effect_{index:02d}")
        values.append(effect[:, index])
    raw = sentinel_raw.reshape(len(sentinel_raw), 6, 21)
    for index in range(6):
        columns.append(f"current_index_{index}_mean")
        values.append(np.nanmean(raw[:, index], axis=1))
    return np.stack(values, axis=1), columns


def trajectory_predictions(
    model: TransferCropEncoder,
    sentinel: dict[str, np.ndarray],
    sequence: np.ndarray,
    sequence_mask: np.ndarray,
    meta: pd.DataFrame,
    transition: pd.DataFrame,
    scale: EncoderScale,
    test_years: list[int],
    variant: str,
    seed: int,
) -> pd.DataFrame:
    selected = transition["season_start_year"].isin(test_years).to_numpy()
    source = transition.loc[selected, "source_index"].to_numpy(int)
    target = transition.loc[selected, "target_index"].to_numpy(int)
    arrays = transform_rows(
        sentinel, sequence, sequence_mask, meta, source, scale
    )
    output = predict_outputs(model, arrays, scale)
    current = sentinel["crop"][source].reshape(len(source), -1)
    following = sentinel["crop"][target].reshape(len(target), -1)
    rows = []
    base = transition.loc[selected].reset_index(drop=True)
    for branch in ("no_future", "full"):
        predicted = current + output[f"{branch}_delta_raw"]
        for index, meta_row in base.iterrows():
            valid = np.isfinite(following[index]) & np.isfinite(current[index])
            error = predicted[index, valid] - following[index, valid]
            persistence_error = current[index, valid] - following[index, valid]
            rows.append({
                **meta_row.to_dict(),
                "variant": variant,
                "branch": branch,
                "seed": seed,
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
                "persistence_rmse": float(np.sqrt(
                    np.mean(persistence_error ** 2)
                )),
            })
    return pd.DataFrame(rows)


def train_one_encoder(
    modis: np.ndarray,
    modis_mask: np.ndarray,
    modis_meta: pd.DataFrame,
    sentinel: dict[str, np.ndarray],
    sequence: np.ndarray,
    sequence_mask: np.ndarray,
    transition: pd.DataFrame,
    train_end: int,
    excluded_group: int | None,
    variant: str,
    seed: int,
) -> tuple[TransferCropEncoder, EncoderScale, dict[str, object]]:
    set_seed(seed + 100 * train_end + (excluded_group or 9))
    scale = build_scale(
        modis, modis_meta, sentinel, sequence,
        transition, train_end, excluded_group,
    )
    model = TransferCropEncoder(modis.shape[2]).to(DEVICE)
    modis_loss = math.nan
    modis_rows = 0
    if variant == "modis_pretrained":
        arrays = modis_transition_arrays(
            modis, modis_mask, modis_meta, scale,
            train_end, excluded_group,
        )
        modis_rows = len(arrays[0])
        modis_loss = train_modis(model, arrays, seed + train_end)
    sentinel_train = sentinel_arrays(
        sentinel, sequence, sequence_mask, transition,
        scale, train_end, excluded_group,
    )
    sentinel_loss = train_sentinel(model, sentinel_train, seed + train_end)
    return model, scale, {
        "train_end": train_end,
        "excluded_group": -1 if excluded_group is None else excluded_group,
        "variant": variant,
        "seed": seed,
        "modis_rows": modis_rows,
        "modis_epochs": PRETRAIN_EPOCHS if variant == "modis_pretrained" else 0,
        "modis_final_loss": modis_loss,
        "sentinel_rows": len(sentinel_train["source"]),
        "sentinel_epochs": FINETUNE_EPOCHS,
        "sentinel_final_loss": sentinel_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(DEVICE),
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    modis, modis_mask, modis_meta, sentinel, meta = load_data()
    sequence, sequence_mask = sentinel_sequences(sentinel["crop"], meta)
    transition = transitions(meta)
    feature_rows = []
    audits = []
    trajectory_rows = []
    feature_columns: list[str] | None = None

    for train_end, test_years in FOLDS:
        for variant in VARIANTS:
            for role_group in [None, 0, 1, 2]:
                models = []
                scales = []
                for seed in SEEDS:
                    model, scale, audit = train_one_encoder(
                        modis, modis_mask, modis_meta, sentinel,
                        sequence, sequence_mask, transition,
                        train_end, role_group, variant, seed,
                    )
                    models.append(model)
                    scales.append(scale)
                    audits.append(audit)
                    if role_group is None:
                        trajectory_rows.append(trajectory_predictions(
                            model, sentinel, sequence, sequence_mask,
                            meta, transition, scale, test_years,
                            variant, seed,
                        ))
                        if train_end == 2020:
                            torch.save({
                                "state_dict": model.state_dict(),
                                "scale": asdict(scale),
                                "variant": variant,
                                "seed": seed,
                                "train_end": train_end,
                                "modis_dim": modis.shape[2],
                                "score_claimed_for_refit": True,
                            }, MODELS / f"encoder_{variant}_seed{seed}_through2020.pt")

                if role_group is None:
                    selected = meta[
                        meta["season_start_year"].isin(test_years)
                        & meta["clock"].eq("mar05")
                    ].index.to_numpy()
                    feature_role = "test_full"
                    target_years = test_years
                else:
                    selected = meta[
                        meta["season_start_year"].between(2017, train_end)
                        & meta["clock"].eq("mar05")
                        & meta["district_id"].map(stable_group).eq(role_group)
                    ].index.to_numpy()
                    feature_role = "train_crossfit"
                    target_years = sorted(
                        meta.loc[selected, "season_start_year"].unique().tolist()
                    )
                seed_features = []
                for model, scale in zip(models, scales):
                    arrays = transform_rows(
                        sentinel, sequence, sequence_mask, meta,
                        selected, scale,
                    )
                    output = predict_outputs(model, arrays, scale)
                    matrix, columns = compact_features(
                        output, sentinel["crop"][selected]
                    )
                    feature_columns = columns
                    seed_features.append(matrix)
                averaged = np.mean(seed_features, axis=0)
                block = meta.loc[selected, [
                    "district_id", "state_name", "district_name",
                    "season_start_year", "clock",
                ]].reset_index(drop=True)
                block["representation_train_end"] = train_end
                block["feature_role"] = feature_role
                block["held_group"] = (
                    -1 if role_group is None else role_group
                )
                block["encoder_variant"] = variant
                encoded_block = pd.DataFrame(
                    averaged,
                    columns=[f"enc__{column}" for column in feature_columns],
                )
                block = pd.concat([block, encoded_block], axis=1)
                feature_rows.append(block)
                print(
                    f"finished train_end={train_end} variant={variant} "
                    f"group={role_group} years={target_years}"
                )

    features = pd.concat(feature_rows, ignore_index=True)
    features.to_parquet(
        DATA / "strict_transfer_encoder_features.parquet", index=False
    )
    pd.DataFrame(audits).to_csv(
        ARTIFACTS / "encoder_training_audit.csv", index=False
    )
    trajectory = pd.concat(trajectory_rows, ignore_index=True)
    trajectory.to_parquet(
        ARTIFACTS / "encoder_trajectory_predictions.parquet", index=False
    )
    trajectory_metrics = (
        trajectory.groupby(["variant", "branch"], as_index=False)
        .agg(
            rows=("rmse", "size"),
            mean_transition_rmse=("rmse", "mean"),
            mean_transition_mae=("mae", "mean"),
            mean_persistence_rmse=("persistence_rmse", "mean"),
        )
    )
    trajectory_metrics.to_csv(
        ARTIFACTS / "encoder_trajectory_metrics.csv", index=False
    )
    manifest = {
        "rows": len(features),
        "features": len(feature_columns or []),
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "folds": {
            str(train_end): years for train_end, years in FOLDS
        },
        "district_crossfit_groups": 3,
        "modis_pretrain_year_start": 2000,
        "sentinel_year_start": 2017,
        "modis_epochs": PRETRAIN_EPOCHS,
        "sentinel_epochs": FINETUNE_EPOCHS,
        "device": str(DEVICE),
        "yield_labels_used": False,
        "later_satellite_used_as_input": False,
        "post_2022_yield_labels_read": False,
    }
    (DATA / "encoder_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps({
        "manifest": manifest,
        "trajectory_metrics": trajectory_metrics.to_dict("records"),
    }, indent=2))


if __name__ == "__main__":
    main()
