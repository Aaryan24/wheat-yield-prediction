#!/usr/bin/env python3
"""V13: pretrain future weather on crop response, then run strict promotion tests."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore", message="enable_nested_tensor")
torch.set_num_threads(int(os.environ.get("V13_TORCH_THREADS", "6")))

V13 = Path(__file__).resolve().parents[1]
ROOT = V13.parents[1]
sys.path.insert(0, str(ROOT))
V12 = V13.parent / "v12_cross_attention_yield"
V11 = V13.parent / "v11_global_wheat_transfer"
DATA = V12 / "data"
OUT = V13 / "artifacts"
MODEL_DIR = V13 / "models"

from rapid_yield_forecast.v12_cross_attention_yield.scripts import run_v12_lab as v12lab  # noqa: E402


DEVICE_NAME = os.environ.get("V13_DEVICE", "cpu")
DEVICE = torch.device(DEVICE_NAME)
v12lab.DEVICE = DEVICE

CLOCKS = ("jan15", "feb15", "mar05")
CLOCK_ORDER = {"jan15": 0, "feb15": 1, "mar05": 2}
FOLDS = ((2018, 2019, "development"), (2019, 2020, "development"), (2020, 2021, "late"), (2020, 2022, "late"))
SEEDS = tuple(int(x) for x in os.environ.get("V13_SEEDS", "42,73").split(","))
EPOCHS = int(os.environ.get("V13_EPOCHS", "60"))
VARIANTS = ("crop_only", "no_future", "full")
VIEWS = ("response", "response_tabular")
DYNAMIC_COLUMNS = np.asarray([2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finite_scale(values: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axes)
    sd = np.nanstd(values, axis=axes)
    return (
        np.where(np.isfinite(mean), mean, 0).astype(np.float32),
        np.where(np.isfinite(sd) & (sd > 1e-5), sd, 1).astype(np.float32),
    )


def robust_scale(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=axis)
    mad = np.nanmedian(np.abs(values - np.expand_dims(median, axis)), axis=axis) * 1.4826
    fallback = np.nanstd(values, axis=axis)
    scale = np.where(np.isfinite(mad) & (mad > 1e-5), mad, fallback)
    return (
        np.where(np.isfinite(median), median, 0).astype(np.float32),
        np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1).astype(np.float32),
    )


@dataclass
class ResponseScale:
    crop_mean: np.ndarray
    crop_sd: np.ndarray
    delta_center: np.ndarray
    delta_scale: np.ndarray
    state_mean: np.ndarray
    state_sd: np.ndarray
    future_mean: np.ndarray
    future_sd: np.ndarray


def load_inputs() -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    packed = np.load(DATA / "v12_dataset.npz")
    data = {name: packed[name] for name in packed.files}
    # PSRI is a ratio and becomes numerically singular when its denominator is
    # near zero. The five other indices have no values outside +/-2, while PSRI
    # contains a small set as large as -102 and +63. They are sensor/math
    # failures, not biological crop states, so mark only those cells missing.
    data["crop"] = data["crop"].copy()
    invalid_psri = np.abs(data["crop"][:, 5, :]) > 2
    data["crop"][:, 5, :][invalid_psri] = np.nan
    data["_invalid_psri_removed"] = np.asarray(int(invalid_psri.sum()))
    meta = pd.read_parquet(DATA / "metadata.parquet").reset_index(drop=True)
    lookup = {
        (row.district_id, int(row.season_start_year), row.clock): i
        for i, row in enumerate(meta.itertuples(index=False))
    }
    rows = []
    for district_id in meta["district_id"].unique():
        for year in sorted(meta["season_start_year"].unique()):
            for source_clock, target_clock in (("jan15", "feb15"), ("feb15", "mar05")):
                source = lookup.get((district_id, int(year), source_clock))
                target = lookup.get((district_id, int(year), target_clock))
                if source is not None and target is not None:
                    rows.append({
                        "source_index": source,
                        "target_index": target,
                        "district_id": district_id,
                        "state_name": meta.loc[source, "state_name"],
                        "season_start_year": int(year),
                        "source_clock": source_clock,
                        "target_clock": target_clock,
                    })
    transitions = pd.DataFrame(rows)
    if len(transitions) != 119 * 6 * 2:
        raise RuntimeError(f"Expected 1428 crop transitions, got {len(transitions)}")
    return data, meta, transitions


class CropResponseNet(nn.Module):
    """Small physical bottleneck: forecast the next satellite crop state."""

    def __init__(self, variant: str, weather_weights: dict[str, torch.Tensor] | None = None):
        super().__init__()
        hidden = 32
        self.variant = variant
        self.crop_proj = nn.Linear(21, hidden)
        self.state_proj = nn.Linear(16, hidden)
        self.future_proj = nn.Linear(16, hidden)
        self.crop_pos = nn.Parameter(torch.randn(1, 6, hidden) * 0.02)
        self.state_pos = nn.Parameter(torch.randn(1, 6, hidden) * 0.02)
        self.future_pos = nn.Parameter(torch.randn(1, 10, hidden) * 0.02)

        def layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                hidden, 4, hidden * 2, dropout=0.08, batch_first=True,
                norm_first=True, activation="gelu",
            )

        self.crop_encoder = nn.TransformerEncoder(layer(), 1)
        self.state_encoder = nn.TransformerEncoder(layer(), 1)
        self.future_encoder = nn.TransformerEncoder(layer(), 1)
        self.crop_to_state = nn.MultiheadAttention(hidden, 4, dropout=0.06, batch_first=True)
        self.crop_to_future = nn.MultiheadAttention(hidden, 4, dropout=0.06, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.delta_head = nn.Sequential(nn.Linear(hidden, 48), nn.GELU(), nn.Linear(48, len(DYNAMIC_COLUMNS)))
        self.sign_head = nn.Linear(hidden, len(DYNAMIC_COLUMNS))
        if weather_weights is not None:
            own = self.state_dict()
            for key, value in weather_weights.items():
                if key in own and own[key].shape == value.shape:
                    own[key].copy_(value)

    @staticmethod
    def masked_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.float().unsqueeze(-1)
        return (sequence * weight).sum(1) / weight.sum(1).clamp_min(1)

    def forward(
        self,
        crop: torch.Tensor,
        state: torch.Tensor,
        future: torch.Tensor,
        state_mask: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        c = self.crop_encoder(self.crop_proj(crop) + self.crop_pos)
        zeros = torch.zeros_like(c)
        s = zeros
        f = torch.zeros(c.shape[0], 10, c.shape[2], device=c.device, dtype=c.dtype)
        state_context = zeros
        future_context = zeros
        if self.variant in ("no_future", "full"):
            s = self.state_encoder(
                self.state_proj(state) + self.state_pos,
                src_key_padding_mask=~state_mask,
            )
            state_context, _ = self.crop_to_state(
                c, s, s, key_padding_mask=~state_mask, need_weights=False
            )
        if self.variant == "full":
            f = self.future_encoder(
                self.future_proj(future) + self.future_pos,
                src_key_padding_mask=~future_mask,
            )
            future_context, _ = self.crop_to_future(
                c, f, f, key_padding_mask=~future_mask, need_weights=False
            )
        fused = self.norm(c + state_context + future_context)
        batch = crop.shape[0]
        zero_pool = torch.zeros(batch, c.shape[2], device=c.device, dtype=c.dtype)
        state_pool = self.masked_pool(s, state_mask) if self.variant in ("no_future", "full") else zero_pool
        future_pool = self.masked_pool(f, future_mask) if self.variant == "full" else zero_pool
        return {
            "delta": self.delta_head(fused),
            "sign": self.sign_head(fused),
            "c_pool": c.mean(1),
            "s_pool": state_pool,
            "f_pool": future_pool,
            "state_context_pool": state_context.mean(1),
            "future_context_pool": future_context.mean(1),
            "fused_pool": fused.mean(1),
        }


def response_scale(
    data: dict[str, np.ndarray],
    transitions: pd.DataFrame,
    train_end: int,
    sequence_scale: v12lab.SequenceScale,
) -> ResponseScale:
    train = transitions["season_start_year"].le(train_end).to_numpy()
    source = transitions.loc[train, "source_index"].to_numpy(int)
    target = transitions.loc[train, "target_index"].to_numpy(int)
    crop_values = np.concatenate([data["crop"][source], data["crop"][target]], axis=0)
    crop_mean, crop_sd = finite_scale(crop_values, (0,))
    current = data["crop"][source][:, :, DYNAMIC_COLUMNS]
    following = data["crop"][target][:, :, DYNAMIC_COLUMNS]
    delta_center, delta_sd = robust_scale(following - current, axis=0)
    return ResponseScale(
        crop_mean=crop_mean,
        crop_sd=crop_sd,
        delta_center=delta_center,
        delta_scale=delta_sd,
        state_mean=sequence_scale.state_mean.astype(np.float32),
        state_sd=sequence_scale.state_sd.astype(np.float32),
        future_mean=sequence_scale.future_mean.astype(np.float32),
        future_sd=sequence_scale.future_sd.astype(np.float32),
    )


def transformed(data: dict[str, np.ndarray], scale: ResponseScale) -> dict[str, np.ndarray]:
    return {
        "crop": np.nan_to_num((data["crop"] - scale.crop_mean) / scale.crop_sd).astype(np.float32),
        "state": np.nan_to_num((data["state"] - scale.state_mean) / scale.state_sd).astype(np.float32),
        "future": np.nan_to_num((data["future"] - scale.future_mean) / scale.future_sd).astype(np.float32),
        "state_mask": data["state_mask"].astype(bool),
        "future_mask": data["future_mask"].astype(bool),
    }


def train_response(
    data: dict[str, np.ndarray],
    transitions: pd.DataFrame,
    train_end: int,
    variant: str,
    seed: int,
    weather_weights: dict[str, torch.Tensor],
    sequence_scale: v12lab.SequenceScale,
) -> tuple[CropResponseNet, ResponseScale, float]:
    set_seed(seed + 1000 * train_end + 17 * list(VARIANTS).index(variant))
    scale = response_scale(data, transitions, train_end, sequence_scale)
    x = transformed(data, scale)
    selected = transitions["season_start_year"].le(train_end).to_numpy()
    source = transitions.loc[selected, "source_index"].to_numpy(int)
    target = transitions.loc[selected, "target_index"].to_numpy(int)
    current_raw = data["crop"][source][:, :, DYNAMIC_COLUMNS]
    target_raw = data["crop"][target][:, :, DYNAMIC_COLUMNS]
    delta_raw = target_raw - current_raw
    delta = ((delta_raw - scale.delta_center) / scale.delta_scale).astype(np.float32)
    mask = np.isfinite(delta_raw)
    delta = np.nan_to_num(delta).astype(np.float32)
    sign = (np.nan_to_num(delta_raw) > 0).astype(np.float32)

    model = CropResponseNet(variant, weather_weights).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=8e-4)
    loader = DataLoader(TensorDataset(torch.arange(len(source))), batch_size=128, shuffle=True)
    last = math.nan
    model.train()
    for _ in range(EPOCHS):
        losses = []
        for (local_idx,) in loader:
            ii = local_idx.numpy()
            source_idx = source[ii]
            out = model(
                torch.from_numpy(x["crop"][source_idx]).to(DEVICE),
                torch.from_numpy(x["state"][source_idx]).to(DEVICE),
                torch.from_numpy(x["future"][source_idx]).to(DEVICE),
                torch.from_numpy(x["state_mask"][source_idx]).to(DEVICE),
                torch.from_numpy(x["future_mask"][source_idx]).to(DEVICE),
            )
            truth = torch.from_numpy(delta[ii]).to(DEVICE)
            valid = torch.from_numpy(mask[ii]).to(DEVICE)
            target_sign = torch.from_numpy(sign[ii]).to(DEVICE)
            loss = torch.nn.functional.smooth_l1_loss(
                out["delta"][valid], truth[valid], beta=0.5
            )
            loss = loss + 0.10 * torch.nn.functional.binary_cross_entropy_with_logits(
                out["sign"][valid], target_sign[valid]
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        last = float(np.mean(losses))
    return model, scale, last


def response_predict(
    model: CropResponseNet,
    data: dict[str, np.ndarray],
    scale: ResponseScale,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = transformed(data, scale)
    predictions = []
    representations = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            ii = indices[start:start + 512]
            out = model(
                torch.from_numpy(x["crop"][ii]).to(DEVICE),
                torch.from_numpy(x["state"][ii]).to(DEVICE),
                torch.from_numpy(x["future"][ii]).to(DEVICE),
                torch.from_numpy(x["state_mask"][ii]).to(DEVICE),
                torch.from_numpy(x["future_mask"][ii]).to(DEVICE),
            )
            scaled_delta = out["delta"].cpu().numpy()
            raw_delta = scale.delta_center + scale.delta_scale * scaled_delta
            sign_probability = torch.sigmoid(out["sign"]).cpu().numpy()
            pools = np.concatenate(
                [
                    out["c_pool"].cpu().numpy(),
                    out["s_pool"].cpu().numpy(),
                    out["f_pool"].cpu().numpy(),
                    out["state_context_pool"].cpu().numpy(),
                    out["future_context_pool"].cpu().numpy(),
                    out["fused_pool"].cpu().numpy(),
                ],
                axis=1,
            )
            summaries = np.concatenate(
                [
                    raw_delta.mean(axis=1),
                    np.abs(raw_delta).mean(axis=1),
                    sign_probability.mean(axis=1),
                ],
                axis=1,
            )
            predictions.append(raw_delta)
            representations.append(np.concatenate([pools, summaries], axis=1).astype(np.float32))
    return np.concatenate(predictions), np.concatenate(representations)


def make_pipeline(kind: str, strength: float) -> Pipeline:
    if kind == "direction":
        model = LogisticRegression(C=strength, max_iter=5000, solver="liblinear")
    else:
        model = Ridge(alpha=strength)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def feature_view(
    representation: np.ndarray,
    tabular: np.ndarray,
    view: str,
) -> np.ndarray:
    if view == "response":
        return representation
    if view == "response_tabular":
        return np.concatenate([representation, tabular], axis=1)
    raise ValueError(view)


def current_direction_baseline() -> pd.DataFrame:
    frame = pd.read_parquet(V12 / "artifacts" / "direction_increment" / "selected_predictions.parquet")
    frame["baseline_probability"] = np.where(
        frame["clock"].eq("jan15"), frame["v11_probability"], frame["probability"]
    )
    return frame[[
        "district_id", "season_start_year", "clock", "period", "state_name",
        "target", "baseline_probability",
    ]].drop_duplicates(["district_id", "season_start_year", "clock"])


def grouped_auc_bootstrap(frame: pd.DataFrame, draws: int = 5000) -> dict[str, float]:
    groups = [part.index.to_numpy() for _, part in frame.groupby(["state_name", "season_start_year"])]
    rng = np.random.default_rng(20260726)
    gains = []
    for _ in range(draws):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        target = frame.loc[idx, "target"].to_numpy()
        if np.unique(target).size < 2:
            continue
        gains.append(
            roc_auc_score(target, frame.loc[idx, "probability"])
            - roc_auc_score(target, frame.loc[idx, "baseline_probability"])
        )
    values = np.asarray(gains)
    return {
        "draws": int(len(values)),
        "mean_auc_gain": float(values.mean()),
        "p025": float(np.quantile(values, 0.025)),
        "p975": float(np.quantile(values, 0.975)),
        "probability_positive": float(np.mean(values > 0)),
    }


def grouped_rmse_bootstrap(frame: pd.DataFrame, draws: int = 5000) -> dict[str, float]:
    groups = [part.index.to_numpy() for _, part in frame.groupby(["state_name", "season_start_year"])]
    rng = np.random.default_rng(20260727)
    gains = []
    for _ in range(draws):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        actual = frame.loc[idx, "actual"].to_numpy()
        anchor = frame.loc[idx, "anchor"].to_numpy()
        prediction = frame.loc[idx, "prediction"].to_numpy()
        gains.append(
            np.sqrt(np.mean((anchor - actual) ** 2))
            - np.sqrt(np.mean((prediction - actual) ** 2))
        )
    values = np.asarray(gains)
    return {
        "draws": draws,
        "mean_rmse_gain": float(values.mean()),
        "p025": float(np.quantile(values, 0.025)),
        "p975": float(np.quantile(values, 0.975)),
        "probability_positive": float(np.mean(values > 0)),
    }


def evaluate_trajectory(
    data: dict[str, np.ndarray],
    transitions: pd.DataFrame,
    prediction_store: dict[tuple[int, str], np.ndarray],
    scales: dict[int, ResponseScale],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows = []
    for _, row in transitions.iterrows():
        year = int(row["season_start_year"])
        if year not in (2019, 2020, 2021, 2022):
            continue
        train_end = 2018 if year == 2019 else 2019 if year == 2020 else 2020
        source = int(row["source_index"])
        target = int(row["target_index"])
        actual = data["crop"][target][:, DYNAMIC_COLUMNS] - data["crop"][source][:, DYNAMIC_COLUMNS]
        valid = np.isfinite(actual)
        for variant in VARIANTS:
            prediction = prediction_store[(train_end, variant)][source]
            error = prediction[valid] - actual[valid]
            persistence_error = -actual[valid]
            scaled_error = error / scales[train_end].delta_scale[valid]
            sample_rows.append({
                **row.to_dict(),
                "period": "development" if year <= 2020 else "late",
                "variant": variant,
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
                "scaled_rmse": float(np.sqrt(np.mean(scaled_error ** 2))),
                "direction_accuracy": float(np.mean((prediction[valid] > 0) == (actual[valid] > 0))),
                "persistence_rmse": float(np.sqrt(np.mean(persistence_error ** 2))),
                "persistence_mae": float(np.mean(np.abs(persistence_error))),
            })
    samples = pd.DataFrame(sample_rows)
    metrics = samples.groupby(["variant", "period"], as_index=False).agg(
        transition_rmse=("rmse", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        transition_mae=("mae", "mean"),
        scaled_rmse=("scaled_rmse", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        direction_accuracy=("direction_accuracy", "mean"),
        persistence_rmse=("persistence_rmse", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        persistence_mae=("persistence_mae", "mean"),
        samples=("rmse", "size"),
    )
    return samples, metrics


def trajectory_uncertainty(samples: pd.DataFrame, draws: int = 5000) -> pd.DataFrame:
    keys = [
        "district_id", "state_name", "season_start_year",
        "source_clock", "target_clock", "period",
    ]
    wide = samples.pivot(index=keys, columns="variant", values="rmse").reset_index()
    rows = []
    for period in ("development", "late"):
        block = wide[wide["period"].eq(period)].reset_index(drop=True)
        groups = [
            part.index.to_numpy()
            for _, part in block.groupby(["state_name", "season_start_year"])
        ]
        for comparator in ("crop_only", "no_future"):
            rng = np.random.default_rng(20260728)
            gains = []
            for _ in range(draws):
                idx = np.concatenate([
                    groups[j] for j in rng.integers(0, len(groups), len(groups))
                ])
                comparator_rmse = np.sqrt(np.mean(block.loc[idx, comparator].to_numpy() ** 2))
                full_rmse = np.sqrt(np.mean(block.loc[idx, "full"].to_numpy() ** 2))
                gains.append(comparator_rmse - full_rmse)
            values = np.asarray(gains)
            rows.append({
                "period": period,
                "comparison": f"{comparator}_minus_full",
                "mean_rmse_gain": float(values.mean()),
                "p025": float(np.quantile(values, 0.025)),
                "p975": float(np.quantile(values, 0.975)),
                "probability_positive": float(np.mean(values > 0)),
                "draws": draws,
            })
    return pd.DataFrame(rows)


def fit_downstream_folds(
    data: dict[str, np.ndarray],
    meta: pd.DataFrame,
    representations: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    years = meta["season_start_year"].to_numpy()
    direction_rows = []
    point_rows = []
    fitted: dict[str, Pipeline] = {}
    for clock in CLOCKS:
        clock_mask = meta["clock"].eq(clock).to_numpy()
        for train_end, test_year, period in FOLDS:
            train = clock_mask & (years <= train_end)
            test = np.where(clock_mask & (years == test_year))[0]
            for variant in VARIANTS:
                representation = representations[(train_end, variant)]
                for view in VIEWS:
                    features = feature_view(representation, data["tabular"], view)
                    for c_value in (0.005, 0.02, 0.1, 0.5):
                        model = make_pipeline("direction", c_value)
                        model.fit(features[train], data["increase"][train].astype(int))
                        probability = model.predict_proba(features[test])[:, 1]
                        for j, idx in enumerate(test):
                            direction_rows.append({
                                "district_id": meta.loc[idx, "district_id"],
                                "state_name": meta.loc[idx, "state_name"],
                                "season_start_year": test_year,
                                "clock": clock,
                                "period": period,
                                "variant": variant,
                                "view": view,
                                "c_value": c_value,
                                "target": int(data["increase"][idx]),
                                "v13_probability": float(probability[j]),
                            })
                    for head in ("direct", "change", "residual"):
                        if head == "direct":
                            target = data["yield_kg_per_ha"]
                        elif head == "change":
                            target = data["yield_kg_per_ha"] / data["lag_1_yield"] - 1
                        else:
                            target = data["yield_kg_per_ha"] / data["anchor_prediction"] - 1
                        for alpha in (100.0, 1000.0, 10000.0):
                            model = make_pipeline("point", alpha)
                            model.fit(features[train], target[train])
                            raw = model.predict(features[test])
                            if head == "change":
                                raw = data["lag_1_yield"][test] * (1 + raw)
                            elif head == "residual":
                                raw = data["anchor_prediction"][test] * (1 + raw)
                            for j, idx in enumerate(test):
                                point_rows.append({
                                    "district_id": meta.loc[idx, "district_id"],
                                    "state_name": meta.loc[idx, "state_name"],
                                    "season_start_year": test_year,
                                    "clock": clock,
                                    "period": period,
                                    "variant": variant,
                                    "view": view,
                                    "head": head,
                                    "alpha": alpha,
                                    "actual": float(data["yield_kg_per_ha"][idx]),
                                    "anchor": float(data["anchor_prediction"][idx]),
                                    "raw_prediction": float(raw[j]),
                                })
    return pd.DataFrame(direction_rows), pd.DataFrame(point_rows), fitted


def select_direction(direction: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = current_direction_baseline()
    merged = direction.merge(
        baseline,
        on=["district_id", "state_name", "season_start_year", "clock", "period", "target"],
        how="inner",
    )
    grid_rows = []
    selected_metrics = []
    selected_predictions = []
    for clock in CLOCKS:
        clock_frame = merged[merged["clock"].eq(clock)]
        for keys, block in clock_frame.groupby(["variant", "view", "c_value"]):
            variant, view, c_value = keys
            dev = block[block["period"].eq("development")]
            for weight in np.linspace(0, 0.5, 11):
                probability = (
                    (1 - weight) * dev["baseline_probability"].to_numpy()
                    + weight * dev["v13_probability"].to_numpy()
                )
                auc = float(roc_auc_score(dev["target"], probability))
                brier = float(brier_score_loss(dev["target"], probability))
                grid_rows.append({
                    "clock": clock, "variant": variant, "view": view,
                    "c_value": c_value, "v13_weight": float(weight),
                    "development_auc": auc, "development_brier": brier,
                    "selection_score": auc - 0.20 * brier,
                })
        grid = pd.DataFrame([r for r in grid_rows if r["clock"] == clock])
        winner = grid.sort_values(
            ["selection_score", "development_auc", "v13_weight"],
            ascending=[False, False, True],
        ).iloc[0]
        chosen = clock_frame[
            clock_frame["variant"].eq(winner["variant"])
            & clock_frame["view"].eq(winner["view"])
            & clock_frame["c_value"].eq(winner["c_value"])
        ].copy()
        chosen["probability"] = (
            (1 - winner["v13_weight"]) * chosen["baseline_probability"]
            + winner["v13_weight"] * chosen["v13_probability"]
        )
        late_pass = False
        for period, part in chosen.groupby("period"):
            auc = float(roc_auc_score(part["target"], part["probability"]))
            brier = float(brier_score_loss(part["target"], part["probability"]))
            base_auc = float(roc_auc_score(part["target"], part["baseline_probability"]))
            base_brier = float(brier_score_loss(part["target"], part["baseline_probability"]))
            boot = grouped_auc_bootstrap(part.reset_index(drop=True)) if period == "late" else {}
            if period == "late":
                late_pass = bool(
                    winner["v13_weight"] > 0
                    and auc > base_auc
                    and brier <= base_brier
                    and boot["p025"] > 0
                )
            selected_metrics.append({
                "clock": clock, "variant": winner["variant"], "view": winner["view"],
                "c_value": float(winner["c_value"]), "v13_weight": float(winner["v13_weight"]),
                "period": period, "auc": auc, "brier": brier,
                "baseline_auc": base_auc, "baseline_brier": base_brier,
                **boot,
            })
        chosen["v13_promoted"] = late_pass
        if not late_pass:
            chosen["probability"] = chosen["baseline_probability"]
        selected_predictions.append(chosen)
        for row in selected_metrics:
            if row["clock"] == clock:
                row["promotion_pass"] = late_pass
    return pd.DataFrame(grid_rows), pd.DataFrame(selected_metrics), pd.concat(selected_predictions, ignore_index=True)


def select_point(point: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows = []
    selected_metrics = []
    selected_predictions = []
    for clock in CLOCKS:
        block = point[point["clock"].eq(clock)]
        dev = block[block["period"].eq("development")]
        anchor_once = dev.drop_duplicates(["district_id", "season_start_year"])
        anchor_rmse = float(mean_squared_error(anchor_once["actual"], anchor_once["anchor"]) ** 0.5)
        for keys, candidate in dev.groupby(["variant", "view", "head", "alpha"]):
            variant, view, head, alpha = keys
            for weight in (0.0, 0.10, 0.25, 0.50, 0.75, 1.0):
                prediction = (1 - weight) * candidate["anchor"] + weight * candidate["raw_prediction"]
                temp = candidate.assign(prediction=prediction)
                rmse = float(mean_squared_error(temp["actual"], prediction) ** 0.5)
                years = sum(
                    mean_squared_error(part["actual"], part["prediction"]) ** 0.5
                    < mean_squared_error(part["actual"], part["anchor"]) ** 0.5
                    for _, part in temp.groupby("season_start_year")
                )
                cells = sum(
                    mean_squared_error(part["actual"], part["prediction"]) ** 0.5
                    < mean_squared_error(part["actual"], part["anchor"]) ** 0.5
                    for _, part in temp.groupby(["state_name", "season_start_year"])
                )
                grid_rows.append({
                    "clock": clock, "variant": variant, "view": view, "head": head,
                    "alpha": alpha, "candidate_weight": weight,
                    "development_anchor_rmse": anchor_rmse,
                    "development_model_rmse": rmse,
                    "development_years_improved": years,
                    "development_cells_improved": cells,
                    "eligible": bool(weight > 0 and rmse < anchor_rmse and years == 2 and cells >= 4),
                })
        grid = pd.DataFrame([r for r in grid_rows if r["clock"] == clock])
        eligible = grid[grid["eligible"]]
        winner = (
            eligible.sort_values(["development_model_rmse", "candidate_weight"]).iloc[0]
            if len(eligible)
            else grid[grid["candidate_weight"].eq(0)].iloc[0]
        )
        chosen = block[
            block["variant"].eq(winner["variant"])
            & block["view"].eq(winner["view"])
            & block["head"].eq(winner["head"])
            & block["alpha"].eq(winner["alpha"])
        ].copy()
        chosen["prediction"] = (
            (1 - winner["candidate_weight"]) * chosen["anchor"]
            + winner["candidate_weight"] * chosen["raw_prediction"]
        )
        late = chosen[chosen["period"].eq("late")].reset_index(drop=True)
        late_anchor_rmse = float(mean_squared_error(late["actual"], late["anchor"]) ** 0.5)
        late_model_rmse = float(mean_squared_error(late["actual"], late["prediction"]) ** 0.5)
        late_years = sum(
            mean_squared_error(part["actual"], part["prediction"]) ** 0.5
            < mean_squared_error(part["actual"], part["anchor"]) ** 0.5
            for _, part in late.groupby("season_start_year")
        )
        late_cells = sum(
            mean_squared_error(part["actual"], part["prediction"]) ** 0.5
            < mean_squared_error(part["actual"], part["anchor"]) ** 0.5
            for _, part in late.groupby(["state_name", "season_start_year"])
        )
        boot = grouped_rmse_bootstrap(late)
        promote = bool(
            winner["candidate_weight"] > 0
            and late_model_rmse < late_anchor_rmse
            and late_years == 2
            and boot["p025"] > 0
        )
        selected_metrics.append({
            **winner.to_dict(),
            "late_anchor_rmse": late_anchor_rmse,
            "late_model_rmse": late_model_rmse,
            "late_years_improved": late_years,
            "late_cells_improved": late_cells,
            **boot,
            "promotion_pass": promote,
        })
        chosen["v13_promoted"] = promote
        if not promote:
            chosen["prediction"] = chosen["anchor"]
        selected_predictions.append(chosen)
    return pd.DataFrame(grid_rows), pd.DataFrame(selected_metrics), pd.concat(selected_predictions, ignore_index=True)


def severe_baseline() -> pd.DataFrame:
    pieces = []
    for clocks, path in [
        (["jan15", "feb15"], V11 / "artifacts" / "global_direction_only" / "blended_predictions.parquet"),
        (["mar05"], V11 / "artifacts" / "direction_sidecar" / "blended_predictions.parquet"),
    ]:
        frame = pd.read_parquet(path)
        frame = frame[
            frame["clock"].isin(clocks)
            & frame["task"].eq("severe_decline")
            & frame["selection_category"].eq("global_required")
        ].copy()
        frame["period"] = frame["period"].replace({
            "development_2019_2020": "development",
            "reused_late_2021_2022": "late",
        })
        pieces.append(frame[[
            "district_id", "season_start_year", "clock",
            "target__severe_drop", "blended_probability",
        ]])
    return pd.concat(pieces).drop_duplicates(["district_id", "season_start_year", "clock"]).rename(
        columns={
            "target__severe_drop": "severe_target",
            "blended_probability": "severe_probability",
        }
    )


def save_deployment_bundle(
    data: dict[str, np.ndarray],
    meta: pd.DataFrame,
    transitions: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    point_metrics: pd.DataFrame,
) -> dict[str, object]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weather_weights, sequence_scale, weather_audit = v12lab.pretrain_weather(2022)
    required = {"full"}
    for metrics in (direction_metrics, point_metrics):
        for variant in metrics.loc[metrics["promotion_pass"].eq(True), "variant"].tolist():
            required.add(str(variant))
    deployment_representations: dict[str, list[np.ndarray]] = {variant: [] for variant in required}
    response_paths = []
    for variant in sorted(required):
        for seed in SEEDS:
            model, scale, loss = train_response(
                data, transitions, 2022, variant, seed, weather_weights, sequence_scale
            )
            _, representation = response_predict(model, data, scale, np.arange(len(meta)))
            deployment_representations[variant].append(representation)
            path = MODEL_DIR / f"crop_response_{variant}_seed{seed}_through2022.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "variant": variant,
                "seed": seed,
                "train_end": 2022,
                "epochs": EPOCHS,
                "dynamic_columns": DYNAMIC_COLUMNS.tolist(),
                "response_scale": {k: v for k, v in asdict(scale).items()},
                "architecture": "CropResponseNet",
                "uses_yield_labels": False,
            }, path)
            response_paths.append(str(path.relative_to(V13)))
    averaged = {
        variant: np.mean(values, axis=0).astype(np.float32)
        for variant, values in deployment_representations.items()
    }
    saved_heads = []
    all_rows = np.ones(len(meta), dtype=bool)
    for clock in CLOCKS:
        clock_rows = meta["clock"].eq(clock).to_numpy()
        dm = direction_metrics[
            direction_metrics["clock"].eq(clock) & direction_metrics["period"].eq("late")
        ].iloc[0]
        if bool(dm["promotion_pass"]):
            features = feature_view(averaged[dm["variant"]], data["tabular"], dm["view"])
            model = make_pipeline("direction", float(dm["c_value"]))
            model.fit(features[clock_rows], data["increase"][clock_rows].astype(int))
            path = MODEL_DIR / f"direction_v13_{clock}_through2022.joblib"
            joblib.dump(model, path)
            saved_heads.append(str(path.relative_to(V13)))
        pm = point_metrics[point_metrics["clock"].eq(clock)].iloc[0]
        if bool(pm["promotion_pass"]):
            features = feature_view(averaged[pm["variant"]], data["tabular"], pm["view"])
            if pm["head"] == "direct":
                target = data["yield_kg_per_ha"]
            elif pm["head"] == "change":
                target = data["yield_kg_per_ha"] / data["lag_1_yield"] - 1
            else:
                target = data["yield_kg_per_ha"] / data["anchor_prediction"] - 1
            model = make_pipeline("point", float(pm["alpha"]))
            model.fit(features[clock_rows], target[clock_rows])
            path = MODEL_DIR / f"point_v13_{clock}_through2022.joblib"
            joblib.dump(model, path)
            saved_heads.append(str(path.relative_to(V13)))
    return {
        "weather_pretraining": weather_audit,
        "response_models": response_paths,
        "promoted_downstream_heads": saved_heads,
        "deployment_fit_years": [2017, 2018, 2019, 2020, 2021, 2022],
        "deployment_fit_score_claimed": False,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data, meta, transitions = load_inputs()
    transition_manifest = {
        "rows": len(transitions),
        "districts": int(transitions["district_id"].nunique()),
        "years": sorted(transitions["season_start_year"].unique().astype(int).tolist()),
        "transitions": ["jan15->feb15", "feb15->mar05"],
        "dynamic_satellite_columns": DYNAMIC_COLUMNS.tolist(),
        "invalid_psri_cells_removed": int(data["_invalid_psri_removed"]),
        "invalid_psri_rule": "abs(PSRI) > 2 is treated as missing",
        "future_satellite_used_as_input": False,
        "yield_used_in_response_pretraining": False,
    }
    (OUT / "transition_manifest.json").write_text(json.dumps(transition_manifest, indent=2))

    representations: dict[tuple[int, str], np.ndarray] = {}
    prediction_store: dict[tuple[int, str], np.ndarray] = {}
    scales: dict[int, ResponseScale] = {}
    training_rows = []
    for train_end in sorted(set(fold[0] for fold in FOLDS)):
        weather_weights, sequence_scale, weather_audit = v12lab.pretrain_weather(train_end)
        variant_representations: dict[str, list[np.ndarray]] = {v: [] for v in VARIANTS}
        variant_predictions: dict[str, list[np.ndarray]] = {v: [] for v in VARIANTS}
        for variant in VARIANTS:
            for seed in SEEDS:
                model, scale, loss = train_response(
                    data, transitions, train_end, variant, seed, weather_weights, sequence_scale
                )
                prediction, representation = response_predict(
                    model, data, scale, np.arange(len(meta))
                )
                variant_predictions[variant].append(prediction)
                variant_representations[variant].append(representation)
                training_rows.append({
                    "train_end": train_end,
                    "variant": variant,
                    "seed": seed,
                    "transition_rows": int(transitions["season_start_year"].le(train_end).sum()),
                    "epochs": EPOCHS,
                    "final_loss": loss,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "weather_pretrain_rows": weather_audit["rows"],
                    "weather_pretrain_final_loss": weather_audit["final_scaled_huber"],
                    "device": str(DEVICE),
                })
                scales[train_end] = scale
        for variant in VARIANTS:
            prediction_store[(train_end, variant)] = np.mean(
                variant_predictions[variant], axis=0
            ).astype(np.float32)
            representations[(train_end, variant)] = np.mean(
                variant_representations[variant], axis=0
            ).astype(np.float32)
    pd.DataFrame(training_rows).to_csv(OUT / "response_training_audit.csv", index=False)

    trajectory_samples, trajectory_metrics = evaluate_trajectory(
        data, transitions, prediction_store, scales
    )
    trajectory_samples.to_parquet(OUT / "trajectory_predictions.parquet", index=False)
    trajectory_metrics.to_csv(OUT / "trajectory_metrics.csv", index=False)
    trajectory_bootstrap = trajectory_uncertainty(trajectory_samples)
    trajectory_bootstrap.to_csv(OUT / "trajectory_uncertainty.csv", index=False)

    direction_raw, point_raw, _ = fit_downstream_folds(data, meta, representations)
    direction_raw.to_parquet(OUT / "direction_candidate_predictions.parquet", index=False)
    point_raw.to_parquet(OUT / "point_candidate_predictions.parquet", index=False)

    direction_grid, direction_metrics, direction_selected = select_direction(direction_raw)
    direction_grid.to_csv(OUT / "direction_development_grid.csv", index=False)
    direction_metrics.to_csv(OUT / "direction_selected_metrics.csv", index=False)
    direction_selected.to_parquet(OUT / "direction_selected_predictions.parquet", index=False)

    point_grid, point_metrics, point_selected = select_point(point_raw)
    point_grid.to_csv(OUT / "point_development_grid.csv", index=False)
    point_metrics.to_csv(OUT / "point_selected_metrics.csv", index=False)
    point_selected.to_parquet(OUT / "point_selected_predictions.parquet", index=False)

    final = point_selected[[
        "district_id", "state_name", "season_start_year", "clock", "period",
        "actual", "anchor", "prediction", "v13_promoted",
    ]].rename(columns={"v13_promoted": "point_v13_promoted"})
    final = final.merge(
        direction_selected[[
            "district_id", "season_start_year", "clock", "target",
            "baseline_probability", "probability", "v13_promoted",
        ]].rename(columns={
            "target": "increase_target",
            "baseline_probability": "increase_baseline_probability",
            "probability": "increase_probability",
            "v13_promoted": "direction_v13_promoted",
        }),
        on=["district_id", "season_start_year", "clock"],
        how="left",
    )
    final = final.merge(
        severe_baseline(),
        on=["district_id", "season_start_year", "clock"],
        how="left",
    )
    final.to_parquet(OUT / "final_predictions.parquet", index=False)

    trajectory_lookup = trajectory_metrics.set_index(["variant", "period"])
    full_dev = trajectory_lookup.loc[("full", "development")]
    full_late = trajectory_lookup.loc[("full", "late")]
    no_future_dev = trajectory_lookup.loc[("no_future", "development")]
    no_future_late = trajectory_lookup.loc[("no_future", "late")]
    trajectory_point_contract = bool(
        full_dev["transition_rmse"] < full_dev["persistence_rmse"]
        and full_dev["transition_rmse"] < no_future_dev["transition_rmse"]
        and full_late["transition_rmse"] < full_late["persistence_rmse"]
        and full_late["transition_rmse"] <= no_future_late["transition_rmse"]
    )
    late_future_uncertainty = trajectory_bootstrap[
        trajectory_bootstrap["period"].eq("late")
        & trajectory_bootstrap["comparison"].eq("no_future_minus_full")
    ].iloc[0]
    trajectory_promoted = bool(
        trajectory_point_contract and late_future_uncertainty["p025"] > 0
    )
    deployment = save_deployment_bundle(
        data, meta, transitions, direction_metrics, point_metrics
    )
    policy = {
        "name": "V13 final clock-specific wheat-yield policy",
        "point": {
            clock: (
                "V13 crop-response correction"
                if bool(point_metrics.loc[point_metrics["clock"].eq(clock), "promotion_pass"].iloc[0])
                else ("V7 locked point forecast" if clock in ("jan15", "feb15") else "V5 locked point forecast")
            )
            for clock in CLOCKS
        },
        "increase_probability": {
            clock: (
                "V13 crop-response blend"
                if bool(direction_metrics.loc[
                    direction_metrics["clock"].eq(clock)
                    & direction_metrics["period"].eq("late"),
                    "promotion_pass",
                ].iloc[0])
                else (
                    "V11 global direction model" if clock == "jan15"
                    else "V12 crop-state blend over V11"
                )
            )
            for clock in CLOCKS
        },
        "severe_decline_probability": "V11 global-required severe-decline sidecar",
        "future_weather_crop_trajectory_promoted": trajectory_promoted,
        "future_weather_crop_trajectory_point_contract_pass": trajectory_point_contract,
        "future_weather_crop_trajectory_status": (
            "strictly promoted"
            if trajectory_promoted
            else "promising point estimate; future-weather increment is not uncertainty-stable"
        ),
        "selection_years": [2019, 2020],
        "reused_confirmation_years": [2021, 2022],
        "post_2022_yield_labels_read": False,
        "deployment": deployment,
    }
    (OUT / "final_policy.json").write_text(json.dumps(policy, indent=2, default=str))
    print("\nTRAJECTORY\n", trajectory_metrics.to_string(index=False))
    print("\nDIRECTION\n", direction_metrics.to_string(index=False))
    print("\nPOINT\n", point_metrics.to_string(index=False))
    print("\nPOLICY\n", json.dumps(policy, indent=2, default=str))


if __name__ == "__main__":
    main()
