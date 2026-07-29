#!/usr/bin/env python3
"""Build leakage-safe, fold-specific V13 future-crop outlook features for V14."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


V14 = Path(__file__).resolve().parents[1]
ROOT = V14.parents[1]
sys.path.insert(0, str(ROOT))
OUT = V14 / "data"

from rapid_yield_forecast.v13_crop_response_final.scripts import run_v13_final as v13  # noqa: E402


SEEDS = tuple(int(x) for x in os.environ.get("V14_OUTLOOK_SEEDS", "42,73").split(","))
GROUPS = int(os.environ.get("V14_OUTLOOK_GROUPS", "3"))
EPOCHS = int(os.environ.get("V14_OUTLOOK_EPOCHS", "60"))
DEVICE = torch.device(os.environ.get("V14_DEVICE", "cpu"))
VARIANTS = ("no_future", "full")
FOLD_TESTS = {2018: [2019], 2019: [2020], 2020: [2021, 2022]}


def compact_features(
    full: np.ndarray,
    no_future: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Turn 6 x 15 predicted crop changes into a compact, interpretable vector."""

    future_effect = full - no_future
    next_crop = current + full
    pieces = []
    names = []

    def add(values: np.ndarray, labels: list[str]) -> None:
        pieces.append(values.astype(np.float32))
        names.extend(labels)

    add(
        np.stack([
            np.nanmean(full, axis=(1, 2)),
            np.nanmean(np.abs(full), axis=(1, 2)),
            np.nanmean(full > 0, axis=(1, 2)),
        ], axis=1),
        ["outlook_full_delta_mean", "outlook_full_delta_abs_mean", "outlook_full_delta_positive_fraction"],
    )
    add(
        np.nanmean(full, axis=2),
        [f"outlook_full_index_{i}_delta_mean" for i in range(full.shape[1])],
    )
    add(
        np.nanmean(full, axis=1),
        [f"outlook_full_summary_{j}_delta_mean" for j in range(full.shape[2])],
    )
    add(
        np.nanmean(next_crop, axis=2),
        [f"outlook_full_index_{i}_next_mean" for i in range(full.shape[1])],
    )
    add(
        np.stack([
            np.nanmean(no_future, axis=(1, 2)),
            np.nanmean(np.abs(no_future), axis=(1, 2)),
            np.nanmean(no_future > 0, axis=(1, 2)),
        ], axis=1),
        ["outlook_no_future_delta_mean", "outlook_no_future_delta_abs_mean", "outlook_no_future_delta_positive_fraction"],
    )
    add(
        np.nanmean(no_future, axis=2),
        [f"outlook_no_future_index_{i}_delta_mean" for i in range(no_future.shape[1])],
    )
    add(
        np.stack([
            np.nanmean(future_effect, axis=(1, 2)),
            np.nanmean(np.abs(future_effect), axis=(1, 2)),
        ], axis=1),
        ["outlook_future_effect_mean", "outlook_future_effect_abs_mean"],
    )
    add(
        np.nanmean(future_effect, axis=2),
        [f"outlook_future_effect_index_{i}_mean" for i in range(future_effect.shape[1])],
    )
    add(
        np.nanmean(future_effect, axis=1),
        [f"outlook_future_effect_summary_{j}_mean" for j in range(future_effect.shape[2])],
    )
    result = np.concatenate(pieces, axis=1)
    if result.shape[1] != len(names):
        raise RuntimeError("Outlook feature name mismatch")
    return result, names


def train_variant_predictions(
    data: dict[str, np.ndarray],
    transitions: pd.DataFrame,
    train_end: int,
    variant: str,
    weather_weights: dict[str, torch.Tensor],
    sequence_scale,
    predict_indices: np.ndarray,
    audit_context: dict[str, object],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = []
    audit = []
    for seed in SEEDS:
        model, scale, loss = v13.train_response(
            data, transitions, train_end, variant, seed,
            weather_weights, sequence_scale,
        )
        prediction, _ = v13.response_predict(model, data, scale, predict_indices)
        predictions.append(prediction)
        audit.append({
            **audit_context,
            "train_end": train_end,
            "variant": variant,
            "seed": seed,
            "epochs": EPOCHS,
            "transition_rows": int(transitions["season_start_year"].le(train_end).sum()),
            "final_loss": loss,
            "parameters": sum(p.numel() for p in model.parameters()),
        })
    return np.mean(predictions, axis=0).astype(np.float32), audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v13.DEVICE = DEVICE
    v13.v12lab.DEVICE = DEVICE
    v13.EPOCHS = EPOCHS
    data, meta, transitions = v13.load_inputs()
    district_order = sorted(meta["district_id"].unique())
    district_group = {
        district: index % GROUPS
        for index, district in enumerate(district_order)
    }
    years = meta["season_start_year"].to_numpy()
    all_rows = []
    audits = []
    feature_names: list[str] | None = None

    for train_end, test_years in FOLD_TESTS.items():
        weather_weights, sequence_scale, weather_audit = v13.v12lab.pretrain_weather(train_end)
        full_test_predictions: dict[str, np.ndarray] = {}
        all_indices = np.arange(len(meta))
        for variant in VARIANTS:
            prediction, audit = train_variant_predictions(
                data, transitions, train_end, variant,
                weather_weights, sequence_scale, all_indices,
                {"feature_role": "test_full", "held_group": -1},
            )
            full_test_predictions[variant] = prediction
            audits.extend(audit)

        crossfit_predictions = {
            variant: np.full((len(meta), 6, len(v13.DYNAMIC_COLUMNS)), np.nan, np.float32)
            for variant in VARIANTS
        }
        for group in range(GROUPS):
            held_districts = {
                district for district, assigned in district_group.items()
                if assigned == group
            }
            held_mask = meta["district_id"].isin(held_districts).to_numpy() & (years <= train_end)
            held_indices = np.where(held_mask)[0]
            training_transitions = transitions[
                ~transitions["district_id"].isin(held_districts)
            ].copy()
            for variant in VARIANTS:
                prediction, audit = train_variant_predictions(
                    data, training_transitions, train_end, variant,
                    weather_weights, sequence_scale, held_indices,
                    {"feature_role": "train_district_crossfit", "held_group": group},
                )
                crossfit_predictions[variant][held_indices] = prediction
                audits.extend(audit)

        selected_indices = np.where(
            (years <= train_end) | np.isin(years, test_years)
        )[0]
        train_mask = years[selected_indices] <= train_end
        full = np.empty((len(selected_indices), 6, len(v13.DYNAMIC_COLUMNS)), np.float32)
        no_future = np.empty_like(full)
        full[train_mask] = crossfit_predictions["full"][selected_indices[train_mask]]
        no_future[train_mask] = crossfit_predictions["no_future"][selected_indices[train_mask]]
        full[~train_mask] = full_test_predictions["full"][selected_indices[~train_mask]]
        no_future[~train_mask] = full_test_predictions["no_future"][selected_indices[~train_mask]]
        if not np.isfinite(full).all() or not np.isfinite(no_future).all():
            raise RuntimeError(f"Missing cross-fitted outlook predictions for cutoff {train_end}")

        current = data["crop"][selected_indices][:, :, v13.DYNAMIC_COLUMNS]
        features, names = compact_features(full, no_future, current)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("Outlook feature order changed")
        base = meta.loc[selected_indices, [
            "district_id", "state_name", "district_name",
            "season_start_year", "clock",
        ]].reset_index(drop=True)
        base["representation_train_end"] = train_end
        base["feature_role"] = np.where(train_mask, "train_crossfit", "test_full")
        feature_frame = pd.DataFrame(features, columns=feature_names)
        all_rows.append(pd.concat([base, feature_frame], axis=1))
        audits.append({
            "feature_role": "weather_pretraining",
            "held_group": -1,
            "train_end": train_end,
            "variant": "weather",
            "seed": 2026,
            "epochs": weather_audit["epochs"],
            "transition_rows": weather_audit["rows"],
            "final_loss": weather_audit["final_scaled_huber"],
            "parameters": np.nan,
        })

    result = pd.concat(all_rows, ignore_index=True)
    keys = ["district_id", "season_start_year", "clock", "representation_train_end"]
    if result.duplicated(keys).any():
        raise RuntimeError("Duplicate V14 outlook rows")
    result.to_parquet(OUT / "strict_outlook_features.parquet", index=False)
    pd.DataFrame(audits).to_csv(OUT / "outlook_training_audit.csv", index=False)
    manifest = {
        "rows": len(result),
        "feature_count": len(feature_names or []),
        "features": feature_names,
        "fold_tests": FOLD_TESTS,
        "district_crossfit_groups": GROUPS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "device": str(DEVICE),
        "yield_labels_used": False,
        "later_satellite_used_as_input": False,
        "training_feature_rule": "exclude the row district's crossfit group",
        "invalid_psri_cells_removed": int(data["_invalid_psri_removed"]),
        "post_2022_yield_labels_read": False,
    }
    (OUT / "outlook_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in [
        "rows", "feature_count", "district_crossfit_groups", "seeds", "epochs", "device"
    ]}, indent=2))


if __name__ == "__main__":
    main()

