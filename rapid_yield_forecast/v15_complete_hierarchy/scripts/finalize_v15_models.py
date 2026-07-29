#!/usr/bin/env python3
"""Refit the promoted V15 components through 2022 for deployment.

These refits are never used for the reported 2019-2022 scores.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


V15 = Path(__file__).resolve().parents[1]
ROOT = V15.parents[1]
sys.path.insert(0, str(ROOT))
DATA = V15 / "data"
ARTIFACTS = V15 / "artifacts"
MODELS = V15 / "models"

from rapid_yield_forecast.v15_complete_hierarchy.scripts import train_v15_encoder as enc  # noqa: E402
from rapid_yield_forecast.v14_anomaly_distribution.scripts import run_v14_lab as lab  # noqa: E402


def fit_xgb_bundle(
    frame: pd.DataFrame,
    features: list[str],
    name: str,
) -> dict[str, object]:
    usable = lab.finite_columns(frame, features, minimum=0.25)
    x, columns = lab.design(frame, usable)
    target = (
        frame[lab.TARGET].to_numpy(float)
        - frame["baseline_weighted_recent"].to_numpy(float)
    )
    models = []
    for seed in lab.SEEDS:
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBRegressor(
                n_estimators=350, max_depth=2, learning_rate=0.025,
                min_child_weight=25, subsample=0.85, colsample_bytree=0.65,
                reg_lambda=50.0, reg_alpha=5.0,
                objective="reg:squarederror", tree_method="hist",
                n_jobs=1, random_state=seed,
            ),
        )
        model.fit(x, target)
        models.append(model)
    bundle = {
        "name": name,
        "models": models,
        "numeric_features": usable,
        "design_columns": columns,
        "depth": 2,
        "seeds": list(lab.SEEDS),
        "train_year_min": 2017,
        "train_year_max": 2022,
        "target": "yield_kg_per_ha - baseline_weighted_recent",
        "score_claimed_for_refit": False,
    }
    joblib.dump(bundle, MODELS / f"{name}.joblib")
    return {
        key: value for key, value in bundle.items()
        if key not in {"models", "design_columns", "numeric_features"}
    } | {
        "numeric_feature_count": len(usable),
        "design_column_count": len(columns),
    }


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    representation_path = DATA / "deployment_encoder_features_through2022.parquet"
    encoder_paths = [
        MODELS / f"encoder_modis_pretrained_seed{seed}_through2022_deployment.pt"
        for seed in enc.SEEDS
    ]
    encoder_audits = []
    if representation_path.exists() and all(path.exists() for path in encoder_paths):
        representations = pd.read_parquet(representation_path)
        for path in encoder_paths:
            packed = torch.load(path, map_location="cpu", weights_only=False)
            encoder_audits.append({
                "train_end": int(packed["train_end"]),
                "variant": packed["variant"],
                "seed": int(packed["seed"]),
                "device": "mps",
                "resumed_from_completed_encoder_refit": True,
                "score_claimed_for_refit": False,
            })
    else:
        modis, modis_mask, modis_meta, sentinel, meta = enc.load_data()
        sequence, sequence_mask = enc.sentinel_sequences(sentinel["crop"], meta)
        transition = enc.transitions(meta)
        models = []
        scales = []
        for seed in enc.SEEDS:
            model, scale, audit = enc.train_one_encoder(
                modis, modis_mask, modis_meta, sentinel,
                sequence, sequence_mask, transition,
                2022, None, "modis_pretrained", seed,
            )
            models.append(model)
            scales.append(scale)
            audit["score_claimed_for_refit"] = False
            encoder_audits.append(audit)
            torch.save({
                "state_dict": model.state_dict(),
                "scale": asdict(scale),
                "variant": "modis_pretrained",
                "seed": seed,
                "train_end": 2022,
                "modis_dim": modis.shape[2],
                "score_claimed_for_refit": False,
            }, MODELS / f"encoder_modis_pretrained_seed{seed}_through2022_deployment.pt")

        selected = meta[
            meta["season_start_year"].between(2017, 2022)
            & meta["clock"].eq("mar05")
        ].index.to_numpy()
        seed_features = []
        feature_columns = None
        for model, scale in zip(models, scales):
            arrays = enc.transform_rows(
                sentinel, sequence, sequence_mask, meta, selected, scale
            )
            output = enc.predict_outputs(model, arrays, scale)
            matrix, feature_columns = enc.compact_features(
                output, sentinel["crop"][selected]
            )
            seed_features.append(matrix)
        averaged = np.mean(seed_features, axis=0)
        representations = meta.loc[selected, [
            "district_id", "state_name", "district_name",
            "season_start_year", "clock",
        ]].reset_index(drop=True)
        encoded = pd.DataFrame(
            averaged,
            columns=[f"enc__{column}" for column in feature_columns],
        )
        representations = pd.concat([representations, encoded], axis=1)
        representations["representation_train_end"] = 2022
        representations["feature_role"] = "deployment_refit"
        representations.to_parquet(representation_path, index=False)

    base, groups, _ = lab.load_panel()
    training = base[base["season_start_year"].between(2017, 2022)].merge(
        representations.drop(columns=[
            "state_name", "district_name", "clock",
            "representation_train_end", "feature_role",
        ]),
        on=["district_id", "season_start_year"], validate="one_to_one",
    )
    current = [
        column for column in representations
        if column.startswith("enc__")
        and (
            "current_index_" in column
            or "no_future_delta_" in column
            or "no_future_fused_pool_" in column
        )
    ]
    xgb_audits = [
        fit_xgb_bundle(
            training, groups["physical"],
            "v15_xgb_base_physical_d2_through2022",
        ),
        fit_xgb_bundle(
            training, groups["physical"] + current,
            "v15_xgb_current_physical_d2_through2022",
        ),
    ]
    recipe = {
        "production_anchor": (
            "../v14_anomaly_distribution/models/"
            "outlook_shadow_xgb_bundle.joblib"
        ),
        "v15_base_bundle": "v15_xgb_base_physical_d2_through2022.joblib",
        "v15_current_bundle": "v15_xgb_current_physical_d2_through2022.joblib",
        "point_formula": (
            "V14_shadow + 1.25 * "
            "(V15_current_xgb - V15_base_xgb)"
        ),
        "gamma": 1.25,
        "distribution_scale": 0.95,
        "distribution_source": "V14 empirical district residual distribution",
        "encoder_seeds": list(enc.SEEDS),
        "encoder_train_end": 2022,
        "xgb_train_years": [2017, 2018, 2019, 2020, 2021, 2022],
        "reported_scores_use_deployment_refit": False,
        "score_claimed_for_refit": False,
    }
    with (MODELS / "v15_deployment_recipe.json").open("w") as handle:
        json.dump(recipe, handle, indent=2)
    summary = {
        "encoder": encoder_audits,
        "xgb": xgb_audits,
        "recipe": recipe,
        "post_2022_yield_labels_read": False,
    }
    with (ARTIFACTS / "deployment_refit_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
