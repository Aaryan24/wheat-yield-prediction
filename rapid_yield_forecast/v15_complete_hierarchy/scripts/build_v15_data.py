#!/usr/bin/env python3
"""Build V15 long-yield and long-MODIS panels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


V15 = Path(__file__).resolve().parents[1]
ROOT = V15.parents[1]
RAPID = V15.parent
DATA = V15 / "data"

ICRISAT = RAPID / "v3" / "external_data_probe" / "icrisat_unapportioned_area_production_yield.json"
CROSSWALK = RAPID / "v5" / "agent_econ_irrigation" / "data" / "processed" / "icrisat_district_crosswalk_119.csv"
BASE = RAPID / "v3" / "data" / "feature_table_v3_extended_history_03-05.parquet"
MODIS = RAPID / "v5" / "agent_modis_history" / "data" / "modis_strict_history.csv"
V12_DATA = RAPID / "v12_cross_attention_yield" / "data" / "v12_dataset.npz"
V12_META = RAPID / "v12_cross_attention_yield" / "data" / "metadata.parquet"

CLOCKS = ("jan15", "feb15", "mar05")
MODIS_PREFIXES = (
    "mod09q1_ndvi",
    "mod09q1_nir",
    "mod09q1_red",
    "mod13q1_evi",
    "mod13q1_ndvi",
)
MODIS_SUFFIXES = (
    "last_valid_mean",
    "novdec_mean_mean",
    "recent48_mean_mean",
    "season_max_mean",
    "season_mean_mean",
    "season_temporal_sd_mean",
    "slope_per_day_mean",
)


def normalise_name(value: str) -> str:
    value = str(value).lower()
    value = value.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "", value)


def load_icrisat() -> pd.DataFrame:
    raw = json.loads(ICRISAT.read_text())
    headers = [item["header"] for item in raw["headers"]]
    wanted = [
        "Dist Code", "Year", "State Name", "Dist Name",
        "WHEAT AREA", "WHEAT PRODUCTION", "WHEAT YIELD",
    ]
    indices = [headers.index(column) for column in wanted]
    frame = pd.DataFrame(
        [[row[index] for index in indices] for row in raw["data"]],
        columns=[
            "icrisat_dist_code", "season_start_year", "icrisat_state_name",
            "icrisat_district_name", "wheat_area_1000ha",
            "wheat_production_1000t", "yield_kg_per_ha",
        ],
    )
    for column in [
        "season_start_year", "wheat_area_1000ha",
        "wheat_production_1000t", "yield_kg_per_ha",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["icrisat_dist_code"] = frame["icrisat_dist_code"].astype(str)
    return frame[
        frame["season_start_year"].between(1990, 2019)
        & frame["yield_kg_per_ha"].gt(0)
    ].copy()


def build_long_yield() -> tuple[pd.DataFrame, pd.DataFrame]:
    crosswalk = pd.read_csv(CROSSWALK)
    crosswalk["icrisat_dist_code"] = (
        crosswalk["icrisat_dist_code"].astype(int).astype(str)
    )
    icrisat = load_icrisat()
    mapped = crosswalk[[
        "district_id", "state_name", "district_name", "icrisat_dist_code"
    ]].merge(
        icrisat[[
            "icrisat_dist_code", "season_start_year",
            "yield_kg_per_ha", "wheat_area_1000ha",
        ]],
        on="icrisat_dist_code",
        how="left",
        validate="one_to_many",
    )
    mapped["source"] = "ICRISAT_DLD_unapportioned"
    mapped["proxy_history"] = False

    base = pd.read_parquet(BASE)
    official = base[[
        "district_id", "state_name", "district_name", "season_start_year",
        "yield_kg_per_ha", "area_ha", "proxy_filled",
        "proxy_source_district",
    ]].copy()
    official["wheat_area_1000ha"] = official["area_ha"] / 1000.0
    official["source"] = "DES_official_model_ready"
    official["proxy_history"] = official["proxy_filled"].fillna(False)

    # ICRISAT and the official panel overlap almost exactly from 2010 onward.
    # Use ICRISAT only before 2010 and the official series from 2010 onward.
    older = mapped[mapped["season_start_year"].lt(2010)].copy()

    # For the six explicitly documented post-split districts, extend older
    # history with the already audited parent district proxy.  Estimate a
    # multiplicative level ratio from years where both current and parent
    # ICRISAT records exist, then apply it only to missing pre-2010 years.
    proxy_map = (
        official.loc[
            official["proxy_source_district"].notna(),
            ["district_id", "state_name", "proxy_source_district"],
        ]
        .drop_duplicates("district_id")
    )
    name_lookup = {
        (row.state_name, normalise_name(row.district_name)): row.district_id
        for row in crosswalk.itertuples(index=False)
    }
    full_icrisat = mapped.copy()
    additions = []
    for row in proxy_map.itertuples(index=False):
        parent_id = name_lookup.get((
            row.state_name, normalise_name(row.proxy_source_district)
        ))
        if parent_id is None:
            continue
        target = full_icrisat[
            full_icrisat["district_id"].eq(row.district_id)
        ][["season_start_year", "yield_kg_per_ha"]].rename(
            columns={"yield_kg_per_ha": "target_yield"}
        )
        parent = full_icrisat[
            full_icrisat["district_id"].eq(parent_id)
        ][[
            "season_start_year", "yield_kg_per_ha",
            "wheat_area_1000ha",
        ]].rename(columns={"yield_kg_per_ha": "parent_yield"})
        overlap = target.merge(parent, on="season_start_year")
        valid = overlap[
            overlap["target_yield"].gt(0) & overlap["parent_yield"].gt(0)
        ]
        ratio = float(np.median(
            valid["target_yield"] / valid["parent_yield"]
        )) if len(valid) >= 2 else 1.0
        ratio = float(np.clip(ratio, 0.70, 1.30))
        existing_years = set(target["season_start_year"])
        missing = parent[
            parent["season_start_year"].lt(2010)
            & ~parent["season_start_year"].isin(existing_years)
        ]
        target_meta = crosswalk[
            crosswalk["district_id"].eq(row.district_id)
        ].iloc[0]
        for source in missing.itertuples(index=False):
            additions.append({
                "district_id": row.district_id,
                "state_name": row.state_name,
                "district_name": target_meta["district_name"],
                "icrisat_dist_code": target_meta["icrisat_dist_code"],
                "season_start_year": int(source.season_start_year),
                "yield_kg_per_ha": float(source.parent_yield * ratio),
                "wheat_area_1000ha": float(source.wheat_area_1000ha),
                "source": f"ICRISAT_parent_proxy:{parent_id}",
                "proxy_history": True,
            })
    if additions:
        older = pd.concat([older, pd.DataFrame(additions)], ignore_index=True)

    older = older.sort_values(
        ["district_id", "season_start_year", "proxy_history"]
    ).drop_duplicates(["district_id", "season_start_year"], keep="first")
    current = official[[
        "district_id", "state_name", "district_name", "season_start_year",
        "yield_kg_per_ha", "wheat_area_1000ha", "source", "proxy_history",
    ]]
    columns = current.columns.tolist()
    long = pd.concat(
        [older.reindex(columns=columns), current],
        ignore_index=True,
    ).sort_values(["district_id", "season_start_year"]).reset_index(drop=True)

    # Source-overlap audit.
    official_overlap = official[
        official["season_start_year"].le(2019)
    ][["district_id", "state_name", "season_start_year", "yield_kg_per_ha"]]
    overlap = official_overlap.merge(
        mapped[[
            "district_id", "season_start_year", "yield_kg_per_ha"
        ]].rename(columns={"yield_kg_per_ha": "icrisat_yield"}),
        on=["district_id", "season_start_year"],
        validate="one_to_one",
    )
    overlap["difference"] = (
        overlap["icrisat_yield"] - overlap["yield_kg_per_ha"]
    )
    overlap["abs_difference"] = overlap["difference"].abs()
    return long, overlap


def build_modis_sequences() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    raw = pd.read_csv(MODIS)
    numeric = [
        column for column in raw.columns
        if column.startswith(MODIS_PREFIXES)
    ]
    combined = raw.groupby(
        [
            "district_id", "state_name", "district_name",
            "season_start_year", "cutoff",
        ],
        as_index=False,
    )[numeric].first()
    features = [
        f"{prefix}_{suffix}"
        for prefix in MODIS_PREFIXES
        for suffix in MODIS_SUFFIXES
        if f"{prefix}_{suffix}" in combined
    ]
    districts = (
        combined[["district_id", "state_name", "district_name"]]
        .drop_duplicates()
        .sort_values("district_id")
    )
    years = list(range(2000, 2023))
    rows = []
    sequences = []
    masks = []
    lookup = {
        (row.district_id, int(row.season_start_year), row.cutoff): row
        for row in combined.itertuples(index=False)
    }
    for district in districts.itertuples(index=False):
        for year in years:
            tokens = []
            token_masks = []
            for clock in CLOCKS:
                row = lookup.get((district.district_id, year, clock))
                values = (
                    np.asarray([getattr(row, column) for column in features], float)
                    if row is not None else np.full(len(features), np.nan)
                )
                tokens.append(values)
                token_masks.append(bool(np.isfinite(values).any()))
            sequences.append(tokens)
            masks.append(token_masks)
            rows.append({
                "sample_id": len(rows),
                "district_id": district.district_id,
                "state_name": district.state_name,
                "district_name": district.district_name,
                "season_start_year": year,
                "district_group": (
                    sum(ord(character) for character in district.district_id) % 3
                ),
            })
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(masks, dtype=bool),
        pd.DataFrame(rows),
        features,
    )


def audit_sentinel() -> dict[str, object]:
    packed = np.load(V12_DATA)
    meta = pd.read_parquet(V12_META)
    crop = packed["crop"].copy()
    invalid_psri = np.abs(crop[:, 5, :]) > 2
    return {
        "rows": len(meta),
        "districts": int(meta["district_id"].nunique()),
        "years": sorted(meta["season_start_year"].unique().tolist()),
        "clocks": sorted(meta["clock"].unique().tolist()),
        "crop_shape": list(crop.shape),
        "experienced_weather_shape": list(packed["state"].shape),
        "future_weather_shape": list(packed["future"].shape),
        "invalid_psri_cells_to_mask": int(invalid_psri.sum()),
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    long, overlap = build_long_yield()
    long.to_parquet(DATA / "long_yield_1990_2022.parquet", index=False)
    overlap.to_csv(DATA / "icrisat_des_overlap.csv", index=False)
    modis, modis_mask, modis_meta, features = build_modis_sequences()
    np.savez_compressed(
        DATA / "modis_sequences_2000_2022.npz",
        sequence=modis,
        mask=modis_mask,
    )
    modis_meta.to_parquet(DATA / "modis_metadata.parquet", index=False)

    observed_per_district = (
        long[long["season_start_year"].lt(2019)]
        .groupby("district_id")["season_start_year"].nunique()
    )
    manifest = {
        "long_yield_rows": len(long),
        "long_yield_years": [
            int(long["season_start_year"].min()),
            int(long["season_start_year"].max()),
        ],
        "districts": int(long["district_id"].nunique()),
        "history_years_before_2019": {
            "minimum": int(observed_per_district.min()),
            "median": float(observed_per_district.median()),
            "maximum": int(observed_per_district.max()),
        },
        "icrisat_des_overlap": {
            "rows": len(overlap),
            "correlation": float(overlap[
                ["yield_kg_per_ha", "icrisat_yield"]
            ].corr().iloc[0, 1]),
            "mae_kg_per_ha": float(overlap["abs_difference"].mean()),
            "bias_icrisat_minus_des": float(overlap["difference"].mean()),
        },
        "modis_rows": len(modis_meta),
        "modis_shape": list(modis.shape),
        "modis_features": features,
        "modis_feature_count": len(features),
        "modis_years": [2000, 2022],
        "sentinel": audit_sentinel(),
        "post_2022_yield_labels_read": False,
    }
    (DATA / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
