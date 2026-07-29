#!/usr/bin/env python3
"""Turn ~44 tiles per district into features a district mean cannot express.

Two real district-seasons from the extracted data:

    district-average NDVI 0.455, tiles spanning 0.418 to 0.489
    district-average NDVI 0.457, tiles spanning 0.156 to 0.579

Every model built so far sees these as the same input.  The first is uniformly
mediocre; the second has a large area of essentially failed crop next to healthy
ground.  They will not produce the same harvest.

Two families of feature are built:

  SHAPE     the spread of tile values within a district-season -- how unequal
            the district is right now.

  ANOMALY   each tile compared against ITS OWN history across seasons, then
            aggregated.  This is the one that should matter: "what fraction of
            this district's area is doing worse than that ground normally
            does" is a damage measure, and it is invisible to a district mean
            because a collapse in one corner is diluted by health elsewhere.

The tile grid is generated from fixed district geometry at a fixed scale, so
tile_index is stable across seasons -- verified: all 119 districts hold a
constant tile count across all 23 seasons.  That stability is what makes
per-tile histories legitimate.

All histories are expanding and shifted by one season, so no feature uses the
target season or anything after it.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
DATA = V16 / "data"
ARTIFACTS = V16 / "artifacts"

INDICES = ["ndvi_season_mean", "ndvi_recent_mean", "ndvi_season_max",
           "ndvi_season_sd", "evi_season_mean", "ndwi_recent_mean"]
KEYS = ["district_id", "season_start_year"]


def load_tiles() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA / "tiles" / "*_mar05.parquet")))
    if not files:
        raise SystemExit("no tile files; run extract_v16_tiles.py first")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return frame.sort_values(["district_id", "tile_index", "season_start_year"])


def shape_features(tiles: pd.DataFrame) -> pd.DataFrame:
    """How unequal is this district right now?"""
    out = []
    for index in INDICES:
        grouped = tiles.groupby(KEYS)[index]
        block = grouped.agg(
            mean="mean", sd="std", p10=lambda s: s.quantile(0.10),
            p25=lambda s: s.quantile(0.25), p50="median",
            p75=lambda s: s.quantile(0.75), p90=lambda s: s.quantile(0.90),
            lo="min", hi="max").add_prefix(f"tile_{index}__")
        out.append(block)
    frame = pd.concat(out, axis=1).reset_index()
    for index in INDICES:
        mean = frame[f"tile_{index}__mean"]
        frame[f"tile_{index}__cv"] = frame[f"tile_{index}__sd"] / mean.replace(0, np.nan)
        frame[f"tile_{index}__iqr"] = (frame[f"tile_{index}__p75"]
                                       - frame[f"tile_{index}__p25"])
        # how far the weak tail sits below the middle: a one-sided damage measure
        frame[f"tile_{index}__lower_gap"] = (frame[f"tile_{index}__p50"]
                                             - frame[f"tile_{index}__p10"])
        frame[f"tile_{index}__upper_gap"] = (frame[f"tile_{index}__p90"]
                                             - frame[f"tile_{index}__p50"])
        frame[f"tile_{index}__skew_ratio"] = (
            frame[f"tile_{index}__lower_gap"]
            / frame[f"tile_{index}__upper_gap"].replace(0, np.nan))
    return frame


def anomaly_features(tiles: pd.DataFrame) -> pd.DataFrame:
    """Each tile against its own past, then summarised over the district."""
    frame = tiles.copy()
    grouped = frame.groupby(["district_id", "tile_index"])
    for index in INDICES:
        prior_mean = grouped[index].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean())
        prior_sd = grouped[index].transform(
            lambda s: s.shift(1).expanding(min_periods=4).std())
        frame[f"z_{index}"] = (frame[index] - prior_mean) / prior_sd.replace(0, np.nan)

    out = []
    for index in INDICES:
        column = f"z_{index}"
        grouped_ds = frame.groupby(KEYS)[column]
        block = grouped_ds.agg(
            mean="mean", sd="std",
            worst_decile=lambda s: s.quantile(0.10),
            best_decile=lambda s: s.quantile(0.90),
            # share of the district's AREA that is below its own normal
            share_below_0=lambda s: float((s < 0).mean()),
            share_below_1sd=lambda s: float((s < -1).mean()),
            share_above_1sd=lambda s: float((s > 1).mean()),
        ).add_prefix(f"tanom_{index}__")
        out.append(block)
    result = pd.concat(out, axis=1).reset_index()

    # mean of the worst fifth of tiles: the size of the damaged patch
    for index in INDICES:
        column = f"z_{index}"
        worst = (frame.groupby(KEYS)[column]
                 .apply(lambda s: s[s <= s.quantile(0.20)].mean())
                 .rename(f"tanom_{index}__worst_quintile_mean").reset_index())
        result = result.merge(worst, on=KEYS, how="left")
    return result


def main() -> None:
    tiles = load_tiles()
    print(f"tiles: {len(tiles)} rows, "
          f"{tiles.groupby(KEYS).ngroups} district-seasons, "
          f"{tiles.groupby(['district_id', 'tile_index']).ngroups} unique tiles")

    shape = shape_features(tiles)
    anomaly = anomaly_features(tiles)
    features = shape.merge(anomaly, on=KEYS, validate="one_to_one")

    tile_counts = tiles.groupby(KEYS).size().rename("tile_count").reset_index()
    features = features.merge(tile_counts, on=KEYS, validate="one_to_one")

    features.to_parquet(DATA / "v16_tile_features.parquet", index=False)

    shape_columns = [c for c in features.columns if c.startswith("tile_")
                     and c != "tile_count"]
    anomaly_columns = [c for c in features.columns if c.startswith("tanom_")]
    groups = {"tile_shape": sorted(shape_columns),
              "tile_anomaly": sorted(anomaly_columns),
              "tile_all": sorted(shape_columns + anomaly_columns)}
    (DATA / "v16_tile_groups.json").write_text(json.dumps(groups, indent=1))

    summary = {
        "district_seasons": int(len(features)),
        "years": [int(features.season_start_year.min()),
                  int(features.season_start_year.max())],
        "districts": int(features.district_id.nunique()),
        "unique_tiles": int(tiles.groupby(["district_id", "tile_index"]).ngroups),
        "tiles_per_district": {
            "min": int(tile_counts.tile_count.min()),
            "median": int(tile_counts.tile_count.median()),
            "max": int(tile_counts.tile_count.max())},
        "feature_counts": {k: len(v) for k, v in groups.items()},
    }
    (ARTIFACTS / "tile_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
