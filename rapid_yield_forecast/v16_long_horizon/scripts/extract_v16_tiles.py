#!/usr/bin/env python3
"""Sub-district MODIS tiles: recover the within-district detail V15 discards.

V15 collapses an entire district -- median area 3,405 km2, up to 9,922 km2 --
into 3 spatial views x 7 summaries.  MMST-ViT's Spatial Transformer instead
partitions each county into ~9 km grids and attends over them; their ablation
shows removing the imagery stream costs corn RMSE 10.5 -> 15.2.

Within-district spread is a direct heat and drought damage signal.  A district
where half the area has collapsed and half is fine has the same mean NDVI as a
district that is uniformly mediocre, and a very different yield outcome.  The
district mean destroys that distinction; tiles keep it.

MODIS is used rather than Sentinel deliberately.  Sentinel starts in 2017 and
would give four test seasons, which is the exact limitation V16 exists to fix.
MODIS runs 2000-2022 at 250 m, so a 9 km tile still holds ~1,300 pixels -- more
than enough for a stable tile summary -- and yields nineteen usable folds.

Notes on running this:
  * the Earth Engine project is in restricted-quota mode, so requests are made
    one district-season-clock at a time with retries and the output is written
    incrementally.  Re-running skips work already on disk.
  * every window ends on or before its forecast clock, so nothing observed
    after the clock enters a feature.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

V16 = Path(__file__).resolve().parents[1]
RAPID = V16.parent
DATA = V16 / "data"
BOUNDARIES = (RAPID / "v5" / "agent_modis_history" / "data"
              / "district_boundaries.geojson")
OUTPUT = DATA / "tiles"

PROJECT = "ugp-prediction"
TILE_METRES = 9000
SCALE = 250
CLOCKS = {"jan15": (1, 15), "feb15": (2, 15), "mar05": (3, 5)}
SEASON_START = (11, 1)          # 1 November of the season-start year


def district_geometry(row) -> ee.Geometry:
    simplified = row.geometry.simplify(0.005, preserve_topology=True)
    mapping = json.loads(gpd.GeoSeries([simplified], crs=4326).to_json())
    return ee.Geometry(mapping["features"][0]["geometry"])


def vegetation_indices(image: ee.Image) -> ee.Image:
    """NDVI, EVI and NDWI from MOD09A1 surface reflectance."""
    red = image.select("sur_refl_b01").multiply(0.0001)
    nir = image.select("sur_refl_b02").multiply(0.0001)
    blue = image.select("sur_refl_b03").multiply(0.0001)
    swir = image.select("sur_refl_b06").multiply(0.0001)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
    evi = (nir.subtract(red)
           .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
           .multiply(2.5).rename("evi"))
    ndwi = nir.subtract(swir).divide(nir.add(swir)).rename("ndwi")
    return ee.Image.cat([ndvi, evi, ndwi]).copyProperties(image, ["system:time_start"])


def clock_windows(season: int, clock: str) -> tuple[str, str, str]:
    month, day = CLOCKS[clock]
    end = f"{season + 1}-{month:02d}-{day:02d}"
    start = f"{season}-{SEASON_START[0]:02d}-{SEASON_START[1]:02d}"
    recent = (pd.Timestamp(end) - pd.Timedelta(days=48)).strftime("%Y-%m-%d")
    return start, recent, end


def tile_table(geometry: ee.Geometry, season: int, clock: str) -> pd.DataFrame:
    """Per-tile summaries for one district, season and forecast clock."""
    start, recent, end = clock_windows(season, clock)
    collection = (ee.ImageCollection("MODIS/061/MOD09A1")
                  .filterBounds(geometry).filterDate(start, end)
                  .map(vegetation_indices))
    recent_collection = collection.filterDate(recent, end)

    stack = ee.Image.cat([
        collection.select("ndvi").mean().rename("ndvi_season_mean"),
        collection.select("ndvi").max().rename("ndvi_season_max"),
        collection.select("ndvi").reduce(ee.Reducer.stdDev()).rename("ndvi_season_sd"),
        recent_collection.select("ndvi").mean().rename("ndvi_recent_mean"),
        collection.select("evi").mean().rename("evi_season_mean"),
        recent_collection.select("evi").mean().rename("evi_recent_mean"),
        collection.select("ndwi").mean().rename("ndwi_season_mean"),
        recent_collection.select("ndwi").mean().rename("ndwi_recent_mean"),
    ])
    grid = geometry.coveringGrid(ee.Projection("EPSG:4326").atScale(TILE_METRES))
    reduced = stack.reduceRegions(collection=grid, reducer=ee.Reducer.mean(),
                                  scale=SCALE)
    features = reduced.getInfo()["features"]
    rows = []
    for index, feature in enumerate(features):
        properties = dict(feature["properties"])
        properties["tile_index"] = index
        rows.append(properties)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2000:2022")
    parser.add_argument("--clocks", default="mar05",
                        help="comma separated; mar05 is the primary clock")
    parser.add_argument("--districts", type=int, default=0,
                        help="limit district count for a trial run")
    parser.add_argument("--retries", type=int, default=4)
    arguments = parser.parse_args()

    start_year, end_year = (int(v) for v in arguments.years.split(":"))
    clocks = [c.strip() for c in arguments.clocks.split(",") if c.strip()]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    ee.Initialize(project=PROJECT)
    frame = gpd.read_file(BOUNDARIES).to_crs(4326)
    if arguments.districts:
        frame = frame.head(arguments.districts)

    total = len(frame) * (end_year - start_year + 1) * len(clocks)
    done = 0
    started = time.time()
    for row in frame.itertuples(index=False):
        geometry = None
        for season in range(start_year, end_year + 1):
            for clock in clocks:
                path = OUTPUT / f"{row.district_id}_{season}_{clock}.parquet"
                done += 1
                if path.exists():
                    continue
                if geometry is None:
                    geometry = district_geometry(row)
                for attempt in range(arguments.retries):
                    try:
                        table = tile_table(geometry, season, clock)
                        table["district_id"] = row.district_id
                        table["state_name"] = row.state_name
                        table["season_start_year"] = season
                        table["clock"] = clock
                        table.to_parquet(path, index=False)
                        break
                    except Exception as error:                      # noqa: BLE001
                        wait = 5 * (attempt + 1)
                        print(f"    retry {attempt + 1} for {path.name}: "
                              f"{type(error).__name__} {str(error)[:90]}",
                              flush=True)
                        time.sleep(wait)
                else:
                    print(f"    GAVE UP on {path.name}", flush=True)
            elapsed = time.time() - started
            print(f"  {row.district_id} {season}: {done}/{total} "
                  f"({elapsed / 60:.1f} min elapsed)", flush=True)


if __name__ == "__main__":
    main()
