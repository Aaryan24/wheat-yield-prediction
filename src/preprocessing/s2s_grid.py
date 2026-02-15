from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr

try:
    import geopandas as gpd
    from shapely.geometry import box
except Exception as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "geopandas and shapely are required for grid generation. "
        "Install them from requirements.txt before running this module."
    ) from exc


@dataclass(frozen=True)
class S2SGridInfo:
    n_lat: int
    n_lon: int
    lat_res: float
    lon_res: float
    bounds: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


def _resolution(values: np.ndarray) -> float:
    if values.size < 2:
        raise ValueError("Need at least two coordinates to compute resolution.")
    return float(abs(values[1] - values[0]))


def build_grid_from_grib(grib_path: Path) -> Tuple[gpd.GeoDataFrame, S2SGridInfo]:
    """Create a GeoDataFrame of S2S grid cells from a sample GRIB file."""
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    lat_res = _resolution(lat)
    lon_res = _resolution(lon)
    half_lat = lat_res / 2.0
    half_lon = lon_res / 2.0

    rows = []
    cell_id = 0
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            geom = box(lo - half_lon, la - half_lat, lo + half_lon, la + half_lat)
            rows.append(
                {
                    "cell_id": cell_id,
                    "i": i,
                    "j": j,
                    "lat": float(la),
                    "lon": float(lo),
                    "geometry": geom,
                }
            )
            cell_id += 1

    grid = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    bounds = (float(lon.min() - half_lon), float(lat.min() - half_lat),
              float(lon.max() + half_lon), float(lat.max() + half_lat))

    info = S2SGridInfo(
        n_lat=len(lat),
        n_lon=len(lon),
        lat_res=lat_res,
        lon_res=lon_res,
        bounds=bounds,
    )
    return grid, info


def grid_index_metadata(grid: gpd.GeoDataFrame) -> Dict[str, int]:
    """Return mapping metadata for consistent flattening of grid arrays."""
    return {
        "n_cells": int(grid.shape[0]),
        "n_lat": int(grid["i"].max() + 1),
        "n_lon": int(grid["j"].max() + 1),
    }
