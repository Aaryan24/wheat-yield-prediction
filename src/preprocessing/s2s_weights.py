from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import box
except Exception as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "geopandas and shapely are required for boundary processing. "
        "Install them from requirements.txt before running this module."
    ) from exc


@dataclass(frozen=True)
class BoundaryConfig:
    path: Path
    layer: str
    country_iso3: str
    state_names: Tuple[str, ...]
    state_field: str
    district_field: str
    district_code_field: str
    country_field: str = "ISO_A3"


def load_boundaries(cfg: BoundaryConfig) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(cfg.path, layer=cfg.layer)
    gdf = gdf[gdf[cfg.country_field] == cfg.country_iso3]
    gdf = gdf[gdf[cfg.state_field].isin(cfg.state_names)]

    gdf = gdf.rename(
        columns={
            cfg.state_field: "state_name",
            cfg.district_field: "district_name",
            cfg.district_code_field: "district_code",
        }
    )

    gdf["district_id"] = gdf["district_code"].fillna(gdf["district_name"]).astype(str)
    gdf["state_name"] = gdf["state_name"].astype(str)
    gdf["district_name"] = gdf["district_name"].astype(str)
    gdf["district_key"] = gdf["state_name"] + "::" + gdf["district_name"]
    gdf = gdf[["district_id", "district_key", "state_name", "district_name", "geometry"]]
    return gdf


def clip_to_bounds(gdf: gpd.GeoDataFrame, bounds: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    min_lon, min_lat, max_lon, max_lat = bounds
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    clipped = gpd.clip(gdf, bbox_geom)
    clipped = clipped[~clipped.geometry.is_empty]
    return clipped


def compute_weights(
    districts: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    area_crs: str = "EPSG:6933",
) -> pd.DataFrame:
    """Compute area-based weights for each district/grid cell intersection."""
    districts_area = districts.to_crs(area_crs)
    grid_area = grid.to_crs(area_crs)

    districts_area = districts_area.copy()
    districts_area["district_area"] = districts_area.geometry.area

    intersections = gpd.overlay(districts_area, grid_area, how="intersection")
    intersections["inter_area"] = intersections.geometry.area

    if "district_area" not in intersections.columns:
        intersections = intersections.merge(
            districts_area[["district_id", "district_area"]], on="district_id", how="left"
        )
        district_area_col = "district_area"
    else:
        district_area_col = "district_area"

    intersections["weight"] = intersections["inter_area"] / intersections[district_area_col]

    weights = intersections[
        ["district_id", "district_key", "state_name", "district_name", "cell_id", "weight"]
    ].copy()

    weights = weights[weights["weight"] > 0].reset_index(drop=True)
    return weights


def build_weight_matrix(weights: pd.DataFrame, n_cells: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return (weight_matrix, district_table)."""
    district_table = (
        weights[["district_id", "district_key", "state_name", "district_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    district_table["district_index"] = np.arange(len(district_table))

    district_index = {
        row.district_id: row.district_index for row in district_table.itertuples()
    }

    matrix = np.zeros((len(district_table), n_cells), dtype=np.float32)
    for row in weights.itertuples():
        d_idx = district_index[row.district_id]
        matrix[d_idx, int(row.cell_id)] = float(row.weight)

    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    return matrix, district_table
