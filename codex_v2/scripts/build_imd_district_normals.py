#!/usr/bin/env python3
"""
Build district-wise IMD climate normals for UGP districts (Punjab/Haryana/UP).

Outputs:
- district_rain_normals_monthly_1971_2020.csv
- district_rain_normals_annual_1971_2020.csv
- district_temp_normals_monthly_<start>_<end>.csv
- run_metadata.json
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import box, shape


STATES = ("Punjab", "Haryana", "Uttar Pradesh")
MONTHS = [
    ("jan", 1),
    ("feb", 2),
    ("mar", 3),
    ("apr", 4),
    ("may", 5),
    ("jun", 6),
    ("jul", 7),
    ("aug", 8),
    ("sep", 9),
    ("oct", 10),
    ("nov", 11),
    ("dec", 12),
]
RAIN_MONTHLY_URL = "https://imdpune.gov.in/climinfo/normals/{m}/layers/Rainfallinmm_1.js"
RAIN_ANNUAL_URL = "https://imdpune.gov.in/climinfo/season/ann/layers/Rainfallinmm_1.js"
IMD_TEMP_BASE = "https://www.imdpune.gov.in/cmpg/Griddata"
TEMP_ENDPOINTS = {
    "tmax": ("maxtemp.php", "maxtemp"),
    "tmin": ("mintemp.php", "mintemp"),
}


def norm_state(text: str) -> str:
    t = " ".join(str(text).strip().lower().replace("&", " and ").split())
    if t in {"up", "uttar pradesh"}:
        return "Uttar Pradesh"
    if t == "haryana":
        return "Haryana"
    if t == "punjab":
        return "Punjab"
    return str(text).strip().title()


def canon_name(text: str) -> str:
    t = str(text).strip().lower()
    t = t.replace("&", " and ").replace("'", "")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = " ".join(t.split())
    return t


def fetch_json_from_js(url: str, timeout: int = 90) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    txt = r.text
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j <= i:
        raise RuntimeError(f"Could not parse JSON payload from {url}")
    return json.loads(txt[i : j + 1])


def extract_rain_features(payload: dict, keep_geometry: bool) -> gpd.GeoDataFrame | pd.DataFrame:
    rows: List[dict] = []
    for feat in payload.get("features", []):
        p = feat.get("properties", {})
        st = norm_state(p.get("STATE", p.get("State", "")))
        if st not in STATES:
            continue
        rec = {
            "imd_id": str(p.get("ID", p.get("id", ""))),
            "state_name": st,
            "district_name_imd": str(p.get("DISTRICT", p.get("District", ""))).strip(),
            "rain_mm": float(p.get("Rainfall (in mm)")) if p.get("Rainfall (in mm)") is not None else np.nan,
        }
        if keep_geometry:
            rec["geometry"] = shape(feat.get("geometry"))
        rows.append(rec)
    if keep_geometry:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return pd.DataFrame(rows)


def load_local_districts(districts_path: Path, boundaries_path: Path) -> gpd.GeoDataFrame:
    ddf = pd.read_parquet(districts_path)[
        ["district_id", "district_index", "state_name", "district_name"]
    ].copy()
    ddf["district_id"] = ddf["district_id"].astype(str)
    ddf["state_name"] = ddf["state_name"].map(norm_state)
    ddf = ddf[ddf["state_name"].isin(STATES)].copy()
    ddf["district_key"] = ddf["district_name"].map(canon_name)
    ddf = ddf.drop_duplicates(["district_id"]).sort_values("district_index").reset_index(drop=True)

    bdf = gpd.read_file(boundaries_path)
    bdf = bdf[bdf["NAM_0"] == "India"][["NAM_1", "NAM_2", "geometry"]].copy()
    bdf["state_name"] = bdf["NAM_1"].map(norm_state)
    bdf = bdf[bdf["state_name"].isin(STATES)].copy()
    bdf["district_key"] = bdf["NAM_2"].map(canon_name)
    bdf = bdf.drop_duplicates(["state_name", "district_key"]).copy()

    out = ddf.merge(
        bdf[["state_name", "district_key", "geometry"]],
        on=["state_name", "district_key"],
        how="left",
    )
    miss = out["geometry"].isna().sum()
    if miss:
        raise RuntimeError(f"Missing geometry for {miss} districts from boundaries file.")
    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=bdf.crs)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def map_local_to_imd_by_overlap(
    local_gdf: gpd.GeoDataFrame, imd_gdf_annual: gpd.GeoDataFrame
) -> pd.DataFrame:
    local = local_gdf[["district_id", "state_name", "geometry"]].copy()
    imd = imd_gdf_annual[["imd_id", "state_name", "geometry"]].copy().rename(
        columns={"state_name": "imd_state"}
    )
    local_m = local.to_crs("EPSG:6933")
    imd_m = imd.to_crs("EPSG:6933")

    inter = gpd.overlay(local_m, imd_m, how="intersection", keep_geom_type=False)
    inter = inter[inter["state_name"] == inter["imd_state"]].copy()
    inter["inter_area"] = inter.geometry.area
    idx = inter.groupby("district_id")["inter_area"].idxmax()
    best = inter.loc[idx, ["district_id", "imd_id"]].copy()

    if best["district_id"].nunique() != local["district_id"].nunique():
        got = set(best["district_id"].astype(str))
        need = set(local["district_id"].astype(str))
        missing = sorted(need - got)
        raise RuntimeError(f"Could not spatially map all districts to IMD rainfall IDs. Missing={missing}")
    return best


def build_rain_normals(
    local_gdf: gpd.GeoDataFrame, district_map: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    monthly_frames: List[pd.DataFrame] = []
    for mname, mnum in MONTHS:
        payload = fetch_json_from_js(RAIN_MONTHLY_URL.format(m=mname))
        rdf = extract_rain_features(payload, keep_geometry=False)
        rdf["month"] = mnum
        monthly_frames.append(rdf[["imd_id", "month", "rain_mm"]])
    rain_monthly_imd = pd.concat(monthly_frames, ignore_index=True)

    rain_monthly = district_map.merge(rain_monthly_imd, on="imd_id", how="left").merge(
        local_gdf[["district_id", "state_name", "district_name"]],
        on="district_id",
        how="left",
    )
    rain_monthly = rain_monthly[
        ["district_id", "state_name", "district_name", "month", "rain_mm"]
    ].sort_values(["state_name", "district_name", "month"])

    annual_payload = fetch_json_from_js(RAIN_ANNUAL_URL)
    annual_df = extract_rain_features(annual_payload, keep_geometry=False)[["imd_id", "rain_mm"]].rename(
        columns={"rain_mm": "annual_rain_mm"}
    )
    rain_annual = district_map.merge(annual_df, on="imd_id", how="left").merge(
        local_gdf[["district_id", "state_name", "district_name"]],
        on="district_id",
        how="left",
    )
    rain_annual = rain_annual[
        ["district_id", "state_name", "district_name", "annual_rain_mm"]
    ].sort_values(["state_name", "district_name"])
    return rain_monthly, rain_annual


def make_temp_grid() -> gpd.GeoDataFrame:
    lons = np.arange(67.5, 98.0, 1.0)[:31]  # 67.5 .. 97.5
    lats = np.arange(37.5, 6.0, -1.0)[:31]  # 37.5 .. 7.5
    rows = []
    cell_id = 0
    for lat in lats:
        for lon in lons:
            rows.append(
                {
                    "cell_id": int(cell_id),
                    "lon": float(lon),
                    "lat": float(lat),
                    "geometry": box(lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5),
                }
            )
            cell_id += 1
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def build_weight_matrix(local_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, List[str]]:
    local = local_gdf[["district_id", "geometry"]].copy().to_crs("EPSG:6933")
    grid = grid_gdf[["cell_id", "geometry"]].copy().to_crs("EPSG:6933")
    inter = gpd.overlay(local, grid, how="intersection", keep_geom_type=False)
    inter["w"] = inter.geometry.area
    agg = inter.groupby(["district_id", "cell_id"], as_index=False)["w"].sum()
    agg["w"] = agg["w"] / agg.groupby("district_id")["w"].transform("sum")

    district_ids = local_gdf["district_id"].astype(str).tolist()
    d_index = {d: i for i, d in enumerate(district_ids)}
    w = np.zeros((len(district_ids), 31 * 31), dtype=np.float32)
    for r in agg.itertuples(index=False):
        w[d_index[str(r.district_id)], int(r.cell_id)] = float(r.w)

    # Fallback for districts with no overlap: nearest center cell by centroid.
    missing_rows = np.where(w.sum(axis=1) == 0)[0]
    if len(missing_rows):
        cents = local_gdf.to_crs("EPSG:4326").geometry.centroid
        lon = grid_gdf["lon"].to_numpy()
        lat = grid_gdf["lat"].to_numpy()
        for ridx in missing_rows:
            cx, cy = float(cents.iloc[ridx].x), float(cents.iloc[ridx].y)
            dist2 = (lon - cx) ** 2 + (lat - cy) ** 2
            best = int(np.argmin(dist2))
            w[ridx, best] = 1.0
    return w, district_ids


def download_temp_year(kind: str, year: int, out_dir: Path) -> Path:
    endpoint, form_field = TEMP_ENDPOINTS[kind]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{kind}_{year}.grd"
    if out_file.exists() and out_file.stat().st_size > 1000:
        return out_file
    url = f"{IMD_TEMP_BASE}/{endpoint}"
    r = requests.post(url, data={form_field: str(year)}, timeout=(30, 240))
    r.raise_for_status()
    if len(r.content) < 1000:
        raise RuntimeError(f"Received tiny payload for {kind} {year}.")
    out_file.write_bytes(r.content)
    return out_file


def parse_temp_grd(path: Path, year: int) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    raw = np.fromfile(path, dtype="<f4")
    n_cells = 31 * 31
    if raw.size % n_cells != 0:
        raise RuntimeError(f"Unexpected size in {path} (float32_count={raw.size})")
    n_days = raw.size // n_cells
    # File layout is day-wise blocks of 31x31 float32 (matches IMD C fread example).
    # Rows in GRD are south->north for our domain, while our grid geometry is north->south,
    # so flip latitude axis to align cell indices with district overlap weights.
    arr = raw.reshape(n_days, 31, 31).astype(np.float32)
    arr = arr[:, ::-1, :]
    arr = arr.reshape(n_days, n_cells)
    # IMD fills outside-domain/missing with high sentinel-like values near 99.9.
    arr[(arr > 90.0) | (arr < -90.0)] = np.nan

    expected = 366 if calendar.isleap(year) else 365
    if n_days == expected:
        dates = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    elif n_days == 366 and expected == 365:
        # Keep Jan 1..Dec 31 for non-leap years if an extra slot is present.
        arr = arr[:365, :]
        dates = pd.date_range(f"{year}-01-01", periods=365, freq="D")
    else:
        dates = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    return arr, dates


def district_daily_from_grid(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr).astype(np.float32)
    arr_z = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    num = arr_z @ w.T
    den = valid @ w.T
    out = np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float32), where=den > 0)
    return out


def build_temp_normals(
    local_gdf: gpd.GeoDataFrame,
    w: np.ndarray,
    district_ids: List[str],
    start_year: int,
    end_year: int,
    temp_cache_dir: Path,
) -> pd.DataFrame:
    n_d = len(district_ids)
    sum_tmax = np.zeros((12, n_d), dtype=np.float64)
    cnt_tmax = np.zeros((12, n_d), dtype=np.float64)
    sum_tmin = np.zeros((12, n_d), dtype=np.float64)
    cnt_tmin = np.zeros((12, n_d), dtype=np.float64)

    for year in range(start_year, end_year + 1):
        pmax = download_temp_year("tmax", year, temp_cache_dir)
        pmin = download_temp_year("tmin", year, temp_cache_dir)
        amax, dates = parse_temp_grd(pmax, year)
        amin, dates2 = parse_temp_grd(pmin, year)
        if len(dates) != len(dates2):
            n = min(len(dates), len(dates2))
            amax = amax[:n, :]
            amin = amin[:n, :]
            dates = dates[:n]

        dmax = district_daily_from_grid(amax, w)  # [days, districts]
        dmin = district_daily_from_grid(amin, w)
        mon = dates.month.to_numpy()
        for m in range(1, 13):
            idx = mon == m
            if not idx.any():
                continue
            vmx = dmax[idx, :]
            vmn = dmin[idx, :]
            sum_tmax[m - 1, :] += np.nansum(vmx, axis=0)
            cnt_tmax[m - 1, :] += np.isfinite(vmx).sum(axis=0)
            sum_tmin[m - 1, :] += np.nansum(vmn, axis=0)
            cnt_tmin[m - 1, :] += np.isfinite(vmn).sum(axis=0)
        print(f"[temp] finished year {year}")

    nmax = np.divide(sum_tmax, cnt_tmax, out=np.full_like(sum_tmax, np.nan), where=cnt_tmax > 0)
    nmin = np.divide(sum_tmin, cnt_tmin, out=np.full_like(sum_tmin, np.nan), where=cnt_tmin > 0)
    nmean = (nmax + nmin) / 2.0

    did_to_info = (
        local_gdf[["district_id", "state_name", "district_name"]]
        .drop_duplicates("district_id")
        .set_index("district_id")
    )
    rows = []
    for di, did in enumerate(district_ids):
        info = did_to_info.loc[did]
        for m in range(1, 13):
            rows.append(
                {
                    "district_id": did,
                    "state_name": str(info["state_name"]),
                    "district_name": str(info["district_name"]),
                    "month": m,
                    "tmax_c": float(nmax[m - 1, di]) if np.isfinite(nmax[m - 1, di]) else np.nan,
                    "tmin_c": float(nmin[m - 1, di]) if np.isfinite(nmin[m - 1, di]) else np.nan,
                    "tmean_c": float(nmean[m - 1, di]) if np.isfinite(nmean[m - 1, di]) else np.nan,
                }
            )
    out = pd.DataFrame(rows).sort_values(["state_name", "district_name", "month"]).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build district-wise IMD rainfall and temperature normals.")
    ap.add_argument(
        "--districts",
        type=Path,
        default=Path("/Users/aaryan/Downloads/ugp/data/processed/s2s_district/districts.parquet"),
    )
    ap.add_argument(
        "--boundaries",
        type=Path,
        default=Path("/Users/aaryan/Downloads/ugp/data/boundaries/World Bank Official Boundaries - Admin 2.gpkg"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            f"/Users/aaryan/Downloads/ugp/codex_v2/experiments/imd_normals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    ap.add_argument("--temp-start-year", type=int, default=1991)
    ap.add_argument("--temp-end-year", type=int, default=2020)
    ap.add_argument(
        "--temp-cache-dir",
        type=Path,
        default=Path("/Users/aaryan/Downloads/ugp/data/imd_temp_raw"),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading local districts + boundaries...")
    local_gdf = load_local_districts(args.districts, args.boundaries)
    print(f"    local districts: {len(local_gdf)}")

    print("[2/6] Fetching IMD annual rainfall geometry + mapping districts...")
    annual_payload = fetch_json_from_js(RAIN_ANNUAL_URL)
    annual_imd_gdf = extract_rain_features(annual_payload, keep_geometry=True)
    district_map = map_local_to_imd_by_overlap(local_gdf, annual_imd_gdf)
    print(f"    mapped districts: {district_map['district_id'].nunique()}")

    print("[3/6] Building district rainfall normals (monthly + annual)...")
    rain_monthly, rain_annual = build_rain_normals(local_gdf, district_map)
    rain_monthly_path = args.out_dir / "district_rain_normals_monthly_1971_2020.csv"
    rain_annual_path = args.out_dir / "district_rain_normals_annual_1971_2020.csv"
    rain_monthly.to_csv(rain_monthly_path, index=False)
    rain_annual.to_csv(rain_annual_path, index=False)
    print(f"    wrote: {rain_monthly_path}")
    print(f"    wrote: {rain_annual_path}")

    print("[4/6] Building 1.0° temp grid weights per district...")
    grid_gdf = make_temp_grid()
    w, district_ids = build_weight_matrix(local_gdf, grid_gdf)
    print(f"    weight matrix: {w.shape}")

    print(
        f"[5/6] Building district temp monthly normals ({args.temp_start_year}-{args.temp_end_year})..."
    )
    temp_normals = build_temp_normals(
        local_gdf=local_gdf,
        w=w,
        district_ids=district_ids,
        start_year=args.temp_start_year,
        end_year=args.temp_end_year,
        temp_cache_dir=args.temp_cache_dir,
    )
    temp_path = args.out_dir / f"district_temp_normals_monthly_{args.temp_start_year}_{args.temp_end_year}.csv"
    temp_normals.to_csv(temp_path, index=False)
    print(f"    wrote: {temp_path}")

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_districts": int(len(local_gdf)),
        "states": list(STATES),
        "rainfall_normals_source": "IMD district rainfall normals (1971-2020), monthly + annual map layers",
        "temperature_source": "IMD gridded max/min temperature binaries (1.0x1.0 degree)",
        "temperature_baseline_years": [int(args.temp_start_year), int(args.temp_end_year)],
        "temperature_grid_assumed_centers_lon": [67.5, 97.5, 1.0],
        "temperature_grid_assumed_centers_lat": [37.5, 7.5, -1.0],
        "outputs": {
            "rain_monthly": str(rain_monthly_path),
            "rain_annual": str(rain_annual_path),
            "temp_monthly": str(temp_path),
        },
    }
    meta_path = args.out_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[6/6] wrote: {meta_path}")
    print("DONE")


if __name__ == "__main__":
    main()
