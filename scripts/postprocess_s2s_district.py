#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import geopandas as gpd
import pandas as pd
import torch
import yaml
from shapely.geometry import box

from src.preprocessing.s2s_grid import build_grid_from_grib


YEAR_RE = re.compile(r"s2s_district_daily_(\d{4})\.parquet$")


def _load_config(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _collect_years(input_dir: Path) -> List[int]:
    years: Set[int] = set()
    for p in input_dir.glob("s2s_district_daily_*.parquet"):
        m = YEAR_RE.search(p.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def _sample_grib(s2s_root: Path, years: List[int]) -> Path:
    for year in years:
        candidates = sorted((s2s_root / str(year)).glob("s2s_temp_*.grib"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError("No S2S temp GRIB found for requested years.")


def _coverage_table(cfg: dict, years: List[int]) -> pd.DataFrame:
    bcfg = cfg["boundaries"]
    gdf = gpd.read_file(Path(bcfg["admin2_path"]), layer=bcfg["admin2_layer"])
    gdf = gdf[gdf[bcfg.get("country_field", "ISO_A3")] == bcfg["country_iso3"]]
    gdf = gdf[gdf[bcfg["state_field"]].isin(bcfg["state_names"])].copy()

    gdf["district_id"] = gdf[bcfg["district_code_field"]].fillna(gdf[bcfg["district_field"]]).astype(str)
    gdf["state_name"] = gdf[bcfg["state_field"]].astype(str)
    gdf["district_name"] = gdf[bcfg["district_field"]].astype(str)

    sample = _sample_grib(Path(cfg["paths"]["raw"]["s2s"]), years)
    _, info = build_grid_from_grib(sample)
    bbox = box(*info.bounds)

    full = gdf.to_crs("EPSG:6933")
    full_area = full.set_index("district_id").geometry.area

    clipped = gdf.copy()
    clipped["geometry"] = clipped.geometry.intersection(bbox)
    clipped = clipped[~clipped.geometry.is_empty].to_crs("EPSG:6933")
    clipped_area = clipped.set_index("district_id").geometry.area

    ratio = (clipped_area / full_area).fillna(0.0)
    out = (
        gdf[["district_id", "state_name", "district_name"]]
        .drop_duplicates()
        .merge(ratio.rename("coverage_ratio"), on="district_id", how="left")
        .fillna({"coverage_ratio": 0.0})
    )
    return out.sort_values(["state_name", "district_name", "district_id"]).reset_index(drop=True)


def _to_pt(parquet_path: Path, pt_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    payload = {col: df[col].to_numpy() for col in df.columns}
    torch.save(payload, pt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply coverage filter and temperature floor to district S2S outputs.")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--input-dir", type=str, default="data/processed/s2s_district")
    parser.add_argument("--output-dir", type=str, default="data/processed/s2s_district_clean")
    parser.add_argument("--min-coverage-ratio", type=float, default=0.8)
    parser.add_argument("--drop-low-coverage", action="store_true")
    parser.add_argument("--temp-floor-c", type=float, default=-2.0)
    parser.add_argument("--export-pt", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    qa_dir = output_dir / "qa"
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    years = _collect_years(input_dir)
    if not years:
        raise FileNotFoundError(f"No yearly daily parquet files found in {input_dir}")

    coverage = _coverage_table(cfg, years)
    coverage["low_coverage_flag"] = coverage["coverage_ratio"] < args.min_coverage_ratio
    if args.drop_low_coverage:
        keep_ids = set(
            coverage.loc[~coverage["low_coverage_flag"], "district_id"].astype(str).tolist()
        )
    else:
        keep_ids = set(coverage["district_id"].astype(str).tolist())
    dropped = coverage[~coverage["district_id"].astype(str).isin(keep_ids)].copy()
    cov_map = coverage.set_index("district_id")[["coverage_ratio", "low_coverage_flag"]]

    floor_k = args.temp_floor_c + 273.15
    summary_rows: List[Dict[str, object]] = []

    for year in years:
        daily_in = input_dir / f"s2s_district_daily_{year}.parquet"
        temp_in = input_dir / f"s2s_district_temp_6h_{year}.parquet"
        if not daily_in.exists() or not temp_in.exists():
            continue

        daily = pd.read_parquet(daily_in)
        temp = pd.read_parquet(temp_in)
        daily_rows_before = len(daily)
        temp_rows_before = len(temp)

        daily = daily[daily["district_id"].astype(str).isin(keep_ids)].copy()
        temp = temp[temp["district_id"].astype(str).isin(keep_ids)].copy()

        # Keep coverage diagnostics in the cleaned output for district-specific handling.
        daily = daily.merge(cov_map, left_on="district_id", right_index=True, how="left")
        temp = temp.merge(cov_map, left_on="district_id", right_index=True, how="left")

        daily_clip_tmin = int((daily["tmin_mean"] < floor_k).sum()) if "tmin_mean" in daily.columns else 0
        daily_clip_tmax = int((daily["tmax_mean"] < floor_k).sum()) if "tmax_mean" in daily.columns else 0
        temp_clip_mn = int((temp["mn2t6_mean"] < floor_k).sum()) if "mn2t6_mean" in temp.columns else 0
        temp_clip_mx = int((temp["mx2t6_mean"] < floor_k).sum()) if "mx2t6_mean" in temp.columns else 0

        if "tmin_mean" in daily.columns:
            daily["tmin_mean"] = daily["tmin_mean"].clip(lower=floor_k)
        if "tmax_mean" in daily.columns:
            daily["tmax_mean"] = daily["tmax_mean"].clip(lower=floor_k)
        if "mn2t6_mean" in temp.columns:
            temp["mn2t6_mean"] = temp["mn2t6_mean"].clip(lower=floor_k)
        if "mx2t6_mean" in temp.columns:
            temp["mx2t6_mean"] = temp["mx2t6_mean"].clip(lower=floor_k)

        daily_out = output_dir / f"s2s_district_daily_{year}.parquet"
        temp_out = output_dir / f"s2s_district_temp_6h_{year}.parquet"
        daily.to_parquet(daily_out, index=False)
        temp.to_parquet(temp_out, index=False)

        if args.export_pt:
            _to_pt(daily_out, output_dir / f"s2s_district_daily_{year}.pt")
            _to_pt(temp_out, output_dir / f"s2s_district_temp_6h_{year}.pt")

        summary_rows.append(
            {
                "year": year,
                "daily_rows_before": daily_rows_before,
                "daily_rows_after": len(daily),
                "temp6h_rows_before": temp_rows_before,
                "temp6h_rows_after": len(temp),
                "daily_tmin_clipped": daily_clip_tmin,
                "daily_tmax_clipped": daily_clip_tmax,
                "temp6h_mn_clipped": temp_clip_mn,
                "temp6h_mx_clipped": temp_clip_mx,
                "districts_kept": daily["district_id"].nunique(),
                "low_coverage_rows_daily": int(daily["low_coverage_flag"].sum()),
                "low_coverage_rows_temp6h": int(temp["low_coverage_flag"].sum()),
            }
        )

    coverage.to_csv(qa_dir / "coverage_ratio_by_district.csv", index=False)
    dropped.to_csv(qa_dir / "dropped_districts_by_coverage.csv", index=False)
    pd.DataFrame(summary_rows).sort_values("year").to_csv(qa_dir / "postprocess_summary.csv", index=False)

    print(f"Output dir: {output_dir}")
    print(f"Years processed: {years}")
    print(f"Coverage threshold: {args.min_coverage_ratio}")
    print(f"Drop low coverage: {args.drop_low_coverage}")
    print(f"Temperature floor: {args.temp_floor_c} C ({floor_k} K)")
    print(f"Dropped districts: {len(dropped)}")


if __name__ == "__main__":
    main()
