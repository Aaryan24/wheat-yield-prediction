#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from src.preprocessing.s2s_grid import build_grid_from_grib


def _load_config(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _yearly_temp_paths(base: Path, start_year: int, end_year: int) -> List[Path]:
    paths: List[Path] = []
    for year in range(start_year, end_year + 1):
        p = base / f"s2s_district_temp_6h_{year}.parquet"
        if p.exists():
            paths.append(p)
    return paths


def _find_sample_grib(s2s_root: Path, start_year: int, end_year: int) -> Path:
    for year in range(start_year, end_year + 1):
        candidates = sorted((s2s_root / str(year)).glob("s2s_temp_*.grib"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError("No sample S2S temp GRIB found.")


def _issue_grib_path(s2s_root: Path, issue_date: str) -> Path:
    y, m, d = issue_date.split("-")
    return s2s_root / y / f"s2s_temp_{y}_{m}_{d}.grib"


def main() -> None:
    parser = argparse.ArgumentParser(description="District-cell breakdown for lowest 6h tmin events.")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2023)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    processed_root = Path(cfg["paths"]["processed"]) / "s2s_district"
    s2s_root = Path(cfg["paths"]["raw"]["s2s"])
    qa_root = processed_root / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)

    temp_paths = _yearly_temp_paths(processed_root, args.start_year, args.end_year)
    if not temp_paths:
        raise FileNotFoundError("No yearly temp parquet files found in processed output folder.")

    # 1) Find coldest 6h mn2t6 event per district over the requested year range.
    all_rows = []
    for p in temp_paths:
        year = int(p.stem.split("_")[-1])
        df = pd.read_parquet(
            p,
            columns=[
                "district_id",
                "district_key",
                "state_name",
                "district_name",
                "issue_date",
                "lead_step_hours",
                "mn2t6_mean",
            ],
        )
        df["year"] = year
        all_rows.append(df)
    temp_all = pd.concat(all_rows, ignore_index=True)

    idx = temp_all.groupby("district_id")["mn2t6_mean"].idxmin()
    district_min = temp_all.loc[idx].copy().reset_index(drop=True)
    district_min["mn2t6_mean_c"] = district_min["mn2t6_mean"] - 273.15
    district_min = district_min.sort_values(["state_name", "district_name", "district_id"])

    district_min_out = qa_root / f"district_min_tmin6h_events_{args.start_year}_{args.end_year}.csv"
    district_min.to_csv(district_min_out, index=False)

    # 2) Prepare district-cell weights and grid-cell coordinates.
    weights = pd.read_parquet(processed_root / "weights.parquet")
    sample_grib = _find_sample_grib(s2s_root, args.start_year, args.end_year)
    grid, _ = build_grid_from_grib(sample_grib)
    cell_map = grid[["cell_id", "i", "j", "lat", "lon"]].copy()
    weights = weights.merge(cell_map, on="cell_id", how="left")

    # 3) For each district minimum event, fetch raw cell-level 6h mn2t6 and weighted components.
    by_issue: Dict[str, pd.DataFrame] = {}
    for issue_date in sorted(district_min["issue_date"].unique().tolist()):
        issue_rows = district_min[district_min["issue_date"] == issue_date].copy()
        issue_file = _issue_grib_path(s2s_root, issue_date)
        if not issue_file.exists():
            issue_rows["event_error"] = "missing_issue_grib_file"
            by_issue[issue_date] = issue_rows
            continue

        ds = xr.open_dataset(issue_file, engine="cfgrib", backend_kwargs={"indexpath": ""})
        step_hours = ds["step"].values.astype("timedelta64[h]").astype(int)
        step_to_idx = {int(h): idx_ for idx_, h in enumerate(step_hours)}
        vals = ds["mn2t6"].values  # shape: (step, lat, lon)

        expanded_rows = []
        for _, event in issue_rows.iterrows():
            lead_h = int(event["lead_step_hours"])
            if lead_h not in step_to_idx:
                expanded_rows.append(
                    {
                        **event.to_dict(),
                        "event_error": "lead_step_missing_in_grib",
                    }
                )
                continue

            sidx = step_to_idx[lead_h]
            district_cells = weights[weights["district_id"] == event["district_id"]].copy()
            for _, cell in district_cells.iterrows():
                i = int(cell["i"])
                j = int(cell["j"])
                raw_k = float(vals[sidx, i, j])
                expanded_rows.append(
                    {
                        "year": int(event["year"]),
                        "state_name": event["state_name"],
                        "district_name": event["district_name"],
                        "district_id": event["district_id"],
                        "issue_date": event["issue_date"],
                        "lead_step_hours": lead_h,
                        "district_min_tmin_k": float(event["mn2t6_mean"]),
                        "district_min_tmin_c": float(event["mn2t6_mean"] - 273.15),
                        "cell_id": int(cell["cell_id"]),
                        "cell_lat": float(cell["lat"]),
                        "cell_lon": float(cell["lon"]),
                        "cell_i": i,
                        "cell_j": j,
                        "weight": float(cell["weight"]),
                        "cell_mn2t6_k": raw_k,
                        "cell_mn2t6_c": raw_k - 273.15,
                        "weighted_component_k": float(cell["weight"]) * raw_k,
                        "weighted_component_c": float(cell["weight"]) * (raw_k - 273.15),
                        "event_error": "",
                    }
                )
        by_issue[issue_date] = pd.DataFrame(expanded_rows)

    breakdown = pd.concat(by_issue.values(), ignore_index=True)

    # 4) Add reconstruction check against aggregated district value.
    ok_rows = breakdown[breakdown["event_error"] == ""].copy()
    recon = (
        ok_rows.groupby(["district_id", "issue_date", "lead_step_hours"], as_index=False)["weighted_component_k"]
        .sum()
        .rename(columns={"weighted_component_k": "reconstructed_tmin_k"})
    )
    breakdown = breakdown.merge(
        recon, on=["district_id", "issue_date", "lead_step_hours"], how="left"
    )
    breakdown["reconstruction_error_k"] = breakdown["reconstructed_tmin_k"] - breakdown["district_min_tmin_k"]

    breakdown_out = qa_root / f"district_min_tmin6h_cell_breakdown_{args.start_year}_{args.end_year}.csv"
    breakdown.to_csv(breakdown_out, index=False)

    print(f"Wrote: {district_min_out}")
    print(f"Wrote: {breakdown_out}")
    print(f"Districts covered: {district_min['district_id'].nunique()}")
    print(f"Breakdown rows: {len(breakdown)}")


if __name__ == "__main__":
    main()
