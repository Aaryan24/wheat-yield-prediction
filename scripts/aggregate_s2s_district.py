#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from src.preprocessing.s2s_grid import build_grid_from_grib
from src.preprocessing.s2s_weights import (
    BoundaryConfig,
    build_weight_matrix,
    clip_to_bounds,
    compute_weights,
    load_boundaries,
)
from src.preprocessing.s2s_aggregate import aggregate_issue_date, iterate_issue_dates


def _load_config(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_parquet_chunk(writer, df: pd.DataFrame, output_path: Path):
    if df.empty:
        return writer
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema)
    writer.write_table(table)
    return writer


def _parquet_to_pt(parquet_path: Path, pt_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    payload = {col: df[col].to_numpy() for col in df.columns}
    torch.save(payload, pt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate S2S data to district level.")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--export-pt", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))

    boundary_cfg = BoundaryConfig(
        path=Path(cfg["boundaries"]["admin2_path"]),
        layer=cfg["boundaries"]["admin2_layer"],
        country_iso3=cfg["boundaries"]["country_iso3"],
        state_names=tuple(cfg["boundaries"]["state_names"]),
        state_field=cfg["boundaries"]["state_field"],
        district_field=cfg["boundaries"]["district_field"],
        district_code_field=cfg["boundaries"]["district_code_field"],
        country_field=cfg["boundaries"].get("country_field", "ISO_A3"),
    )

    s2s_root = Path(cfg["paths"]["raw"]["s2s"])
    output_root = Path(cfg["paths"]["processed"]) / "s2s_district"
    _ensure_dir(output_root)

    sample_candidates = list((s2s_root / str(args.start_year)).glob("s2s_temp_*.grib"))
    if not sample_candidates:
        raise FileNotFoundError(
            f"No S2S temp files found in {s2s_root}/{args.start_year}."
        )
    sample_grib = sample_candidates[0]
    grid, grid_info = build_grid_from_grib(sample_grib)

    districts = load_boundaries(boundary_cfg)
    districts = clip_to_bounds(districts, grid_info.bounds)

    weights_df = compute_weights(districts, grid)
    weights_path = output_root / "weights.parquet"
    weights_df.to_parquet(weights_path, index=False)

    weights, district_table = build_weight_matrix(weights_df, grid.shape[0])
    district_table_path = output_root / "districts.parquet"
    district_table.to_parquet(district_table_path, index=False)

    year_map = iterate_issue_dates(s2s_root, args.start_year, args.end_year)

    for year, issues in year_map.items():
        temp_path = output_root / f"s2s_district_temp_6h_{year}.parquet"
        daily_path = output_root / f"s2s_district_daily_{year}.parquet"

        temp_writer = None
        daily_writer = None
        skipped = 0
        for issue in issues:
            try:
                temp_df, daily_df = aggregate_issue_date(issue, weights, district_table)
            except Exception as exc:
                skipped += 1
                print(f"[WARN] Skipping issue date {issue.date.date()} ({year}): {exc}")
                continue
            temp_writer = _write_parquet_chunk(temp_writer, temp_df, temp_path)
            daily_writer = _write_parquet_chunk(daily_writer, daily_df, daily_path)

        if temp_writer is not None:
            temp_writer.close()
        if daily_writer is not None:
            daily_writer.close()
        if skipped:
            print(f"[INFO] Year {year}: skipped {skipped} issue date(s) due to corrupted/incomplete files.")

        if args.export_pt:
            _parquet_to_pt(temp_path, output_root / f"s2s_district_temp_6h_{year}.pt")
            _parquet_to_pt(daily_path, output_root / f"s2s_district_daily_{year}.pt")


if __name__ == "__main__":
    main()
