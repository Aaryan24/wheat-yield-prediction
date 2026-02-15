#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


@dataclass(frozen=True)
class FileSummary:
    split: str
    year: int
    dataset: str
    path: str
    rows: int
    columns: int
    districts: int
    issue_dates: int


def _load_config(config_path: Path) -> dict:
    with config_path.open("r") as fh:
        return yaml.safe_load(fh)


def _rel_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    src_abs = src.resolve()
    dst_abs_parent = dst.parent.resolve()
    link_target = Path(os.path.relpath(src_abs, dst_abs_parent))
    # Fallback to absolute path if relative conversion fails.
    try:
        dst.symlink_to(link_target)
    except Exception:
        dst.symlink_to(src_abs)


def _summary_from_parquet(path: Path, split: str, dataset: str, year: int) -> FileSummary:
    cols = None
    if dataset == "daily":
        cols = ["district_id", "issue_date"]
    else:
        cols = ["district_id", "issue_date"]
    df_small = pd.read_parquet(path, columns=cols)
    row_count = len(df_small)
    districts = int(df_small["district_id"].nunique())
    issue_dates = int(df_small["issue_date"].nunique())

    # Read only schema width by loading a tiny head.
    df_head = pd.read_parquet(path).head(1)
    return FileSummary(
        split=split,
        year=year,
        dataset=dataset,
        path=str(path),
        rows=row_count,
        columns=len(df_head.columns),
        districts=districts,
        issue_dates=issue_dates,
    )


def _write_readme(out_root: Path) -> None:
    text = """# Informer Ready Data Layout

This folder is a train-ready view of district-level S2S weather outputs.
The files are symlinked from source folders to avoid data duplication.

## Structure
- `actual/daily/`: yearly district daily features (`s2s_district_daily_YYYY.parquet`)
- `actual/temp_6h/`: yearly district 6-hour temperature features (`s2s_district_temp_6h_YYYY.parquet`)
- `actual/metadata/`: district/grid metadata and coverage diagnostics
- `qa/`: QA reports used for sanity checks
- `preview/`: small CSV previews for manual inspection
- `manifests/`: machine-readable inventory and split mappings

## Notes
- All districts are retained in this layout.
- `low_coverage_flag` marks districts with limited S2S footprint overlap.
- Temperature floor clipping has already been applied in clean outputs.
"""
    (out_root / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical Informer-ready data layout from processed S2S files.")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--clean-dir", type=str, default="data/processed/s2s_district_clean")
    parser.add_argument("--source-dir", type=str, default="data/processed/s2s_district")
    parser.add_argument("--out-dir", type=str, default="data/processed/informer_ready")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    temporal = cfg["temporal"]
    split_map: Dict[str, List[int]] = {
        "train": list(temporal["train_years"]),
        "val": list(temporal["val_years"]),
        "test": list(temporal["test_years"]),
    }

    clean_dir = Path(args.clean_dir)
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)

    actual_daily_dir = out_dir / "actual" / "daily"
    actual_temp_dir = out_dir / "actual" / "temp_6h"
    actual_meta_dir = out_dir / "actual" / "metadata"
    qa_dir = out_dir / "qa"
    preview_dir = out_dir / "preview"
    manifest_dir = out_dir / "manifests"

    for p in [actual_daily_dir, actual_temp_dir, actual_meta_dir, qa_dir, preview_dir, manifest_dir]:
        p.mkdir(parents=True, exist_ok=True)

    summaries: List[FileSummary] = []
    split_rows: List[dict] = []
    all_years = sorted(set(split_map["train"] + split_map["val"] + split_map["test"]))

    split_for_year: Dict[int, str] = {}
    for split, years in split_map.items():
        for year in years:
            split_for_year[year] = split

    for year in all_years:
        split = split_for_year[year]
        daily_src = clean_dir / f"s2s_district_daily_{year}.parquet"
        temp_src = clean_dir / f"s2s_district_temp_6h_{year}.parquet"
        if not daily_src.exists() or not temp_src.exists():
            raise FileNotFoundError(f"Missing clean files for year {year}: {daily_src.name} or {temp_src.name}")

        daily_dst = actual_daily_dir / daily_src.name
        temp_dst = actual_temp_dir / temp_src.name
        _rel_symlink(daily_src, daily_dst)
        _rel_symlink(temp_src, temp_dst)

        summaries.append(_summary_from_parquet(daily_dst, split=split, dataset="daily", year=year))
        summaries.append(_summary_from_parquet(temp_dst, split=split, dataset="temp_6h", year=year))

        split_rows.append(
            {
                "year": year,
                "split": split,
                "daily_file": str(daily_dst),
                "temp_6h_file": str(temp_dst),
            }
        )

    metadata_files = [
        clean_dir / "qa" / "coverage_ratio_by_district.csv",
        clean_dir / "qa" / "postprocess_summary.csv",
        clean_dir / "qa" / "dropped_districts_by_coverage.csv",
        source_dir / "districts.parquet",
        source_dir / "weights.parquet",
    ]
    for src in metadata_files:
        if src.exists():
            _rel_symlink(src, actual_meta_dir / src.name)

    qa_sources = sorted((source_dir / "qa").glob("*"))
    for src in qa_sources:
        if src.is_file():
            _rel_symlink(src, qa_dir / src.name)

    preview_sources = sorted((source_dir / "preview_2017").glob("*"))
    for src in preview_sources:
        if src.is_file():
            _rel_symlink(src, preview_dir / src.name)

    summary_df = pd.DataFrame([s.__dict__ for s in summaries]).sort_values(["year", "dataset"])
    summary_df.to_csv(manifest_dir / "file_inventory.csv", index=False)
    pd.DataFrame(split_rows).sort_values("year").to_csv(manifest_dir / "split_map.csv", index=False)

    schema_daily = pd.read_parquet(clean_dir / f"s2s_district_daily_{all_years[0]}.parquet").columns.tolist()
    schema_temp = pd.read_parquet(clean_dir / f"s2s_district_temp_6h_{all_years[0]}.parquet").columns.tolist()

    with (manifest_dir / "schema_daily.json").open("w") as fh:
        json.dump(schema_daily, fh, indent=2)
    with (manifest_dir / "schema_temp_6h.json").open("w") as fh:
        json.dump(schema_temp, fh, indent=2)

    _write_readme(out_dir)

    print(f"Prepared Informer-ready layout at: {out_dir}")
    print(f"Years linked: {all_years}")
    print(f"Inventory: {manifest_dir / 'file_inventory.csv'}")
    print(f"Split map: {manifest_dir / 'split_map.csv'}")


if __name__ == "__main__":
    main()
