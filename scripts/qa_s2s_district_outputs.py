#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import yaml


DATE_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})")


def _load_config(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _dates_for_pattern(year_dir: Path, pattern: str) -> Set[str]:
    out: Set[str] = set()
    for p in year_dir.glob(pattern):
        m = DATE_RE.search(p.name)
        if not m:
            continue
        out.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return out


def _expected_issue_dates(s2s_root: Path, year: int) -> Set[str]:
    year_dir = s2s_root / str(year)
    if not year_dir.exists():
        return set()
    temp_dates = _dates_for_pattern(year_dir, "s2s_temp_*.grib")
    wind_dates = _dates_for_pattern(year_dir, "s2s_wind_*.grib")
    accum_dates = _dates_for_pattern(year_dir, "s2s_accum_*.grib")
    return temp_dates & wind_dates & accum_dates


def _read_issue_dates(parquet_path: Path) -> Set[str]:
    if not parquet_path.exists():
        return set()
    df = pd.read_parquet(parquet_path, columns=["issue_date"])
    return set(df["issue_date"].dropna().astype(str).unique().tolist())


def _variable_stats(parquet_path: Path, year: int, dataset: str) -> List[dict]:
    if not parquet_path.exists():
        return []
    df = pd.read_parquet(parquet_path)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    rows: List[dict] = []
    for col in numeric_cols:
        rows.append(
            {
                "year": year,
                "dataset": dataset,
                "variable": col,
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "nan_count": int(df[col].isna().sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="QA checks for district-level S2S outputs.")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2023)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    s2s_root = Path(cfg["paths"]["raw"]["s2s"])
    processed_root = Path(cfg["paths"]["processed"]) / "s2s_district"
    qa_root = processed_root / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)

    skipped_rows: List[dict] = []
    coverage_rows: List[dict] = []
    stats_rows: List[dict] = []

    for year in range(args.start_year, args.end_year + 1):
        daily_path = processed_root / f"s2s_district_daily_{year}.parquet"
        temp_path = processed_root / f"s2s_district_temp_6h_{year}.parquet"

        expected = _expected_issue_dates(s2s_root, year)
        actual_daily = _read_issue_dates(daily_path)
        actual_temp = _read_issue_dates(temp_path)
        actual = actual_daily & actual_temp
        missing = sorted(expected - actual)

        for date_str in missing:
            skipped_rows.append(
                {
                    "year": year,
                    "issue_date": date_str,
                    "status": "missing_in_output",
                    "note": "Likely corrupted or incomplete source GRIB file(s).",
                }
            )

        coverage_rows.append(
            {
                "year": year,
                "expected_issue_dates": len(expected),
                "actual_issue_dates": len(actual),
                "missing_issue_dates": len(missing),
                "daily_rows": int(pd.read_parquet(daily_path, columns=["issue_date"]).shape[0])
                if daily_path.exists()
                else 0,
                "temp6h_rows": int(pd.read_parquet(temp_path, columns=["issue_date"]).shape[0])
                if temp_path.exists()
                else 0,
            }
        )

        stats_rows.extend(_variable_stats(daily_path, year, "daily"))
        stats_rows.extend(_variable_stats(temp_path, year, "temp6h"))

    skipped_df = pd.DataFrame(skipped_rows).sort_values(["year", "issue_date"]) if skipped_rows else pd.DataFrame(
        columns=["year", "issue_date", "status", "note"]
    )
    coverage_df = pd.DataFrame(coverage_rows).sort_values("year")
    stats_df = pd.DataFrame(stats_rows).sort_values(["year", "dataset", "variable"])

    skipped_df.to_csv(qa_root / f"skipped_issue_dates_{args.start_year}_{args.end_year}.csv", index=False)
    coverage_df.to_csv(qa_root / f"coverage_summary_{args.start_year}_{args.end_year}.csv", index=False)
    stats_df.to_csv(qa_root / f"variable_min_max_{args.start_year}_{args.end_year}.csv", index=False)

    print(f"QA reports written to: {qa_root}")
    print(f"Skipped issue rows: {len(skipped_df)}")
    print(f"Coverage rows: {len(coverage_df)}")
    print(f"Variable stats rows: {len(stats_df)}")


if __name__ == "__main__":
    main()
