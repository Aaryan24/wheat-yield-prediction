#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import torch


DAILY_RE = re.compile(r"^s2s_district_daily_(\d{4})\.parquet$")
TEMP_RE = re.compile(r"^s2s_district_temp_6h_(\d{4})\.parquet$")


def _collect_years(input_dir: Path) -> List[int]:
    years = set()
    for p in input_dir.glob("s2s_district_daily_*.parquet"):
        m = DAILY_RE.match(p.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def _to_pt(parquet_path: Path, pt_path: Path, overwrite: bool) -> Tuple[int, int]:
    if pt_path.exists() and not overwrite:
        df = pd.read_parquet(parquet_path, columns=["district_id", "issue_date"])
        return len(df), len(df.columns)
    df = pd.read_parquet(parquet_path)
    payload = {col: df[col].to_numpy() for col in df.columns}
    torch.save(payload, pt_path)
    return len(df), len(df.columns)


def _pairs_for_year(input_dir: Path, years: Iterable[int]) -> List[Tuple[int, Path, Path]]:
    out: List[Tuple[int, Path, Path]] = []
    for year in years:
        daily = input_dir / f"s2s_district_daily_{year}.parquet"
        temp = input_dir / f"s2s_district_temp_6h_{year}.parquet"
        if daily.exists() and temp.exists():
            out.append((year, daily, temp))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export district S2S parquet files to torch .pt payloads."
    )
    parser.add_argument("--input-dir", type=str, default="data/processed/s2s_district")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    years_found = _collect_years(input_dir)
    years = [y for y in years_found if args.start_year <= y <= args.end_year]
    pairs = _pairs_for_year(input_dir, years)
    if not pairs:
        raise FileNotFoundError(
            f"No matching daily/temp parquet pairs found in {input_dir} for requested years."
        )

    for year, daily_in, temp_in in pairs:
        daily_out = output_dir / f"s2s_district_daily_{year}.pt"
        temp_out = output_dir / f"s2s_district_temp_6h_{year}.pt"

        daily_rows, daily_cols = _to_pt(daily_in, daily_out, overwrite=args.overwrite)
        temp_rows, temp_cols = _to_pt(temp_in, temp_out, overwrite=args.overwrite)

        print(
            f"[{year}] daily: {daily_in.name} -> {daily_out.name} "
            f"(rows={daily_rows}, cols={daily_cols})"
        )
        print(
            f"[{year}] temp6h: {temp_in.name} -> {temp_out.name} "
            f"(rows={temp_rows}, cols={temp_cols})"
        )

    print(f"Done. Wrote .pt files to: {output_dir}")


if __name__ == "__main__":
    main()

