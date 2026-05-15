#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codex_v2.src.data.build_dataset_v2 import (
    build_reforecast_climatology_frame,
    load_weather_year,
    pick_issue_date_for_operational_label,
)
from codex_v2.src.data.opdate_profiles_v2 import PROFILE_TEN_DAY_DEC1_APR30, SUPPORTED_OPDATE_PROFILES, opdates_for_profile


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reforecast climatology table keyed to actual benchmark issue dates.")
    parser.add_argument("--config-data", type=str, default="codex_v2/configs/data_v2.yaml")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument(
        "--opdate-profile",
        choices=sorted(SUPPORTED_OPDATE_PROFILES),
        default=PROFILE_TEN_DAY_DEC1_APR30,
    )
    parser.add_argument("--max-lead-day", type=int, default=25)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def _resolve_issue_month_days(weather_dir: Path, years: List[int], opdate_profile: str) -> List[str]:
    issue_days = set()
    for year in sorted({int(y) for y in years}):
        weather_year = load_weather_year(Path(weather_dir) / f"s2s_district_daily_{year}.parquet", weather_cols=[])
        for op_label in opdates_for_profile(opdate_profile):
            sel = pick_issue_date_for_operational_label(
                weather_year_df=weather_year,
                season_year=int(year),
                operational_label=str(op_label),
            )
            issue_days.add(sel.issue_date.strftime("%m-%d"))
    return sorted(issue_days)


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(Path(args.config_data))
    weather_dir = Path(cfg["paths"]["weather_dir"])
    years = [int(y) for y in args.years]
    issue_month_days = _resolve_issue_month_days(
        weather_dir=weather_dir,
        years=years,
        opdate_profile=str(args.opdate_profile),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    climatology = build_reforecast_climatology_frame(
        weather_dir=weather_dir,
        years=years,
        issue_month_days=issue_month_days,
        max_lead_day=int(args.max_lead_day),
        progress_callback=lambda msg: print(msg, flush=True),
    )
    climatology.to_parquet(out_path, index=False)

    print(
        json.dumps(
            {
                "out": str(out_path),
                "n_rows": int(len(climatology)),
                "n_issue_month_days": int(len(issue_month_days)),
                "issue_month_days": issue_month_days,
                "max_lead_day": int(args.max_lead_day),
                "years": years,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
