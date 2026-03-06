from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from codex_v2.src.data.build_dataset_v2 import pick_issue_date_for_operational_label
from codex_v2.src.data.opdate_profiles_v2 import build_five_day_dec1_apr30_labels


def test_five_day_dec1_apr30_profile_labels_exact() -> None:
    labels = build_five_day_dec1_apr30_labels()
    expected = [
        "12-01", "12-06", "12-11", "12-16", "12-21", "12-26", "12-31",
        "01-05", "01-10", "01-15", "01-20", "01-25", "01-30",
        "02-04", "02-09", "02-14", "02-19", "02-24",
        "03-01", "03-06", "03-11", "03-16", "03-21", "03-26", "03-31",
        "04-05", "04-10", "04-15", "04-20", "04-25", "04-30",
    ]
    assert labels == expected
    assert len(labels) == 31


def test_issue_date_mapping_valid_for_profile_2017_2022() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "codex_v2" / "configs" / "data_v2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    weather_dir = Path(cfg["paths"]["weather_dir"])

    labels = build_five_day_dec1_apr30_labels()
    for year in range(2017, 2023):
        w_path = weather_dir / f"s2s_district_daily_{year}.parquet"
        if not w_path.exists():
            pytest.skip(f"Missing weather file for integration test: {w_path}")

        df = pd.read_parquet(w_path, columns=["issue_date"]).drop_duplicates()
        if df.empty:
            pytest.skip(f"No issue_date rows in weather file: {w_path}")
        weather_year_df = pd.DataFrame({"issue_date": pd.to_datetime(df["issue_date"])})

        for label in labels:
            sel = pick_issue_date_for_operational_label(
                weather_year_df=weather_year_df,
                season_year=year,
                operational_label=label,
            )
            assert sel.issue_date <= sel.target_date, (year, label, sel.issue_date, sel.target_date)
