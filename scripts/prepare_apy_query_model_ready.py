#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class YearCols:
    season_label: str
    area_col: str
    production_col: str
    yield_col: str


DISTRICT_ALIASES: Dict[str, str] = {
    "charki dadri": "charkhi dadri",
    "hisar": "hissar",
    "sonipat": "sonepat",
    "yamunanagar": "yamuna nagar",
    "firozepur": "firozpur",
    "nawanshahr": "nawan shehar",
    "s a s nagar": "mohali",
    "s a s nagar sahibzada ajit singh nagar": "mohali",
    "s a s nagar mohal": "mohali",
    "sas nagar": "mohali",
    "s.a.s nagar": "mohali",
    "budaun": "badaun",
    "kanpur nagar": "kanpur",
    "kheri": "lakhimpur kheri",
    "kushi nagar": "kushinagar",
    "mau": "maunathbhanjan",
    "sant kabeer nagar": "sant kabir nagar",
    "sant ravidas nagar": "sant ravi das nagar",
}


def _norm(text: str) -> str:
    x = str(text).strip().lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _canon_district(name: str) -> str:
    n = _norm(name)
    return DISTRICT_ALIASES.get(n, n)


def _flatten_des_table(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        raise RuntimeError("Expected DES export table with multi-level header.")
    cols: List[str] = []
    for top, sub in df.columns:
        t = str(top).strip()
        s = str(sub).strip()
        if t in {"S.No.", "State", "District"}:
            cols.append(t)
            continue
        if "Area" in s:
            metric = "area_ha"
        elif "Production" in s:
            metric = "production_tonnes"
        elif "Yield" in s:
            metric = "yield_ton_per_ha"
        else:
            metric = _norm(s).replace(" ", "_")
        cols.append(f"{t}__{metric}")
    out = df.copy()
    out.columns = cols
    return out


def _extract_year_cols(columns: List[str]) -> List[YearCols]:
    labels = sorted({c.split("__")[0] for c in columns if "__" in c})
    out: List[YearCols] = []
    for y in labels:
        area = f"{y}__area_ha"
        prod = f"{y}__production_tonnes"
        yld = f"{y}__yield_ton_per_ha"
        if not all(c in columns for c in [area, prod, yld]):
            raise RuntimeError(f"Missing expected DES metric columns for season {y}.")
        out.append(YearCols(season_label=y, area_col=area, production_col=prod, yield_col=yld))
    return out


def _parse_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def _long_from_wide(df_wide: pd.DataFrame, years: List[YearCols]) -> pd.DataFrame:
    rows = []
    for y in years:
        tmp = df_wide[["S.No.", "State", "District", y.area_col, y.production_col, y.yield_col]].copy()
        tmp = tmp.rename(
            columns={
                y.area_col: "area_ha",
                y.production_col: "production_tonnes",
                y.yield_col: "yield_ton_per_ha",
            }
        )
        tmp["season_label"] = y.season_label
        m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", y.season_label)
        if not m:
            raise RuntimeError(f"Could not parse season label: {y.season_label}")
        tmp["season_start_year"] = int(m.group(1))
        tmp["season_end_year"] = int(m.group(2))
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    for c in ["area_ha", "production_tonnes", "yield_ton_per_ha"]:
        out[c] = _parse_num(out[c])
    return out


def _merge_malerkotla_into_sangrur(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = df_long.copy()
    x["state_norm"] = x["State"].map(_norm)
    x["district_norm"] = x["District"].map(_canon_district)

    punjab = _norm("Punjab")
    sangrur = _canon_district("Sangrur")
    malerkotla = _canon_district("Malerkotla")

    audit_rows = []
    for season in sorted(x["season_label"].unique()):
        mask_season = x["season_label"] == season
        mask_sang = mask_season & (x["state_norm"] == punjab) & (x["district_norm"] == sangrur)
        mask_mal = mask_season & (x["state_norm"] == punjab) & (x["district_norm"] == malerkotla)

        sang = x.loc[mask_sang]
        mal = x.loc[mask_mal]
        if sang.empty and mal.empty:
            continue

        if sang.empty and not mal.empty:
            # Defensive fallback: if Sangrur is absent, rename Malerkotla to Sangrur.
            x.loc[mask_mal, "District"] = "Sangrur"
            x.loc[mask_mal, "district_norm"] = sangrur
            sang = x.loc[mask_mal]
            mal = pd.DataFrame(columns=x.columns)
            mask_sang = mask_season & (x["state_norm"] == punjab) & (x["district_norm"] == sangrur)

        sang_area_before = float(sang["area_ha"].iloc[0]) if not sang.empty and pd.notna(sang["area_ha"].iloc[0]) else 0.0
        sang_prod_before = (
            float(sang["production_tonnes"].iloc[0])
            if not sang.empty and pd.notna(sang["production_tonnes"].iloc[0])
            else 0.0
        )
        sang_yield_before = (
            float(sang["yield_ton_per_ha"].iloc[0])
            if not sang.empty and pd.notna(sang["yield_ton_per_ha"].iloc[0])
            else float("nan")
        )
        mal_area = float(mal["area_ha"].iloc[0]) if not mal.empty and pd.notna(mal["area_ha"].iloc[0]) else 0.0
        mal_prod = (
            float(mal["production_tonnes"].iloc[0])
            if not mal.empty and pd.notna(mal["production_tonnes"].iloc[0])
            else 0.0
        )

        new_area = sang_area_before + mal_area
        new_prod = sang_prod_before + mal_prod
        # Keep original Sangrur yield when Malerkotla contributes no numeric data.
        if mal_area == 0.0 and mal_prod == 0.0 and pd.notna(sang_yield_before):
            new_yield = sang_yield_before
        else:
            new_yield = (new_prod / new_area) if new_area > 0 else float("nan")

        x.loc[mask_sang, "area_ha"] = new_area
        x.loc[mask_sang, "production_tonnes"] = new_prod
        x.loc[mask_sang, "yield_ton_per_ha"] = new_yield

        audit_rows.append(
            {
                "season_label": season,
                "sangrur_area_before": sang_area_before,
                "sangrur_prod_before": sang_prod_before,
                "sangrur_yield_before_ton_per_ha": sang_yield_before,
                "malerkotla_area_added": mal_area,
                "malerkotla_prod_added": mal_prod,
                "sangrur_area_after": new_area,
                "sangrur_prod_after": new_prod,
                "sangrur_yield_after_ton_per_ha": new_yield,
                "malerkotla_row_present": not mal.empty,
                "malerkotla_numeric_contribution": (mal_area > 0.0 or mal_prod > 0.0),
            }
        )

    # Drop Malerkotla completely after merge.
    x = x[~((x["state_norm"] == punjab) & (x["district_norm"] == malerkotla))].copy()
    x = x.drop(columns=["state_norm", "district_norm"])
    audit = pd.DataFrame(audit_rows).sort_values("season_label").reset_index(drop=True)
    return x, audit


def _map_to_target_119(df_long: pd.DataFrame, districts_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.read_parquet(districts_path)[["district_id", "state_name", "district_name"]].copy()
    target["state_norm"] = target["state_name"].map(_norm)
    target["district_norm"] = target["district_name"].map(_canon_district)

    x = df_long.copy()
    x["state_norm"] = x["State"].map(_norm)
    x["district_norm"] = x["District"].map(_canon_district)

    merged = x.merge(
        target[["district_id", "state_name", "district_name", "state_norm", "district_norm"]],
        on=["state_norm", "district_norm"],
        how="left",
        suffixes=("", "_target"),
    )
    unmatched = merged[merged["district_id"].isna()].copy()
    mapped = merged[merged["district_id"].notna()].copy()
    return mapped, unmatched


def _build_model_ready_panel(mapped: pd.DataFrame, districts_path: Path) -> pd.DataFrame:
    target = pd.read_parquet(districts_path)[["district_id", "state_name", "district_name"]].copy()
    target = target[target["state_name"].isin(["Punjab", "Haryana", "Uttar Pradesh"])].copy()

    seasons = (
        mapped[["season_label", "season_start_year", "season_end_year"]]
        .drop_duplicates()
        .sort_values("season_start_year")
        .reset_index(drop=True)
    )
    target["key"] = 1
    seasons["key"] = 1
    skeleton = target.merge(seasons, on="key", how="inner").drop(columns=["key"])

    use_cols = [
        "district_id",
        "season_label",
        "season_start_year",
        "season_end_year",
        "area_ha",
        "production_tonnes",
        "yield_ton_per_ha",
    ]
    data = (
        mapped[use_cols]
        .drop_duplicates(subset=["district_id", "season_label"], keep="last")
        .copy()
    )

    out = skeleton.merge(
        data,
        on=["district_id", "season_label", "season_start_year", "season_end_year"],
        how="left",
    )
    out["yield_kg_per_ha"] = out["yield_ton_per_ha"] * 1000.0
    out["crop"] = "Wheat"
    out["season"] = "Rabi"
    out["source"] = "DES APY Query Report"
    out["source_file"] = "data/yields/apy_query_report.xls"
    out = out.sort_values(["state_name", "district_name", "season_start_year"]).reset_index(drop=True)
    return out


def run(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_raw = pd.read_html(in_path)[0]
    wide = _flatten_des_table(wide_raw)
    years = _extract_year_cols(list(wide.columns))

    # Save converted raw CSV (same wide shape, parseable, no HTML wrapper).
    raw_wide_csv = out_dir / "apy_query_report_raw_wide.csv"
    wide.to_csv(raw_wide_csv, index=False)

    long_df = _long_from_wide(wide, years)
    merged_long, sangrur_audit = _merge_malerkotla_into_sangrur(long_df)
    mapped, unmatched = _map_to_target_119(merged_long, Path(args.districts_path))
    panel = _build_model_ready_panel(mapped, Path(args.districts_path))

    # Diagnostics.
    miss = panel[panel["yield_ton_per_ha"].isna()][
        ["district_id", "state_name", "district_name", "season_label"]
    ].copy()
    coverage = (
        panel.assign(has_data=panel["yield_ton_per_ha"].notna())
        .groupby(["state_name", "season_label"], as_index=False)["has_data"]
        .sum()
        .rename(columns={"has_data": "districts_with_data"})
    )
    totals = panel[["state_name", "district_id"]].drop_duplicates().groupby("state_name").size()
    coverage["districts_total"] = coverage["state_name"].map(totals)
    coverage["coverage_pct"] = (coverage["districts_with_data"] / coverage["districts_total"]) * 100.0

    # Outputs.
    model_ready_csv = out_dir / "apy_query_report_model_ready_119.csv"
    long_csv = out_dir / "apy_query_report_long_after_merge.csv"
    audit_csv = out_dir / "apy_query_report_sangrur_malerkotla_audit.csv"
    unmatched_csv = out_dir / "apy_query_report_unmatched_after_mapping.csv"
    coverage_csv = out_dir / "apy_query_report_model_ready_coverage.csv"
    missing_csv = out_dir / "apy_query_report_model_ready_missing.csv"
    summary_json = out_dir / "apy_query_report_model_ready_summary.json"

    panel.to_csv(model_ready_csv, index=False)
    merged_long.to_csv(long_csv, index=False)
    sangrur_audit.to_csv(audit_csv, index=False)
    unmatched.to_csv(unmatched_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)
    miss.to_csv(missing_csv, index=False)

    summary = {
        "input_file": str(in_path),
        "raw_rows": int(len(wide)),
        "raw_districts_by_state": wide.groupby("State")["District"].nunique().to_dict(),
        "seasons": [y.season_label for y in years],
        "rows_after_malerkotla_drop": int(len(merged_long)),
        "unique_districts_after_malerkotla_drop": int(
            merged_long[["State", "District"]].drop_duplicates().shape[0]
        ),
        "mapped_rows": int(len(mapped)),
        "unmatched_rows": int(len(unmatched)),
        "panel_rows": int(len(panel)),
        "panel_unique_districts": int(panel["district_id"].nunique()),
        "panel_missing_cells": int(len(miss)),
        "model_ready_file": str(model_ready_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"Converted raw CSV: {raw_wide_csv}")
    print(f"Model-ready CSV: {model_ready_csv}")
    print(f"Sangrur/Malerkotla audit: {audit_csv}")
    print(f"Coverage report: {coverage_csv}")
    print(f"Missing report: {missing_csv}")
    print(f"Summary: {summary_json}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare model-ready 119-district yield panel from DES APY export.")
    p.add_argument(
        "--input",
        type=str,
        default="data/yields/apy_query_report.xls",
        help="DES APY export file (HTML table with .xls extension).",
    )
    p.add_argument(
        "--districts-path",
        type=str,
        default="data/processed/s2s_district/districts.parquet",
        help="Target districts parquet used in this project.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/yields",
        help="Directory for CSV outputs.",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
