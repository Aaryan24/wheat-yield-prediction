#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_RESOURCE_ID = "f4435c3b-96be-4002-9839-aa3897dc732b"
DEFAULT_STATES = ["Punjab", "Haryana", "Uttar Pradesh"]
DEFAULT_DEMO_API_KEY = "579b464db66ec23bdd0000019fc84f43ca52437351b43702f5998234"
DEFAULT_PUNJAB_RESOURCE_PAGE = (
    "https://www.data.gov.in/resource/"
    "district-wise-yeild-under-wheat-cultivation-punjab-1968-2022-april-march"
)


DISTRICT_ALIASES = {
    # Uttar Pradesh renamed districts.
    "prayagraj": "allahabad",
    "ayodhya": "faizabad",
    "bhadohi": "sant ravi das nagar",
    "sant ravidas nagar": "sant ravi das nagar",
    "jyotiba phule nagar": "amroha",
    "mahamaya nagar": "hathras",
    "kanshiram nagar": "kasganj",
    "panchsheel nagar": "hapur",
    "chhatrapati shahuji maharaj nagar": "amethi",
    "bhim nagar": "sambhal",
    # Haryana spelling / historical names.
    "hisar": "hissar",
    "nuh": "mewat",
    "sonipat": "sonepat",
    "yamunanagar": "yamuna nagar",
    # Punjab spelling / historical names.
    "ferozepur": "firozpur",
    "sri muktsar sahib": "muktsar",
    "shahid bhagat singh nagar": "nawan shehar",
    "sas nagar": "mohali",
    "s a s nagar": "mohali",
}


STATE_ALIASES = {
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "haryana": "Haryana",
    "punjab": "Punjab",
}


@dataclass(frozen=True)
class OGDColumnMap:
    state: Optional[str]
    district: Optional[str]
    crop: Optional[str]
    season: Optional[str]
    year: Optional[str]
    area: Optional[str]
    production: Optional[str]
    yield_value: Optional[str]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _normalize_name(text: str) -> str:
    x = str(text).strip().lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\b(dist|district)\b", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _canonical_district(text: str) -> str:
    base = _normalize_name(text)
    return DISTRICT_ALIASES.get(base, base)


def _canonical_state(text: str) -> str:
    base = _normalize_name(text)
    return STATE_ALIASES.get(base, str(text).strip())


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    x = str(value).strip()
    if x == "":
        return float("nan")
    x = x.replace(",", "")
    if x in {"-", "--", "NA", "N/A", "nan", "NaN"}:
        return float("nan")
    try:
        return float(x)
    except ValueError:
        match = re.search(r"-?\d+(\.\d+)?", x)
        if not match:
            return float("nan")
        return float(match.group(0))


def _parse_year(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        year = int(value)
        if 1900 <= year <= 2100:
            return year
    s = str(value)
    match = re.search(r"(19|20)\d{2}", s)
    if not match:
        return None
    return int(match.group(0))


def _http_get_json(url: str, params: Dict[str, object], timeout_s: int = 120) -> dict:
    query = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query}"
    req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_text(url: str, timeout_s: int = 120) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _first_existing(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    lookup = {_normalize_key(c): c for c in columns}
    for candidate in candidates:
        key = _normalize_key(candidate)
        if key in lookup:
            return lookup[key]
    return None


def _detect_ogd_columns(sample_records: List[dict]) -> OGDColumnMap:
    if not sample_records:
        raise ValueError("No sample records available to detect OGD columns.")
    cols = list(sample_records[0].keys())
    return OGDColumnMap(
        state=_first_existing(cols, ["state_name", "state", "state"]),
        district=_first_existing(cols, ["district_name", "district", "districts"]),
        crop=_first_existing(cols, ["crop", "crop_name"]),
        season=_first_existing(cols, ["season", "season_name"]),
        year=_first_existing(cols, ["crop_year", "year", "agri_year", "agricultural_year"]),
        area=_first_existing(cols, ["area", "area_ha", "area_hectare"]),
        production=_first_existing(cols, ["production", "prod", "production_tonnes"]),
        yield_value=_first_existing(cols, ["yield", "yeild", "yield_kg_per_ha"]),
    )


def _fetch_paginated(
    endpoint: str,
    base_params: Dict[str, object],
    limit: int,
    max_pages: int,
) -> List[dict]:
    rows: List[dict] = []
    offset = 0
    page = 0
    while page < max_pages:
        params = dict(base_params)
        params["offset"] = offset
        params["limit"] = limit
        payload = _http_get_json(endpoint, params=params)
        recs = payload.get("records", [])
        if not recs:
            break
        rows.extend(recs)
        if len(recs) < limit:
            break
        offset += limit
        page += 1
    return rows


def fetch_ogd_records(
    resource_id: str,
    api_key: str,
    states: List[str],
    crop_keyword: str,
    limit: int,
    max_pages: int,
) -> Tuple[pd.DataFrame, OGDColumnMap]:
    endpoint = f"https://api.data.gov.in/resource/{resource_id}"
    sample = _http_get_json(
        endpoint,
        params={
            "api-key": api_key,
            "format": "json",
            "limit": 3,
            "offset": 0,
        },
    ).get("records", [])
    if not sample:
        raise RuntimeError(
            "OGD API returned no records. Check resource ID/API key "
            "or verify the resource has API access."
        )
    col_map = _detect_ogd_columns(sample)

    if not col_map.year or not col_map.state or not col_map.district:
        raise RuntimeError(
            "Could not detect required columns (state/district/year) in OGD response. "
            "Please inspect the resource schema manually."
        )

    base = {
        "api-key": api_key,
        "format": "json",
    }
    if col_map.crop:
        base[f"filters[{col_map.crop}]"] = crop_keyword

    records: List[dict] = []
    for state in states:
        state_params = dict(base)
        state_params[f"filters[{col_map.state}]"] = state
        chunk = _fetch_paginated(
            endpoint=endpoint,
            base_params=state_params,
            limit=limit,
            max_pages=max_pages,
        )
        _log(f"Fetched {len(chunk):,} rows from OGD for state={state}.")
        records.extend(chunk)

    if not records:
        # Filter syntax can fail if the schema uses unexpected keys; fetch full and filter locally.
        _log("State/crop filters returned no rows; falling back to unfiltered OGD pull.")
        records = _fetch_paginated(
            endpoint=endpoint,
            base_params=base,
            limit=limit,
            max_pages=max_pages,
        )
        _log(f"Fetched {len(records):,} rows from unfiltered OGD pull.")

    return pd.DataFrame(records), col_map


def _normalize_yield_series(values: pd.Series) -> pd.Series:
    out = values.astype(float)
    med = out.dropna().median()
    if pd.isna(med):
        return out
    # Likely quintal/ha.
    if 5 <= med <= 100:
        return out * 100.0
    # Likely tonnes/ha.
    if 0.2 <= med < 10:
        return out * 1000.0
    # Assume already kg/ha.
    return out


def prepare_ogd_frame(
    df: pd.DataFrame,
    col_map: OGDColumnMap,
    states: List[str],
    crop_keyword: str,
    start_year: int,
    resource_id: str,
) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for src, dst in [
        (col_map.state, "state_name_raw"),
        (col_map.district, "district_name_raw"),
        (col_map.crop, "crop_raw"),
        (col_map.season, "season_raw"),
        (col_map.year, "year_raw"),
        (col_map.area, "area_raw"),
        (col_map.production, "production_raw"),
        (col_map.yield_value, "yield_raw"),
    ]:
        if src:
            rename_map[src] = dst
    out = df.rename(columns=rename_map).copy()

    required = ["state_name_raw", "district_name_raw", "year_raw"]
    for col in required:
        if col not in out.columns:
            raise RuntimeError(f"Required column missing after rename: {col}")

    out["state_name_raw"] = out["state_name_raw"].astype(str).str.strip()
    out["district_name_raw"] = out["district_name_raw"].astype(str).str.strip()
    out["state_name"] = out["state_name_raw"].map(_canonical_state)
    out["state_norm"] = out["state_name"].map(_normalize_name)
    out["district_norm"] = out["district_name_raw"].map(_canonical_district)
    out["year"] = out["year_raw"].map(_parse_year)
    out = out[out["year"].notna()].copy()
    out["year"] = out["year"].astype(int)
    out = out[out["year"] >= start_year].copy()

    target_state_norms = {_normalize_name(x) for x in states}
    out = out[out["state_norm"].isin(target_state_norms)].copy()

    if "crop_raw" in out.columns:
        out["crop_raw"] = out["crop_raw"].astype(str)
        out = out[out["crop_raw"].str.contains(crop_keyword, case=False, na=False)].copy()

    out["area_ha"] = out["area_raw"].map(_to_float) if "area_raw" in out.columns else float("nan")
    out["production_tonnes"] = (
        out["production_raw"].map(_to_float) if "production_raw" in out.columns else float("nan")
    )
    if "yield_raw" in out.columns:
        y_raw = out["yield_raw"].map(_to_float)
        out["yield_kg_per_ha"] = _normalize_yield_series(y_raw)
        out["yield_source"] = "reported"
    else:
        out["yield_kg_per_ha"] = float("nan")
        out["yield_source"] = "derived"

    missing_yield = out["yield_kg_per_ha"].isna()
    if missing_yield.any():
        ratio = out["production_tonnes"] / out["area_ha"]
        out.loc[missing_yield, "yield_kg_per_ha"] = _normalize_yield_series(ratio)[missing_yield]
        out.loc[missing_yield, "yield_source"] = "derived"

    out = out[out["yield_kg_per_ha"].notna()].copy()
    out = out[out["yield_kg_per_ha"] > 0].copy()

    out["season"] = out["season_raw"].astype(str) if "season_raw" in out.columns else ""
    out["source_dataset"] = f"ogd_api:{resource_id}"
    out["source_url"] = f"https://api.data.gov.in/resource/{resource_id}"
    return out[
        [
            "state_name",
            "state_norm",
            "district_name_raw",
            "district_norm",
            "year",
            "season",
            "area_ha",
            "production_tonnes",
            "yield_kg_per_ha",
            "yield_source",
            "source_dataset",
            "source_url",
        ]
    ].copy()


def _season_score(season: str) -> int:
    s = _normalize_name(season)
    if "rabi" in s:
        return 4
    if "winter" in s:
        return 3
    if "whole" in s or "annual" in s:
        return 2
    if s == "":
        return 1
    return 0


def _extract_csv_url_from_resource_page(resource_page_url: str) -> Optional[str]:
    html = _http_get_text(resource_page_url)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    if not hrefs:
        return None
    scored: List[Tuple[int, str]] = []
    for href in hrefs:
        href_l = href.lower()
        score = 0
        if ".csv" in href_l:
            score += 4
        if "format=csv" in href_l:
            score += 3
        if "/storage/" in href_l:
            score += 2
        if "download" in href_l:
            score += 1
        if score > 0:
            scored.append((score, href))
    if not scored:
        return None
    best = sorted(scored, key=lambda x: x[0], reverse=True)[0][1]
    return urllib.parse.urljoin(resource_page_url, best)


def _parse_state_csv_wide(
    csv_url: str,
    state_name: str,
    start_year: int,
    source_tag: str,
) -> pd.DataFrame:
    raw = pd.read_csv(csv_url)
    if raw.empty:
        return pd.DataFrame()

    district_col = raw.columns[0]
    for col in raw.columns:
        if raw[col].dtype == "object" and raw[col].nunique(dropna=True) > 8:
            district_col = col
            break

    year_cols = [c for c in raw.columns if re.search(r"(19|20)\d{2}", str(c))]
    if not year_cols:
        return pd.DataFrame()

    long = raw[[district_col] + year_cols].rename(columns={district_col: "district_name_raw"})
    long = long.melt(id_vars=["district_name_raw"], var_name="year_raw", value_name="yield_raw")
    long["year"] = long["year_raw"].map(_parse_year)
    long = long[long["year"].notna()].copy()
    long["year"] = long["year"].astype(int)
    long = long[long["year"] >= start_year].copy()
    long["yield_kg_per_ha"] = _normalize_yield_series(long["yield_raw"].map(_to_float))
    long = long[long["yield_kg_per_ha"].notna()].copy()
    long = long[long["yield_kg_per_ha"] > 0].copy()

    long["state_name"] = state_name
    long["state_norm"] = long["state_name"].map(_normalize_name)
    long["district_norm"] = long["district_name_raw"].map(_canonical_district)
    long["season"] = ""
    long["area_ha"] = float("nan")
    long["production_tonnes"] = float("nan")
    long["yield_source"] = "reported"
    long["source_dataset"] = source_tag
    long["source_url"] = csv_url
    return long[
        [
            "state_name",
            "state_norm",
            "district_name_raw",
            "district_norm",
            "year",
            "season",
            "area_ha",
            "production_tonnes",
            "yield_kg_per_ha",
            "yield_source",
            "source_dataset",
            "source_url",
        ]
    ].copy()


def _load_target_districts(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = ["district_id", "state_name", "district_name"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"District table missing columns: {missing}")
    out = df[needed].copy()
    out["state_norm"] = out["state_name"].map(_normalize_name)
    out["district_norm"] = out["district_name"].map(_canonical_district)
    return out


def _match_to_target(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = source_df.copy().reset_index(drop=False).rename(columns={"index": "_src_index"})
    base = base.merge(
        target_df[["district_id", "state_name", "district_name", "state_norm", "district_norm"]],
        on=["state_norm", "district_norm"],
        how="left",
        suffixes=("", "_target"),
    )
    unmatched = base[base["district_id"].isna()].copy()
    if unmatched.empty:
        base = base.drop(columns=["_src_index"])
        return base, unmatched

    state_options: Dict[str, List[str]] = (
        target_df.groupby("state_norm")["district_norm"].apply(lambda s: sorted(set(s))).to_dict()
    )
    norm_to_row: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    for row in target_df.itertuples(index=False):
        norm_to_row[(row.state_norm, row.district_norm)] = (
            row.district_id,
            row.state_name,
            row.district_name,
        )

    fill_rows = []
    for row in unmatched.itertuples(index=False):
        choices = state_options.get(row.state_norm, [])
        if not choices:
            continue
        guess = difflib.get_close_matches(row.district_norm, choices, n=1, cutoff=0.92)
        if not guess:
            continue
        best = guess[0]
        district_id, state_name, district_name = norm_to_row[(row.state_norm, best)]
        fill_rows.append(
            {
                "row_index": row._src_index,
                "district_id": district_id,
                "state_name": state_name,
                "district_name": district_name,
                "district_norm": best,
            }
        )

    if fill_rows:
        fill_df = pd.DataFrame(fill_rows)
        for rec in fill_df.itertuples(index=False):
            idx = int(rec.row_index)
            mask = base["_src_index"] == idx
            base.loc[mask, "district_id"] = rec.district_id
            base.loc[mask, "state_name"] = rec.state_name
            base.loc[mask, "district_name"] = rec.district_name
            base.loc[mask, "district_norm"] = rec.district_norm

    unmatched_after = base[base["district_id"].isna()].copy()
    unmatched_after = unmatched_after.drop(columns=["_src_index"], errors="ignore")
    base = base.drop(columns=["_src_index"])
    return base, unmatched_after


def _select_best_record(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["season_score"] = x["season"].map(_season_score)
    x = x.sort_values(
        [
            "district_id",
            "year",
            "season_score",
            "area_ha",
            "yield_source",
            "yield_kg_per_ha",
        ],
        ascending=[True, True, False, False, True, False],
    )
    x = x.drop_duplicates(subset=["district_id", "year"], keep="first")
    return x


def _parse_state_assignment(items: List[str], kind: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"{kind} entry must be 'State=URL'. Got: {item}")
        state, url = item.split("=", 1)
        out.append((state.strip(), url.strip()))
    return out


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = _load_target_districts(Path(args.districts_path))
    target = target[target["state_name"].isin(args.states)].copy()
    _log(f"Loaded {len(target)} target districts from {args.districts_path}.")

    api_key = args.api_key or DEFAULT_DEMO_API_KEY
    if not args.api_key:
        _log("Using the public demo OGD API key. For stable runs, set --api-key or OGD_API_KEY.")

    ogd_raw, col_map = fetch_ogd_records(
        resource_id=args.resource_id,
        api_key=api_key,
        states=args.states,
        crop_keyword=args.crop_keyword,
        limit=args.limit,
        max_pages=args.max_pages,
    )
    ogd_long = prepare_ogd_frame(
        df=ogd_raw,
        col_map=col_map,
        states=args.states,
        crop_keyword=args.crop_keyword,
        start_year=args.start_year,
        resource_id=args.resource_id,
    )
    _log(f"Prepared {len(ogd_long):,} OGD wheat rows after filtering.")

    supplemental_frames: List[pd.DataFrame] = []
    resource_assignments = _parse_state_assignment(args.supplemental_resource_page, "resource page")
    csv_assignments = _parse_state_assignment(args.supplemental_csv, "CSV")

    if args.use_default_punjab_resource:
        resource_assignments.append(("Punjab", DEFAULT_PUNJAB_RESOURCE_PAGE))

    for state_name, resource_page in resource_assignments:
        try:
            csv_url = _extract_csv_url_from_resource_page(resource_page)
            if not csv_url:
                _log(f"Could not find CSV link in resource page: {resource_page}")
                continue
            source_tag = f"ogd_resource_page:{resource_page}"
            sup = _parse_state_csv_wide(
                csv_url=csv_url,
                state_name=state_name,
                start_year=args.start_year,
                source_tag=source_tag,
            )
            if not sup.empty:
                _log(f"Added {len(sup):,} supplemental rows from {resource_page}.")
                supplemental_frames.append(sup)
        except Exception as exc:
            _log(f"Supplemental resource page failed ({resource_page}): {exc}")

    for state_name, csv_url in csv_assignments:
        try:
            source_tag = f"direct_csv:{csv_url}"
            sup = _parse_state_csv_wide(
                csv_url=csv_url,
                state_name=state_name,
                start_year=args.start_year,
                source_tag=source_tag,
            )
            if not sup.empty:
                _log(f"Added {len(sup):,} supplemental rows from direct CSV.")
                supplemental_frames.append(sup)
        except Exception as exc:
            _log(f"Supplemental CSV failed ({csv_url}): {exc}")

    combined = pd.concat([ogd_long] + supplemental_frames, ignore_index=True)
    matched, unmatched = _match_to_target(combined, target)
    matched = matched[matched["district_id"].notna()].copy()
    selected = _select_best_record(matched)

    if selected.empty:
        raise RuntimeError("No matched district-year yield rows were produced.")

    latest_year = int(selected["year"].max())
    years = list(range(args.start_year, latest_year + 1))
    skeleton = (
        target[["district_id", "state_name", "district_name"]]
        .assign(key=1)
        .merge(pd.DataFrame({"year": years, "key": 1}), on="key", how="inner")
        .drop(columns=["key"])
    )
    panel = skeleton.merge(
        selected[
            [
                "district_id",
                "year",
                "yield_kg_per_ha",
                "season",
                "area_ha",
                "production_tonnes",
                "yield_source",
                "source_dataset",
                "source_url",
                "district_name_raw",
            ]
        ],
        on=["district_id", "year"],
        how="left",
    )
    panel = panel.sort_values(["state_name", "district_name", "year"]).reset_index(drop=True)

    missing = panel[panel["yield_kg_per_ha"].isna()][
        ["district_id", "state_name", "district_name", "year"]
    ].copy()
    coverage = (
        panel.assign(has_data=panel["yield_kg_per_ha"].notna())
        .groupby(["state_name", "year"], as_index=False)["has_data"]
        .sum()
        .rename(columns={"has_data": "districts_with_data"})
    )
    coverage["districts_total"] = coverage["state_name"].map(target["state_name"].value_counts())
    coverage["coverage_pct"] = (coverage["districts_with_data"] / coverage["districts_total"]) * 100.0

    panel_path = out_dir / f"wheat_yield_119_districts_{args.start_year}_{latest_year}.csv"
    missing_path = out_dir / f"wheat_yield_119_missing_{args.start_year}_{latest_year}.csv"
    coverage_path = out_dir / f"wheat_yield_119_coverage_{args.start_year}_{latest_year}.csv"
    unmatched_path = out_dir / f"wheat_yield_119_unmatched_raw_{args.start_year}_{latest_year}.csv"
    meta_path = out_dir / f"wheat_yield_119_metadata_{args.start_year}_{latest_year}.json"

    panel.to_csv(panel_path, index=False)
    missing.to_csv(missing_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    if not unmatched.empty:
        unmatched.to_csv(unmatched_path, index=False)

    metadata = {
        "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "start_year": args.start_year,
        "latest_year_in_output": latest_year,
        "target_district_count": int(target["district_id"].nunique()),
        "records_selected": int(len(selected)),
        "missing_district_years": int(len(missing)),
        "ogd_resource_id": args.resource_id,
        "ogd_api_endpoint": f"https://api.data.gov.in/resource/{args.resource_id}",
        "supplemental_resource_pages": [x[1] for x in resource_assignments],
        "supplemental_csvs": [x[1] for x in csv_assignments],
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    _log("")
    _log("Done.")
    _log(f"Panel: {panel_path}")
    _log(f"Coverage: {coverage_path}")
    _log(f"Missing: {missing_path}")
    if not unmatched.empty:
        _log(f"Unmatched raw rows: {unmatched_path}")
    _log(f"Metadata: {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download/assemble wheat yield data for the 119 project districts "
            "(Punjab, Haryana, Uttar Pradesh) from 2017 to latest available year."
        )
    )
    parser.add_argument(
        "--districts-path",
        type=str,
        default="data/processed/s2s_district/districts.parquet",
        help="Path to target district table (must contain district_id/state_name/district_name).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/yields",
        help="Output directory for final panel + diagnostics.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2017,
        help="First year to keep in the final panel.",
    )
    parser.add_argument(
        "--resource-id",
        type=str,
        default=DEFAULT_RESOURCE_ID,
        help="OGD API resource ID for district crop statistics.",
    )
    parser.add_argument(
        "--crop-keyword",
        type=str,
        default="wheat",
        help="Crop keyword filter (case-insensitive).",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=DEFAULT_STATES,
        help="States to include.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OGD API key. If omitted, uses OGD_API_KEY env var, then public demo key.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Page size for OGD API pagination.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=2000,
        help="Safety cap on number of pages per API pull.",
    )
    parser.add_argument(
        "--supplemental-resource-page",
        action="append",
        default=[],
        metavar="STATE=URL",
        help=(
            "Optional state resource page URL(s). Script tries to extract CSV and use it "
            "as supplemental/override yield source."
        ),
    )
    parser.add_argument(
        "--supplemental-csv",
        action="append",
        default=[],
        metavar="STATE=URL",
        help="Optional direct CSV URL(s) in STATE=URL form.",
    )
    parser.add_argument(
        "--use-default-punjab-resource",
        action="store_true",
        default=False,
        help=(
            "Try Punjab OGD resource page (1968-2022) as supplemental source. "
            "Useful for extending the latest year for Punjab."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.api_key is None:
        env_key = None
        try:
            import os

            env_key = os.environ.get("OGD_API_KEY")
        except Exception:
            env_key = None
        args.api_key = env_key
    run(args)
