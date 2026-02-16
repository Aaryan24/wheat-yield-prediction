from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from codex_v2.src.data.feature_engineering_v2 import (
    SAT_BASE_FEATURES,
    WEATHER_BASE_FEATURES,
    add_missingness_indicators,
    engineer_satellite_features,
    engineer_weather_features,
)
from codex_v2.src.data.satellite_alignment_v2 import (
    canon_district,
    coverage_rows_for_sample,
    impute_satellite_by_state_mean,
    load_sangrur_malerkotla_weights,
    load_sangrur_malerkotla_weights_rows,
    load_satellite_merged,
    norm,
    sat_step_active_mask,
    satellite_tensor_for_season,
)


STATE_ALIASES = {
    "uttar pradesh": "Uttar Pradesh",
    "uttar_pradesh": "Uttar Pradesh",
    "haryana": "Haryana",
    "punjab": "Punjab",
}


def _load_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _canon_state(text: str) -> str:
    n = norm(text)
    return STATE_ALIASES.get(n, str(text).strip())


def parse_operational_target_date(label: str, season_year: int) -> pd.Timestamp:
    m = re.match(r"^\s*(\d{2})[-/](\d{2})\s*$", str(label))
    if not m:
        raise ValueError(f"Operational date label must be MM-DD. Got: {label}")
    month = int(m.group(1))
    day = int(m.group(2))
    year = season_year if month >= 9 else season_year + 1
    return pd.Timestamp(dt.date(year, month, day))


@dataclass(frozen=True)
class SeasonSelection:
    season_year: int
    operational_label: str
    target_date: pd.Timestamp
    issue_date: pd.Timestamp


@dataclass
class DatasetBundle:
    weather_x: np.ndarray
    weather_mask: np.ndarray
    sat_x: np.ndarray
    sat_mask: np.ndarray
    y_raw: np.ndarray
    y_target: np.ndarray
    sample_years: np.ndarray
    sample_operational_dates: np.ndarray
    sample_issue_dates: np.ndarray
    sample_opdate_idx: np.ndarray
    district_df: pd.DataFrame
    adjacency: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_years: List[int]
    val_years: List[int]
    test_years: List[int]
    target_mode: str
    target_mean: np.ndarray
    target_std: np.ndarray
    weather_feature_names: List[str]
    sat_feature_names: List[str]
    coverage_report: pd.DataFrame
    sangrur_weights_report: pd.DataFrame
    config_resolved: Dict[str, object]

    def inverse_target_array(self, arr: np.ndarray) -> np.ndarray:
        if self.target_mode == "raw":
            return arr.astype(np.float32)
        if self.target_mode == "district_demeaned":
            return (arr + self.target_mean[None, :]).astype(np.float32)
        if self.target_mode == "district_zscore":
            return (arr * self.target_std[None, :] + self.target_mean[None, :]).astype(np.float32)
        raise ValueError(f"Unsupported target_mode={self.target_mode}")


def load_district_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["district_id", "state_name", "district_name", "district_index"]].copy()
    df = df.sort_values("district_index").reset_index(drop=True)
    df["state_norm"] = df["state_name"].map(norm)
    df["district_norm"] = df["district_name"].map(canon_district)
    return df


def load_yield_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["district_id", "season_start_year", "yield_kg_per_ha", "area_ha"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Yield file missing required columns: {missing}")
    out = df[needed].copy()
    out["district_id"] = out["district_id"].astype(str)
    out["season_start_year"] = out["season_start_year"].astype(int)
    out["yield_kg_per_ha"] = pd.to_numeric(out["yield_kg_per_ha"], errors="coerce")
    return out


def load_weather_year(path: Path, weather_cols: Sequence[str]) -> pd.DataFrame:
    keep = ["district_id", "issue_date", "lead_day"] + list(weather_cols)
    df = pd.read_parquet(path, columns=keep)
    df["district_id"] = df["district_id"].astype(str)
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    return df


def pick_issue_date_for_operational_label(
    weather_year_df: pd.DataFrame,
    season_year: int,
    operational_label: str,
) -> SeasonSelection:
    issues = pd.to_datetime(sorted(weather_year_df["issue_date"].unique()))
    target_date = parse_operational_target_date(operational_label, season_year)
    valid = issues[issues <= target_date]
    issue_date = pd.Timestamp(valid[-1]) if len(valid) > 0 else pd.Timestamp(issues[0])
    return SeasonSelection(
        season_year=season_year,
        operational_label=operational_label,
        target_date=target_date,
        issue_date=issue_date,
    )


def weather_tensor_for_season(
    weather_df: pd.DataFrame,
    district_ids: Sequence[str],
    issue_date: pd.Timestamp,
    horizon_days: int,
    weather_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(district_ids)
    f = len(weather_cols)
    x = np.zeros((n, horizon_days, f), dtype=np.float32)
    m = np.zeros((n, horizon_days), dtype=np.float32)

    sub = weather_df[weather_df["issue_date"] == issue_date].copy()
    sub = sub[sub["lead_day"] <= horizon_days]

    for i, did in enumerate(district_ids):
        rows = sub[sub["district_id"] == did].sort_values("lead_day")
        if rows.empty:
            continue
        lead_map = {int(r.lead_day): r for r in rows.itertuples(index=False)}
        for lead in range(1, horizon_days + 1):
            row = lead_map.get(lead)
            if row is None:
                continue
            vals = [float(getattr(row, c)) for c in weather_cols]
            arr = np.array(vals, dtype=np.float32)
            if np.isnan(arr).any():
                continue
            x[i, lead - 1, :] = arr
            m[i, lead - 1] = 1.0

    return x, m


def build_adjacency_from_boundaries(
    district_df: pd.DataFrame,
    boundary_config_path: Path,
) -> np.ndarray:
    cfg = _load_yaml(boundary_config_path)
    bcfg = cfg["boundaries"]

    path = Path(bcfg["admin2_path"])
    layer = bcfg["admin2_layer"]
    state_field = bcfg["state_field"]
    district_field = bcfg["district_field"]
    country_field = bcfg["country_field"]
    country_iso3 = bcfg["country_iso3"]

    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")

    gdf = gpd.read_file(path, layer=layer)
    gdf = gdf[gdf[country_field] == country_iso3].copy()
    gdf["state_name"] = gdf[state_field].astype(str)
    gdf["district_name"] = gdf[district_field].astype(str)
    gdf = gdf[gdf["state_name"].isin(["Punjab", "Haryana", "Uttar Pradesh"])].copy()
    gdf["state_norm"] = gdf["state_name"].map(norm)
    gdf["district_norm"] = gdf["district_name"].map(canon_district)

    lookup = {
        (r.state_norm, r.district_norm): r.district_id
        for r in district_df.itertuples(index=False)
    }
    gdf["district_id"] = [lookup.get((s, d), None) for s, d in zip(gdf["state_norm"], gdf["district_norm"])]
    gdf = gdf[gdf["district_id"].notna()].copy()
    gdf = gdf.drop_duplicates(subset=["district_id"]).copy().set_index("district_id")

    ids = district_df["district_id"].astype(str).tolist()
    n = len(ids)
    adj = np.eye(n, dtype=np.float32)

    for i in range(n):
        id_i = ids[i]
        if id_i not in gdf.index:
            continue
        gi = gdf.loc[id_i].geometry
        for j in range(i + 1, n):
            id_j = ids[j]
            if id_j not in gdf.index:
                continue
            gj = gdf.loc[id_j].geometry
            if gi.touches(gj) or gi.intersects(gj):
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Fallback: if a node has only self-loop, connect it to nearest districts in same state.
    deg = adj.sum(axis=1)
    isolated = np.where(deg <= 1.0)[0]
    if len(isolated) > 0:
        centroids = {}
        for did in ids:
            if did in gdf.index:
                centroids[did] = gdf.loc[did].geometry.centroid
        for idx in isolated:
            did = ids[idx]
            state = district_df.iloc[idx]["state_name"]
            if did not in centroids:
                continue
            candidates = [
                (j, centroids[ids[j]].distance(centroids[did]))
                for j in range(n)
                if j != idx
                and district_df.iloc[j]["state_name"] == state
                and ids[j] in centroids
            ]
            candidates = sorted(candidates, key=lambda x: x[1])[:3]
            for j, _ in candidates:
                adj[idx, j] = 1.0
                adj[j, idx] = 1.0

    return adj.astype(np.float32)


def fit_masked_scaler(
    values: np.ndarray,
    mask: np.ndarray,
    sample_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # values: [S, N, T, F], mask: [S, N, T], sample_idx: [K]
    if len(sample_idx) == 0:
        mean = np.zeros(values.shape[-1], dtype=np.float32)
        std = np.ones(values.shape[-1], dtype=np.float32)
        return mean, std

    sub_v = values[sample_idx]
    sub_m = mask[sample_idx].astype(bool)
    rep = np.repeat(sub_m[..., None], sub_v.shape[-1], axis=-1)
    flat = sub_v[rep].reshape(-1, sub_v.shape[-1])
    if flat.size == 0:
        mean = np.zeros(values.shape[-1], dtype=np.float32)
        std = np.ones(values.shape[-1], dtype=np.float32)
        return mean, std

    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)


def compute_target_stats(
    y_raw: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # y_raw shape [S, N]
    if len(train_idx) == 0:
        raise RuntimeError("Empty train_idx. Cannot compute target stats.")
    base = y_raw[train_idx]
    mean = base.mean(axis=0).astype(np.float32)
    std = base.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_target_transform(
    y_raw: np.ndarray,
    target_mode: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    mode = str(target_mode).strip().lower()
    if mode == "raw":
        return y_raw.astype(np.float32)
    if mode == "district_demeaned":
        return (y_raw - target_mean[None, :]).astype(np.float32)
    if mode == "district_zscore":
        return ((y_raw - target_mean[None, :]) / target_std[None, :]).astype(np.float32)
    raise ValueError(f"Unsupported target_mode={target_mode}")


def _resolve_operational_dates(cfg: dict, operational_dates: Optional[Sequence[str]]) -> List[str]:
    if operational_dates:
        return [str(x) for x in operational_dates]
    op_cfg = cfg.get("operational_dates", {})
    primary = op_cfg.get("primary", [])
    secondary = op_cfg.get("secondary", [])
    merged = [str(x) for x in primary + secondary]
    if not merged:
        raise RuntimeError("No operational dates provided and no defaults in config.")
    return merged


def _resolve_years(cfg: dict, yield_df: pd.DataFrame) -> List[int]:
    years = cfg.get("years")
    if years:
        return sorted([int(y) for y in years])
    return sorted(yield_df["season_start_year"].astype(int).unique().tolist())


def _resolve_split(cfg: dict) -> Tuple[List[int], List[int], List[int]]:
    splits = cfg.get("splits", {})
    train_years = [int(y) for y in splits.get("train_years", [])]
    val_years = [int(y) for y in splits.get("val_years", [])]
    test_years = [int(y) for y in splits.get("test_years", [])]
    if not train_years or not val_years or not test_years:
        raise RuntimeError("Split years missing in config: expected train_years/val_years/test_years.")
    return train_years, val_years, test_years


def build_dataset_v2(
    data_config_path: Path,
    mode: str,
    target_mode: str,
    horizon_days: int,
    operational_dates: Optional[Sequence[str]] = None,
    apply_sat_mask_fix: bool = True,
    use_engineered_weather: bool = True,
    use_engineered_satellite: bool = True,
    use_missingness_indicators: bool = True,
    sat_seq_len: Optional[int] = None,
) -> DatasetBundle:
    cfg = _load_yaml(Path(data_config_path))
    paths = cfg["paths"]

    district_df = load_district_table(Path(paths["districts"]))
    yield_df = load_yield_panel(Path(paths["yield_file"]))
    sat_map = load_satellite_merged(Path(paths["sat_merged_dir"]))
    sangrur_weights = load_sangrur_malerkotla_weights(Path(paths["sangrur_audit"]))
    sangrur_weights_report = load_sangrur_malerkotla_weights_rows(Path(paths["sangrur_audit"]))

    train_years, val_years, test_years = _resolve_split(cfg)
    years = _resolve_years(cfg, yield_df)
    op_dates = _resolve_operational_dates(cfg, operational_dates)

    seq_len = int(sat_seq_len or cfg.get("satellite", {}).get("seq_len", 43))
    district_ids = district_df["district_id"].astype(str).tolist()

    weather_by_year: Dict[int, pd.DataFrame] = {}
    for year in years:
        w_path = Path(paths["weather_dir"]) / f"s2s_district_daily_{year}.parquet"
        if not w_path.exists():
            raise FileNotFoundError(f"Missing weather file: {w_path}")
        weather_by_year[year] = load_weather_year(w_path, WEATHER_BASE_FEATURES)

    weather_list: List[np.ndarray] = []
    weather_mask_list: List[np.ndarray] = []
    sat_list: List[np.ndarray] = []
    sat_mask_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    sample_years: List[int] = []
    sample_opdates: List[str] = []
    sample_issue_dates: List[str] = []
    coverage_rows: List[dict] = []

    # Stable op-date index map for shared-mode conditioning.
    op_vocab = {label: i for i, label in enumerate(op_dates)}

    for op_label in op_dates:
        for year in years:
            weather_year_df = weather_by_year[year]
            sel = pick_issue_date_for_operational_label(
                weather_year_df=weather_year_df,
                season_year=year,
                operational_label=op_label,
            )

            wx, wm = weather_tensor_for_season(
                weather_df=weather_year_df,
                district_ids=district_ids,
                issue_date=sel.issue_date,
                horizon_days=horizon_days,
                weather_cols=WEATHER_BASE_FEATURES,
            )

            sx, sm, missing_ids = satellite_tensor_for_season(
                sat_map=sat_map,
                district_df=district_df,
                season_year=year,
                op_date=sel.issue_date,
                seq_len=seq_len,
                sangrur_weights=sangrur_weights,
            )
            active_steps = sat_step_active_mask(
                season_year=year,
                seq_len=seq_len,
                op_date=sel.issue_date,
            )
            sm_before = sm.copy()
            sx, sm = impute_satellite_by_state_mean(
                sat_x=sx,
                sat_mask=sm,
                district_df=district_df,
                active_steps=active_steps,
                set_mask_valid=bool(apply_sat_mask_fix),
            )
            coverage_rows.extend(
                coverage_rows_for_sample(
                    district_df=district_df,
                    season_year=year,
                    operational_date=op_label,
                    active_steps=active_steps,
                    sat_mask_before=sm_before,
                    sat_mask_after=sm,
                    missing_district_ids=missing_ids,
                )
            )

            yy = (
                yield_df[yield_df["season_start_year"] == int(year)]
                .set_index("district_id")
                .reindex(district_ids)["yield_kg_per_ha"]
                .to_numpy(dtype=np.float32)
            )
            if np.isnan(yy).any():
                miss_n = int(np.isnan(yy).sum())
                raise RuntimeError(f"Yield labels missing for season {year}, districts missing={miss_n}.")

            weather_list.append(wx)
            weather_mask_list.append(wm)
            sat_list.append(sx)
            sat_mask_list.append(sm)
            y_list.append(yy)
            sample_years.append(int(year))
            sample_opdates.append(str(op_label))
            sample_issue_dates.append(str(sel.issue_date.date()))

    weather_arr = np.stack(weather_list, axis=0).astype(np.float32)
    weather_mask_arr = np.stack(weather_mask_list, axis=0).astype(np.float32)
    sat_arr = np.stack(sat_list, axis=0).astype(np.float32)
    sat_mask_arr = np.stack(sat_mask_list, axis=0).astype(np.float32)
    y_raw = np.stack(y_list, axis=0).astype(np.float32)
    sample_years_arr = np.array(sample_years, dtype=np.int32)
    sample_opdate_arr = np.array(sample_opdates)
    sample_issue_arr = np.array(sample_issue_dates)
    sample_op_idx = np.array([op_vocab[x] for x in sample_opdates], dtype=np.int64)

    weather_feature_names = list(WEATHER_BASE_FEATURES)
    sat_feature_names = list(SAT_BASE_FEATURES)

    if use_engineered_weather:
        weather_arr, weather_feature_names = engineer_weather_features(weather_arr, weather_mask_arr)

    if use_engineered_satellite:
        sat_arr, sat_feature_names = engineer_satellite_features(sat_arr, sat_mask_arr)

    if use_missingness_indicators:
        weather_arr, w_extra, sat_arr, s_extra = add_missingness_indicators(
            weather_x=weather_arr,
            weather_mask=weather_mask_arr,
            sat_x=sat_arr,
            sat_mask=sat_mask_arr,
        )
        weather_feature_names = weather_feature_names + w_extra
        sat_feature_names = sat_feature_names + s_extra

    train_idx = np.where(np.isin(sample_years_arr, np.array(train_years, dtype=np.int32)))[0]
    val_idx = np.where(np.isin(sample_years_arr, np.array(val_years, dtype=np.int32)))[0]
    test_idx = np.where(np.isin(sample_years_arr, np.array(test_years, dtype=np.int32)))[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            "Split produced empty partition. Check years/opdates and split config. "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    w_mean, w_std = fit_masked_scaler(weather_arr, weather_mask_arr, train_idx)
    s_mean, s_std = fit_masked_scaler(sat_arr, sat_mask_arr, train_idx)
    weather_arr = apply_scaler(weather_arr, w_mean, w_std)
    sat_arr = apply_scaler(sat_arr, s_mean, s_std)

    target_mean, target_std = compute_target_stats(y_raw, train_idx=train_idx)
    y_target = apply_target_transform(
        y_raw=y_raw,
        target_mode=target_mode,
        target_mean=target_mean,
        target_std=target_std,
    )

    adjacency = build_adjacency_from_boundaries(
        district_df=district_df,
        boundary_config_path=Path(paths["boundary_config"]),
    )

    coverage_report = pd.DataFrame(coverage_rows)

    resolved = {
        "mode": str(mode),
        "target_mode": str(target_mode),
        "horizon_days": int(horizon_days),
        "operational_dates": op_dates,
        "years": years,
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years,
        "apply_sat_mask_fix": bool(apply_sat_mask_fix),
        "use_engineered_weather": bool(use_engineered_weather),
        "use_engineered_satellite": bool(use_engineered_satellite),
        "use_missingness_indicators": bool(use_missingness_indicators),
        "weather_feature_dim": int(weather_arr.shape[-1]),
        "sat_feature_dim": int(sat_arr.shape[-1]),
        "n_districts": int(len(district_df)),
        "n_samples": int(len(sample_years_arr)),
        "sat_seq_len": int(seq_len),
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "scaler_weather_mean": w_mean.tolist(),
        "scaler_weather_std": w_std.tolist(),
        "scaler_sat_mean": s_mean.tolist(),
        "scaler_sat_std": s_std.tolist(),
    }

    return DatasetBundle(
        weather_x=weather_arr,
        weather_mask=weather_mask_arr,
        sat_x=sat_arr,
        sat_mask=sat_mask_arr,
        y_raw=y_raw,
        y_target=y_target,
        sample_years=sample_years_arr,
        sample_operational_dates=sample_opdate_arr,
        sample_issue_dates=sample_issue_arr,
        sample_opdate_idx=sample_op_idx,
        district_df=district_df,
        adjacency=adjacency,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        target_mode=str(target_mode),
        target_mean=target_mean,
        target_std=target_std,
        weather_feature_names=weather_feature_names,
        sat_feature_names=sat_feature_names,
        coverage_report=coverage_report,
        sangrur_weights_report=sangrur_weights_report,
        config_resolved=resolved,
    )


def iter_opdate_subsets(dataset: DatasetBundle) -> Iterable[Tuple[str, np.ndarray]]:
    labels = dataset.sample_operational_dates
    unique = sorted(set(labels.tolist()))
    for op in unique:
        idx = np.where(labels == op)[0]
        yield op, idx
