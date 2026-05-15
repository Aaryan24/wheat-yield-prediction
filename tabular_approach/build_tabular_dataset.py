#!/usr/bin/env python3
import os
import re
import warnings
from pathlib import Path
import datetime as dt

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

ROOT_DIR = Path(r"d:\IIT\Academics\BCS\wheat-yield-prediction")
SNAPSHOT_DIR = ROOT_DIR / "wheat_yield_2010_2022_snapshot"
OP_DATE_STR = "03-05"
OUT_FILE = ROOT_DIR / "tabular_approach" / f"flattened_dataset_opdate_{OP_DATE_STR}_v2.csv"

def _norm(text):
    x = str(text).strip().lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def _get_op_date(season_year: int) -> pd.Timestamp:
    m, d = map(int, OP_DATE_STR.split("-"))
    y = season_year if m >= 9 else season_year + 1
    return pd.Timestamp(dt.date(y, m, d))

def process_agri():
    agri_dir = ROOT_DIR / "Agri-economics-variables"
    
    fert = pd.read_csv(agri_dir / "ICRISAT-District Level Data-fertilizers.csv")
    irr = pd.read_csv(agri_dir / "ICRISAT-District Level Data-irrigated-area.csv")
    lu = pd.read_csv(agri_dir / "ICRISAT-District Level Data-landuse.csv")
    
    for df in [fert, irr, lu]:
        df['state_norm'] = df['State Name'].apply(_norm)
        df['dist_norm'] = df['Dist Name'].apply(_norm)
        
    merged = fert.merge(irr, on=['Year', 'state_norm', 'dist_norm'], how='outer', suffixes=('', '_drop'))
    merged = merged.merge(lu, on=['Year', 'state_norm', 'dist_norm'], how='outer', suffixes=('', '_drop'))
    
    drop_cols = [c for c in merged.columns if c.endswith('_drop')]
    merged = merged.drop(columns=drop_cols)
    
    # Rename columns to standard
    rename_map = {
        'Year': 'season_start_year',
        'NITROGEN RABI CONSUMPTION (tons)': 'nitro_rabi_tons',
        'PHOSPHATE RABI CONSUMPTION (tons)': 'phos_rabi_tons',
        'POTASH RABI CONSUMPTION (tons)': 'potash_rabi_tons',
        'TOTAL RABI CONSUMPTION (tons)': 'total_fert_rabi_tons',
        'WHEAT IRRIGATED AREA (1000 ha)': 'wheat_irrigated_area_1000ha',
        'NET CROPPED AREA (1000 ha)': 'net_cropped_area_1000ha',
        'GROSS CROPPED AREA (1000 ha)': 'gross_cropped_area_1000ha'
    }
    merged = merged.rename(columns=rename_map)
    return merged

def process_weather(years, district_ids):
    w_dir = SNAPSHOT_DIR / "data" / "weather_s2s_daily_2010_2022"
    all_weather = []
    
    months_to_keep = {10: 'oct', 11: 'nov', 12: 'dec', 1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr'}
    
    for y in years:
        fpath = w_dir / f"s2s_district_daily_{y}.parquet"
        if not fpath.exists():
            continue
        print(f"Processing weather for {y}...")
        df = pd.read_parquet(fpath)
        df = df[df['district_id'].isin(district_ids)]
        df['issue_date'] = pd.to_datetime(df['issue_date'])
        df['target_date'] = df['issue_date'] + pd.to_timedelta(df['lead_day'], unit='D')
        
        op_date = _get_op_date(y)
        
        # Split 1: Observed (target <= op_date)
        hist = df[df['target_date'] <= op_date]
        if not hist.empty:
            # take min lead_day per district and target_date
            idx = hist.groupby(['district_id', 'target_date'])['lead_day'].idxmin()
            hist = hist.loc[idx]
            
        # Split 2: Forecasted (target > op_date)
        fut = df[df['target_date'] > op_date]
        if not fut.empty:
            # Must be issued ON OR BEFORE op_date
            fut = fut[fut['issue_date'] <= op_date]
            if not fut.empty:
                # Take the latest issue date per district and target_date
                idx = fut.groupby(['district_id', 'target_date'])['issue_date'].idxmax()
                fut = fut.loc[idx]
        
        season_df = pd.concat([hist, fut])
        season_df['month'] = season_df['target_date'].dt.month
        season_df = season_df[season_df['month'].isin(months_to_keep.keys())]
        
        # Weather feature engineering
        season_df['heat_stress_day'] = (season_df['tmax_mean'] > 303.15).astype(int)
        season_df['dtr'] = season_df['tmax_mean'] - season_df['tmin_mean']
        season_df['gdd'] = np.maximum(0, (season_df['tmax_mean'] + season_df['tmin_mean']) / 2 - 278.15)
        
        agg_dict = {
            'tmax_mean': 'mean', 'tmin_mean': 'mean', 'tp_mean': 'mean', 
            'ssrd_mean': 'mean', 'wind_speed_mean': 'mean',
            'dtr': 'mean', 'heat_stress_day': 'sum', 'gdd': 'sum'
        }
        
        monthly = season_df.groupby(['district_id', 'month']).agg(agg_dict).reset_index()
        monthly['month_str'] = monthly['month'].map(months_to_keep)
        monthly['season_start_year'] = y
        agg_cols = list(agg_dict.keys())
        
        # Pivot
        pivot = monthly.pivot(index=['district_id', 'season_start_year'], columns='month_str', values=agg_cols)
        pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
        pivot = pivot.reset_index()
        
        # Season level engineered feature: total precip (mean * days_in_month proxy -> just sum of means is fine as relative feature)
        # Better: sum of tp_mean over the 7 months
        tp_cols = [c for c in pivot.columns if c.startswith('tp_mean')]
        pivot['season_precip_proxy'] = pivot[tp_cols].sum(axis=1)
        
        all_weather.append(pivot)
        
    return pd.concat(all_weather, ignore_index=True)

def process_satellite(years, district_df):
    sat_dir = SNAPSHOT_DIR / "data" / "remote_sensing_landsat_compat"
    all_sat = []
    
    months_to_keep = {10: 'oct', 11: 'nov', 12: 'dec', 1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr'}
    
    # Load all sat files
    sat_files = list(sat_dir.glob("*.csv"))
    print(f"Loading {len(sat_files)} satellite files...")
    df_list = []
    for f in sat_files:
        try:
            d = pd.read_csv(f)
            # The filename has the district/state, or we just merge later if we can.
            # Actually, Landsat files don't have district_id inside.
            # Filename format: {State}_{District}_remote_sensing_data.csv
            stem = f.stem.split("_remote_")[0]
            if stem.startswith("Uttar_Pradesh_"):
                state = "Uttar Pradesh"
                dist = stem[len("Uttar_Pradesh_"):]
            else:
                state, dist = stem.split("_", 1)
            dist = dist.replace("_", " ")
            
            d['state_norm'] = _norm(state)
            d['dist_norm'] = _norm(dist)
            df_list.append(d)
        except:
            pass
            
    sat_df = pd.concat(df_list, ignore_index=True)
    sat_df['end_date'] = pd.to_datetime(sat_df['end_date'])
    sat_df['start_date'] = pd.to_datetime(sat_df['start_date'])
    
    # Merge with district_df to get district_id
    district_df['state_norm'] = district_df['state_name'].apply(_norm)
    district_df['dist_norm'] = district_df['district_name'].apply(_norm)
    
    # Custom mappings if needed, but norm usually handles it
    # E.g. SAS Nagar -> Mohali
    district_aliases = {
        "s a s nagar": "mohali", "sas nagar": "mohali", "gurgaon": "gurugram", 
        "mewat": "nuh", "budaun": "badaun", "kheri": "lakhimpur kheri"
    }
    district_df['dist_norm'] = district_df['dist_norm'].apply(lambda x: district_aliases.get(x, x))
    sat_df['dist_norm'] = sat_df['dist_norm'].apply(lambda x: district_aliases.get(x, x))
    
    sat_df = sat_df.merge(district_df[['district_id', 'state_norm', 'dist_norm']], on=['state_norm', 'dist_norm'], how='inner')
    
    for y in years:
        print(f"Processing satellite for {y}...")
        df_y = sat_df[sat_df['year'] == y].copy()
        if df_y.empty:
            continue
            
        op_date = _get_op_date(y)
        
        # Mask out anything ending after op_date
        df_y = df_y[df_y['end_date'] <= op_date]
        if df_y.empty:
            continue
            
        df_y['month'] = df_y['start_date'].dt.month
        df_y = df_y[df_y['month'].isin(months_to_keep.keys())]
        
        df_y['NDWI'] = (df_y['B8'] - df_y['B12']) / (df_y['B8'] + df_y['B12'] + 1e-8)
        agg_cols = ['B7', 'B8', 'B8A', 'B12', 'NDWI']
        monthly = df_y.groupby(['district_id', 'month'])[agg_cols].mean().reset_index()
        monthly['month_str'] = monthly['month'].map(months_to_keep)
        monthly['season_start_year'] = y
        
        pivot = monthly.pivot(index=['district_id', 'season_start_year'], columns='month_str', values=agg_cols)
        pivot.columns = [f"sat_{col[0]}_mean_{col[1]}" for col in pivot.columns]
        pivot = pivot.reset_index()
        
        all_sat.append(pivot)
        
    if not all_sat:
        return pd.DataFrame()
    return pd.concat(all_sat, ignore_index=True)


def main():
    print("Loading Yield Data...")
    yield_file = SNAPSHOT_DIR / "data" / "yields" / "des_apy_wheat_rabi_2010_2022_model_ready_119.csv"
    df_yield = pd.read_csv(yield_file)
    years = sorted(df_yield['season_start_year'].unique())
    district_ids = df_yield['district_id'].unique()
    
    print("Processing Agri-Economic Data...")
    df_agri = process_agri()
    
    # Merge agri to yield
    df_yield['state_norm'] = df_yield['state_name'].apply(_norm)
    df_yield['dist_norm'] = df_yield['district_name'].apply(_norm)
    
    # Apply aliases
    district_aliases = {
        "s a s nagar": "mohali", "sas nagar": "mohali", "gurgaon": "gurugram", 
        "mewat": "nuh", "budaun": "badaun", "kheri": "lakhimpur kheri"
    }
    df_yield['dist_norm'] = df_yield['dist_norm'].apply(lambda x: district_aliases.get(x, x))
    df_agri['dist_norm'] = df_agri['dist_norm'].apply(lambda x: district_aliases.get(x, x))
    
    base_df = df_yield.merge(df_agri, on=['season_start_year', 'state_norm', 'dist_norm'], how='left')
    
    print("Processing Weather Data...")
    df_weather = process_weather(years, district_ids)
    base_df = base_df.merge(df_weather, on=['district_id', 'season_start_year'], how='left')
    
    print("Processing Satellite Data...")
    df_sat = process_satellite(years, df_yield[['district_id', 'state_name', 'district_name']].drop_duplicates())
    if not df_sat.empty:
        base_df = base_df.merge(df_sat, on=['district_id', 'season_start_year'], how='left')
        
    # Clean up columns
    drop_cols = ['state_norm', 'dist_norm', 'State Code', 'State Name', 'Dist Name', 'Dist Code']
    base_df = base_df.drop(columns=[c for c in drop_cols if c in base_df.columns])
    
    # Agronomic ratios
    base_df['nitrogen_per_ha'] = base_df['nitro_rabi_tons'] / base_df['area_ha']
    base_df['npk_ratio'] = base_df['nitro_rabi_tons'] / (base_df['total_fert_rabi_tons'] + 1e-8)
    base_df['irrigation_pct'] = base_df['wheat_irrigated_area_1000ha'] / (base_df['area_ha'] / 1000 + 1e-8)
    
    # Sort
    base_df = base_df.sort_values(['season_start_year', 'district_id']).reset_index(drop=True)
    
    print(f"\nFinal dataset shape: {base_df.shape}")
    os.makedirs(OUT_FILE.parent, exist_ok=True)
    base_df.to_csv(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
