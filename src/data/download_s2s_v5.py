#!/usr/bin/env python3
"""
S2S ECMWF Reforecast Data Download Script (v5 - Corrected Dates)

IMPORTANT: ECMWF S2S reforecasts are available only on specific dates:
- Before Nov 2024: Monday & Thursday (twice weekly)
- After Nov 2024: Every odd day of the month

This script correctly handles the Mon/Thu schedule for historical data.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange
from typing import Optional, List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('download_s2s.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "region": {
        "north": 32,
        "south": 24,
        "west": 73,
        "east": 85,
    },
    "grid": "0.4/0.4",
    
    # Parameter groups with their step configurations
    "param_groups": {
        "temp": {
            "codes": "121/122",  # mx2t6, mn2t6
            "step_start": 0,     # Changed to 0 as requested
            "step_end": 1104,
            "step_interval": 6,
            "description": "Temperature (6-hourly)"
        },
        "wind": {
            "codes": "165/166",  # 10u, 10v
            "step_start": 0,     # Instantaneous - step 0 available
            "step_end": 1104,
            "step_interval": 6,
            "description": "Wind (6-hourly)"
        },
        "accum": {
            "codes": "228228/169",  # tp, ssrd
            "step_start": 24,    # Accumulated - starts at step 24
            "step_end": 1104,
            "step_interval": 24,
            "description": "Precip/Solar (24-hourly)"
        }
    },
}


def get_mondays_thursdays(year: int, month: int) -> List[int]:
    """
    Get all Mondays and Thursdays in a given month.
    
    S2S reforecasts (before Nov 2024) were produced for Mon/Thu only.
    """
    days = []
    _, last_day = monthrange(year, month)
    
    for day in range(1, last_day + 1):
        date = datetime(year, month, day)
        weekday = date.weekday()  # 0=Monday, 3=Thursday
        if weekday in [0, 3]:  # Monday or Thursday
            days.append(day)
    
    return days


def check_ecmwf_api():
    """Check if ECMWF API is configured."""
    try:
        from ecmwfapi import ECMWFDataServer
        server = ECMWFDataServer()
        logger.info("✓ ECMWF API configured")
        return True
    except Exception as e:
        logger.error(f"✗ ECMWF API not configured: {e}")
        return False


def download_param_group(
    realtime_date: str,
    hindcast_date: str,
    param_group: str,
    output_dir: Path,
    dry_run: bool = False
) -> Optional[Path]:
    """Download a parameter group for a specific hindcast date."""
    from ecmwfapi import ECMWFDataServer
    
    group_config = CONFIG["param_groups"][param_group]
    hd_parts = hindcast_date.split('-')
    
    output_file = output_dir / f"s2s_{param_group}_{hd_parts[0]}_{hd_parts[1]}_{hd_parts[2]}.grib"
    
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)
        if file_size > 0.01:  # > 10 KB
            logger.info(f"✓ Exists: {output_file.name} ({file_size:.2f} MB)")
            return output_file
    
    # Build step string
    steps = "/".join(str(h) for h in range(
        group_config["step_start"],
        group_config["step_end"] + 1,
        group_config["step_interval"]
    ))
    
    area = f"{CONFIG['region']['north']}/{CONFIG['region']['west']}/{CONFIG['region']['south']}/{CONFIG['region']['east']}"
    
    request = {
        "class": "s2",
        "dataset": "s2s",
        "origin": "ecmf",
        "type": "cf",
        "stream": "enfh",
        "expver": "prod",
        "model": "glob",
        "levtype": "sfc",
        "param": group_config["codes"],
        "date": realtime_date,
        "hdate": hindcast_date,
        "time": "00:00:00",
        "step": steps,
        "area": area,
        "grid": CONFIG["grid"],
        "target": str(output_file)
    }
    
    if dry_run:
        logger.info(f"[DRY] {param_group}: rt={realtime_date}, hd={hindcast_date}")
        return None
    
    logger.info(f"⏳ {param_group}: hd={hindcast_date} (rt={realtime_date})...")
    
    try:
        server = ECMWFDataServer()
        server.retrieve(request)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Done: {output_file.name} ({file_size:.2f} MB)")
        return output_file
        
    except Exception as e:
        if output_file.exists() and output_file.stat().st_size > 1024:
            logger.warning(f"⚠️ Partial: {output_file.name}")
            return output_file
        logger.error(f"✗ Failed: {param_group} {hindcast_date} - {e}")
        if output_file.exists():
            output_file.unlink()
        return None


def find_nearest_mon_thu(year: int, month: int, day: int) -> datetime:
    """
    Find the nearest Monday or Thursday to a given date.
    Used to find proper realtime date for a hindcast.
    """
    date = datetime(year, month, day)
    weekday = date.weekday()
    
    # Days to nearest Mon (0) or Thu (3)
    if weekday == 0 or weekday == 3:
        return date
    elif weekday in [1, 2]:  # Tue, Wed -> closest is Mon or Thu
        days_to_mon = weekday
        days_to_thu = 3 - weekday
        if days_to_mon <= days_to_thu:
            return date - timedelta(days=days_to_mon)
        else:
            return date + timedelta(days=days_to_thu)
    elif weekday == 4:  # Fri -> Thu is 1 day back
        return date - timedelta(days=1)
    elif weekday == 5:  # Sat -> Thu is 2 days back
        return date - timedelta(days=2)
    else:  # Sun (6) -> Mon is 1 day forward
        return date + timedelta(days=1)


def download_season(
    year: int,
    output_dir: str = "data/s2s",
    dry_run: bool = False,
    param_groups: List[str] = None
) -> Dict:
    """
    Download S2S data for one wheat season using correct Mon/Thu dates.
    
    A wheat season spans Oct of year to Apr of year+1.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if param_groups is None:
        param_groups = list(CONFIG["param_groups"].keys())
    
    logger.info(f"📥 S2S Reforecast Download")
    logger.info(f"   Season: Oct {year} - Apr {year+1}")
    logger.info(f"   Dates: Mondays & Thursdays only")
    logger.info(f"   Parameters: {param_groups}")
    logger.info(f"   Output: {output_path.absolute()}")
    logger.info("")
    
    results = {"success": 0, "failed": 0}
    
    # Generate all Mon/Thu dates for the wheat season
    dates_to_download = []
    
    # Oct-Dec of year
    for month in [10, 11, 12]:
        mon_thu_days = get_mondays_thursdays(year, month)
        for day in mon_thu_days:
            dates_to_download.append((year, month, day))
    
    # Jan-Apr of year+1
    for month in [1, 2, 3, 4]:
        mon_thu_days = get_mondays_thursdays(year + 1, month)
        for day in mon_thu_days:
            dates_to_download.append((year + 1, month, day))
    
    logger.info(f"   Total dates: {len(dates_to_download)} Mon/Thu dates")
    logger.info(f"   Total downloads: {len(dates_to_download) * len(param_groups)}")
    logger.info("")
    
    total_downloads = len(dates_to_download) * len(param_groups)
    current = 0
    
    for y, m, d in dates_to_download:
        hindcast_date = f"{y}-{m:02d}-{d:02d}"
        
        # Find realtime date: same Mon/Thu in recent past year
        # Use most recent year where that date has passed
        today = datetime.now()
        
        # Try to find matching Mon/Thu in recent years (2025, 2024, etc.)
        rt_date = None
        for rt_year in [2025, 2024, 2023]:
            try:
                candidate = datetime(rt_year, m, d)
                if candidate < today and candidate.weekday() in [0, 3]:
                    rt_date = candidate
                    break
            except ValueError:
                continue
        
        if rt_date is None:
            # Fallback: find nearest Mon/Thu
            rt_date = find_nearest_mon_thu(2025, m, d)
            if rt_date >= today:
                rt_date = find_nearest_mon_thu(2024, m, d)
        
        realtime_date = rt_date.strftime("%Y-%m-%d")
        
        for group in param_groups:
            current += 1
            logger.info(f"[{current}/{total_downloads}] {group} for {hindcast_date}")
            
            result = download_param_group(
                realtime_date=realtime_date,
                hindcast_date=hindcast_date,
                param_group=group,
                output_dir=output_path,
                dry_run=dry_run
            )
            
            if result:
                results["success"] += 1
            else:
                results["failed"] += 1
    
    logger.info("")
    logger.info(f"📊 Season {year} Summary")
    logger.info(f"   Success: {results['success']}")
    logger.info(f"   Failed:  {results['failed']}")
    
    return results


def verify_downloads(data_dir: str = "data/s2s", year: int = None) -> Dict:
    """Verify downloaded GRIB files."""
    import cfgrib
    
    data_path = Path(data_dir)
    
    if year:
        patterns = [f"s2s_*_{year}_*.grib", f"s2s_*_{year+1}_0[1-4]_*.grib"]
        files = []
        for p in patterns:
            files.extend(data_path.glob(p))
        files = sorted(set(files))
    else:
        files = sorted(data_path.glob("s2s_*.grib"))
    
    logger.info(f"🔍 Verifying {len(files)} files...")
    
    results = {"valid": 0, "invalid": 0, "total_size_mb": 0}
    
    for f in files:
        try:
            datasets = cfgrib.open_datasets(str(f))
            size_mb = f.stat().st_size / (1024 * 1024)
            results["total_size_mb"] += size_mb
            
            vars_found = []
            for ds in datasets:
                vars_found.extend(list(ds.data_vars.keys()))
            
            logger.info(f"✓ {f.name}: {vars_found} ({size_mb:.2f} MB)")
            results["valid"] += 1
            
        except Exception as e:
            logger.error(f"✗ {f.name}: {e}")
            results["invalid"] += 1
    
    logger.info("")
    logger.info(f"📊 Verification Summary")
    logger.info(f"   Valid:   {results['valid']}")
    logger.info(f"   Invalid: {results['invalid']}")
    logger.info(f"   Total:   {results['total_size_mb']:.1f} MB")
    
    return results


def list_dates(year: int):
    """List all Mon/Thu dates for a wheat season."""
    print(f"\n📅 Mon/Thu dates for {year} wheat season:\n")
    
    # Oct-Dec
    for month in [10, 11, 12]:
        days = get_mondays_thursdays(year, month)
        month_name = datetime(year, month, 1).strftime("%b")
        day_strs = [f"{d:02d}" for d in days]
        print(f"  {year}-{month_name}: {', '.join(day_strs)} ({len(days)} days)")
    
    # Jan-Apr
    for month in [1, 2, 3, 4]:
        days = get_mondays_thursdays(year + 1, month)
        month_name = datetime(year + 1, month, 1).strftime("%b")
        day_strs = [f"{d:02d}" for d in days]
        print(f"  {year+1}-{month_name}: {', '.join(day_strs)} ({len(days)} days)")
    
    # Count total
    total = 0
    for month in [10, 11, 12]:
        total += len(get_mondays_thursdays(year, month))
    for month in [1, 2, 3, 4]:
        total += len(get_mondays_thursdays(year + 1, month))
    
    print(f"\n  Total: {total} reforecast dates")
    print(f"  Downloads: {total * 3} files (3 param groups)")


def main():
    parser = argparse.ArgumentParser(description="Download S2S ECMWF reforecast data")
    parser.add_argument("--year", type=int, help="Wheat season year (e.g., 2017)")
    parser.add_argument("--output-dir", type=str, default="data/s2s")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify downloads")
    parser.add_argument("--list-dates", action="store_true", help="List Mon/Thu dates")
    parser.add_argument("--params", type=str, nargs="+", 
                        choices=["temp", "wind", "accum"],
                        help="Specific param groups to download")
    
    args = parser.parse_args()
    
    if not check_ecmwf_api():
        sys.exit(1)
    
    if args.list_dates:
        list_dates(args.year or 2017)
    elif args.verify:
        verify_downloads(args.output_dir, args.year)
    elif args.year:
        download_season(args.year, args.output_dir, args.dry_run, args.params)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
