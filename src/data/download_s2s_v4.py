#!/usr/bin/env python3
"""
S2S ECMWF Reforecast Data Download Script (v4 - Production)

Handles different step frequencies for different parameter types:
- 6-hourly: temperature (mx2t6, mn2t6), wind (10u, 10v)
- 24-hourly: accumulated (tp, ssrd)

Downloads S2S reforecast data for wheat yield prediction.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
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
            "step_start": 6,
            "step_end": 1104,
            "step_interval": 6,
            "description": "Temperature (6-hourly)"
        },
        "wind": {
            "codes": "165/166",  # 10u, 10v
            "step_start": 6,
            "step_end": 1104,
            "step_interval": 6,
            "description": "Wind (6-hourly)"
        },
        "accum": {
            "codes": "228228/169",  # tp, ssrd
            "step_start": 24,
            "step_end": 1104,
            "step_interval": 24,  # 24-hourly for accumulated
            "description": "Precip/Solar (24-hourly)"
        }
    },
    
    # Wheat season months (Oct-Apr)
    "wheat_months": [10, 11, 12, 1, 2, 3, 4],
}


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
        logger.info(f"✓ Exists: {output_file.name} ({file_size:.2f} MB)")
        return output_file
    
    # Build step string with correct interval
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
        logger.info(f"[DRY] {param_group}: {hindcast_date}")
        return None
    
    logger.info(f"⏳ {param_group}: {hindcast_date}...")
    
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


def download_season(
    year: int,
    output_dir: str = "data/s2s",
    dry_run: bool = False,
    param_groups: List[str] = None
) -> Dict:
    """
    Download S2S data for one wheat season.
    
    A wheat season spans Oct of year to Apr of year+1.
    E.g., 2017 season = Oct 2017 - Apr 2018
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if param_groups is None:
        param_groups = list(CONFIG["param_groups"].keys())
    
    logger.info(f"📥 Downloading {year} wheat season (Oct {year} - Apr {year+1})")
    logger.info(f"   Parameters: {param_groups}")
    logger.info(f"   Output: {output_path.absolute()}")
    logger.info("")
    
    results = {"success": 0, "failed": 0, "skipped": 0}
    
    # Generate all dates for the wheat season
    dates_to_download = []
    
    # Oct-Dec of year
    for month in [10, 11, 12]:
        _, last_day = monthrange(year, month)
        for day in range(1, last_day + 1, 2):  # Every odd day
            dates_to_download.append((year, month, day))
    
    # Jan-Apr of year+1
    for month in [1, 2, 3, 4]:
        _, last_day = monthrange(year + 1, month)
        for day in range(1, last_day + 1, 2):
            dates_to_download.append((year + 1, month, day))
    
    total_downloads = len(dates_to_download) * len(param_groups)
    current = 0
    
    for y, m, d in dates_to_download:
        # Realtime date must be in the past. Use 2026 since we're now in Feb 2026.
        # For any month/day, use 2026-MM-DD if that date is in the past.
        from datetime import datetime, date
        target_date = date(2026, m, d)
        today = date(2026, 2, 4)  # Current date
        
        if target_date <= today:
            rt_year = 2026
        else:
            # For future dates in 2026 (e.g., Mar-Apr), use 2025
            rt_year = 2025
        realtime_date = f"{rt_year}-{m:02d}-{d:02d}"
        hindcast_date = f"{y}-{m:02d}-{d:02d}"
        
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
    pattern = f"s2s_*_{year}_*.grib" if year else "s2s_*.grib"
    files = sorted(data_path.glob(pattern))
    
    if not files:
        # Also check for year+1 (Jan-Apr of next year)
        if year:
            pattern2 = f"s2s_*_{year+1}_*.grib"
            files = sorted(data_path.glob(pattern)) + sorted(data_path.glob(pattern2))
    
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


def main():
    parser = argparse.ArgumentParser(description="Download S2S ECMWF reforecast data")
    parser.add_argument("--year", type=int, help="Wheat season year (e.g., 2017)")
    parser.add_argument("--output-dir", type=str, default="data/s2s")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify downloads")
    parser.add_argument("--params", type=str, nargs="+", 
                        choices=["temp", "wind", "accum"],
                        help="Specific param groups to download")
    
    args = parser.parse_args()
    
    if not check_ecmwf_api():
        sys.exit(1)
    
    if args.verify:
        verify_downloads(args.output_dir, args.year)
    elif args.year:
        download_season(args.year, args.output_dir, args.dry_run, args.params)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
