#!/usr/bin/env python3
"""
S2S ECMWF Data Download Script

Downloads S2S reforecast data for wheat yield prediction.
- Period: 2017-2024 (wheat seasons)
- Variables: Tmax, Tmin, Precip, Solar, 10u, 10v
- Resolution: 0.4° (~36 km)
- Region: Punjab/Haryana/UP (24-32°N, 73-85°E)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

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
    "grid": "0.4/0.4",  # Native 36km resolution
    "variables": {
        "mx2t6": 121,    # Max temp 2m (6h)
        "mn2t6": 122,    # Min temp 2m (6h)
        "tp": 228228,    # Total precipitation
        "ssrd": 169,     # Surface solar radiation downwards
        "10u": 165,      # 10m U wind
        "10v": 166,      # 10m V wind
    },
    "lead_time_hours": 1104,  # 46 days
    "step_hours": 6,          # 6-hourly data
}

# Wheat season months (Oct-Apr)
WHEAT_SEASONS = [
    # (year, months_in_that_year)
    # For season 2017-18: Oct-Dec 2017 + Jan-Apr 2018
]

def get_wheat_season_months(start_year: int, end_year: int) -> list:
    """Generate list of (year, month) tuples for wheat seasons."""
    months = []
    for year in range(start_year, end_year + 1):
        # Oct-Dec of current year (sowing)
        for month in [10, 11, 12]:
            months.append((year, month))
        # Jan-Apr of next year (growth + harvest)
        for month in [1, 2, 3, 4]:
            months.append((year + 1, month))
    return months


def check_ecmwf_api():
    """Check if ECMWF API is configured."""
    try:
        from ecmwfapi import ECMWFDataServer
        server = ECMWFDataServer()
        logger.info("✓ ECMWF API configured")
        return True
    except Exception as e:
        logger.error(f"✗ ECMWF API not configured: {e}")
        logger.error("Please create ~/.ecmwfapirc with your credentials")
        logger.error("Get your key from: https://api.ecmwf.int/v1/key/")
        return False


def download_s2s_month(
    year: int,
    month: int,
    output_dir: Path,
    dry_run: bool = False
) -> Optional[Path]:
    """
    Download S2S reforecast for one month.
    
    Args:
        year: Year (e.g., 2020)
        month: Month (1-12)
        output_dir: Directory to save files
        dry_run: If True, print request but don't download
    
    Returns:
        Path to downloaded file, or None if failed
    """
    from ecmwfapi import ECMWFDataServer
    
    output_file = output_dir / f"s2s_ecmwf_{year}_{month:02d}.grib"
    
    # Skip if already exists
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Already exists: {output_file.name} ({file_size:.1f} MB)")
        return output_file
    
    # Build step string (0-1104 by 6)
    steps = "/".join(str(h) for h in range(0, CONFIG["lead_time_hours"] + 1, CONFIG["step_hours"]))
    
    # Build param string
    params = "/".join(str(v) for v in CONFIG["variables"].values())
    
    # Determine number of days in month
    if month in [1, 3, 5, 7, 8, 10, 12]:
        last_day = 31
    elif month in [4, 6, 9, 11]:
        last_day = 30
    else:  # February
        last_day = 28  # S2S reforecasts skip leap years anyway
    
    # Build hdate string (every 2 days for reforecasts)
    # Format: YYYY-MM-DD/to/YYYY-MM-DD/by/2
    hdate_start = f"{year}-{month:02d}-01"
    hdate_end = f"{year}-{month:02d}-{last_day}"
    hdate = f"{hdate_start}/to/{hdate_end}/by/2"
    
    # Area: N/W/S/E
    area = f"{CONFIG['region']['north']}/{CONFIG['region']['west']}/{CONFIG['region']['south']}/{CONFIG['region']['east']}"
    
    request = {
        "class": "s2",
        "dataset": "s2s",
        "origin": "ecmf",
        "type": "cf",           # Control forecast (could also use "pf" for perturbed)
        "stream": "enfh",       # Ensemble forecast hindcast
        "expver": "prod",
        "model": "glob",
        "levtype": "sfc",
        "param": params,
        "hdate": hdate,
        "time": "00:00:00",
        "step": steps,
        "area": area,
        "grid": CONFIG["grid"],
        "target": str(output_file)
    }
    
    if dry_run:
        logger.info(f"[DRY RUN] Would download: {year}-{month:02d}")
        logger.info(f"  Request: {request}")
        return None
    
    logger.info(f"⏳ Downloading: {year}-{month:02d}...")
    
    try:
        server = ECMWFDataServer()
        server.retrieve(request)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Complete: {output_file.name} ({file_size:.1f} MB)")
        return output_file
        
    except Exception as e:
        logger.error(f"✗ Failed: {year}-{month:02d} - {e}")
        # Remove partial file if exists
        if output_file.exists():
            output_file.unlink()
        return None


def download_all(
    start_year: int = 2017,
    end_year: int = 2023,
    output_dir: str = "data/s2s",
    dry_run: bool = False
) -> dict:
    """
    Download all S2S data for wheat seasons.
    
    Args:
        start_year: First wheat season start year
        end_year: Last wheat season start year
        output_dir: Directory to save files
        dry_run: If True, print requests but don't download
    
    Returns:
        Dict with counts of successful, skipped, and failed downloads
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    months = get_wheat_season_months(start_year, end_year)
    
    logger.info(f"📥 S2S Download Plan")
    logger.info(f"   Period: {start_year}-{end_year + 1}")
    logger.info(f"   Months: {len(months)}")
    logger.info(f"   Output: {output_path.absolute()}")
    logger.info("")
    
    results = {"success": 0, "skipped": 0, "failed": 0}
    
    for i, (year, month) in enumerate(months):
        logger.info(f"[{i+1}/{len(months)}] Processing {year}-{month:02d}")
        
        result = download_s2s_month(year, month, output_path, dry_run)
        
        if result is None and not dry_run:
            results["failed"] += 1
        elif result and result.exists():
            results["success"] += 1
        else:
            results["skipped"] += 1
    
    logger.info("")
    logger.info(f"📊 Download Summary")
    logger.info(f"   Success: {results['success']}")
    logger.info(f"   Skipped: {results['skipped']}")
    logger.info(f"   Failed:  {results['failed']}")
    
    return results


def verify_downloads(data_dir: str = "data/s2s") -> dict:
    """Verify all downloaded files."""
    import xarray as xr
    import cfgrib
    
    data_path = Path(data_dir)
    files = sorted(data_path.glob("s2s_ecmwf_*.grib"))
    
    logger.info(f"🔍 Verifying {len(files)} files...")
    
    results = {"valid": 0, "invalid": 0, "errors": []}
    
    expected_vars = set(CONFIG["variables"].keys())
    
    for f in files:
        try:
            datasets = cfgrib.open_datasets(str(f))
            
            found_vars = set()
            for ds in datasets:
                found_vars.update(ds.data_vars.keys())
            
            # Map common variable name variations
            var_mapping = {
                'u10': '10u', 'v10': '10v',
                'mx2t': 'mx2t6', 'mn2t': 'mn2t6'
            }
            found_vars = {var_mapping.get(v, v) for v in found_vars}
            
            missing = expected_vars - found_vars
            
            if missing:
                logger.warning(f"⚠️  {f.name}: Missing {missing}")
                results["invalid"] += 1
                results["errors"].append((f.name, f"Missing: {missing}"))
            else:
                logger.info(f"✓ {f.name}: OK")
                results["valid"] += 1
                
        except Exception as e:
            logger.error(f"✗ {f.name}: {e}")
            results["invalid"] += 1
            results["errors"].append((f.name, str(e)))
    
    logger.info("")
    logger.info(f"📊 Verification Summary")
    logger.info(f"   Valid:   {results['valid']}")
    logger.info(f"   Invalid: {results['invalid']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Download S2S ECMWF data")
    parser.add_argument("--start-year", type=int, default=2017,
                        help="First wheat season start year (default: 2017)")
    parser.add_argument("--end-year", type=int, default=2023,
                        help="Last wheat season start year (default: 2023)")
    parser.add_argument("--output-dir", type=str, default="data/s2s",
                        help="Output directory (default: data/s2s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print requests without downloading")
    parser.add_argument("--verify", action="store_true",
                        help="Verify downloaded files")
    parser.add_argument("--test", action="store_true",
                        help="Test with single month (Oct 2020)")
    
    args = parser.parse_args()
    
    # Check API configuration
    if not check_ecmwf_api():
        sys.exit(1)
    
    if args.verify:
        verify_downloads(args.output_dir)
    elif args.test:
        logger.info("🧪 Test mode: downloading Oct 2020 only")
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        download_s2s_month(2020, 10, output_path, args.dry_run)
    else:
        download_all(args.start_year, args.end_year, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
