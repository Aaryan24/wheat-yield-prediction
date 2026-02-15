#!/usr/bin/env python3
"""
S2S ECMWF Reforecast Data Download Script (v2)

Downloads S2S reforecast data for wheat yield prediction.
- Period: 2017-2024 (wheat seasons via hindcast dates)
- Variables: Tmax, Tmin, Precip, Solar, 10u, 10v
- Resolution: 0.4° (~36 km)
- Region: Punjab/Haryana/UP (24-32°N, 73-85°E)

Key insight: S2S reforecasts are organized by (realtime_date, hindcast_dates) pairs.
For each realtime forecast date, hindcasts are available for the same calendar
day across ~20 years of historical dates.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

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
    "step_start": 6,          # Start at step 6 (step 0 not available for all params)
    
    # S2S reforecast years available in hindcast
    "hindcast_years": list(range(2006, 2025)),  # 2006-2024
    
    # Wheat season relevant months
    "wheat_months": [10, 11, 12, 1, 2, 3, 4],  # Oct-Apr
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
        logger.error("Please create ~/.ecmwfapirc with your credentials")
        logger.error("Get your key from: https://api.ecmwf.int/v1/key/")
        return False


def get_forecast_dates_for_month(year: int, month: int) -> List[str]:
    """
    Get S2S forecast dates for a given month.
    ECMWF runs forecasts on Mondays and Thursdays (approximately twice weekly).
    For reforecasts, we need specific dates that exist in the archive.
    
    Returns list of dates in YYYY-MM-DD format.
    """
    from calendar import monthrange
    
    dates = []
    _, last_day = monthrange(year, month)
    
    # Generate dates every 2 days (standard S2S reforecast frequency)
    for day in range(1, last_day + 1, 2):
        dates.append(f"{year}-{month:02d}-{day:02d}")
    
    return dates


def download_s2s_single_date(
    realtime_date: str,
    hindcast_years: List[int],
    output_dir: Path,
    dry_run: bool = False,
    target_years: List[int] = None
) -> Optional[Path]:
    """
    Download S2S reforecast for a single realtime date with multiple hindcast years.
    
    Args:
        realtime_date: Realtime date in YYYY-MM-DD format (model version date)
        hindcast_years: List of years for hindcast dates
        output_dir: Directory to save files
        dry_run: If True, print request but don't download
        target_years: If specified, only download hindcasts for these years
    
    Returns:
        Path to downloaded file, or None if failed
    """
    from ecmwfapi import ECMWFDataServer
    
    # Parse realtime date
    rt_year, rt_month, rt_day = realtime_date.split('-')
    
    # Build hindcast dates (same month-day in past years)
    if target_years:
        years_to_use = [y for y in target_years if y in hindcast_years]
    else:
        years_to_use = hindcast_years
    
    # Filter to wheat-relevant years (2017-2024 for our project)
    years_to_use = [y for y in years_to_use if 2017 <= y <= 2024]
    
    if not years_to_use:
        logger.warning(f"No valid hindcast years for {realtime_date}")
        return None
    
    hdate_list = [f"{y}-{rt_month}-{rt_day}" for y in years_to_use]
    hdate_str = "/".join(hdate_list)
    
    # Output filename
    output_file = output_dir / f"s2s_ecmwf_rf_{rt_month}_{rt_day}.grib"
    
    # Skip if already exists
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Already exists: {output_file.name} ({file_size:.1f} MB)")
        return output_file
    
    # Build step string (6-1104 by 6) - start from 6 as step 0 is not available for all params
    steps = "/".join(str(h) for h in range(CONFIG["step_start"], CONFIG["lead_time_hours"] + 1, CONFIG["step_hours"]))
    
    # Build param string
    params = "/".join(str(v) for v in CONFIG["variables"].values())
    
    # Area: N/W/S/E
    area = f"{CONFIG['region']['north']}/{CONFIG['region']['west']}/{CONFIG['region']['south']}/{CONFIG['region']['east']}"
    
    # The request uses the DATE (realtime) and HDATE (hindcast dates)
    request = {
        "class": "s2",
        "dataset": "s2s",
        "origin": "ecmf",
        "type": "cf",           # Control forecast
        "stream": "enfh",       # Ensemble forecast hindcast
        "expver": "prod",
        "model": "glob",
        "levtype": "sfc",
        "param": params,
        "date": realtime_date,  # The model version date
        "hdate": hdate_str,     # Hindcast dates (same month-day in past years)
        "time": "00:00:00",
        "step": steps,
        "area": area,
        "grid": CONFIG["grid"],
        "target": str(output_file)
    }
    
    if dry_run:
        logger.info(f"[DRY RUN] Would download: {realtime_date}")
        logger.info(f"  Hindcast dates: {hdate_str}")
        logger.info(f"  Request: {request}")
        return None
    
    logger.info(f"⏳ Downloading: {realtime_date} (hdates: {len(years_to_use)} years)...")
    
    try:
        server = ECMWFDataServer()
        server.retrieve(request)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Complete: {output_file.name} ({file_size:.1f} MB)")
        return output_file
        
    except Exception as e:
        error_str = str(e)
        # Check if the file was actually downloaded (partial success)
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)
            if file_size > 0.1:  # If we got at least 100 KB, keep it
                logger.warning(f"⚠️ Partial download: {output_file.name} ({file_size:.1f} MB) - {error_str}")
                return output_file
            else:
                logger.error(f"✗ Failed (removing tiny file): {realtime_date} - {error_str}")
                output_file.unlink()
                return None
        else:
            logger.error(f"✗ Failed: {realtime_date} - {error_str}")
            return None


def download_wheat_season(
    start_year: int = 2017,
    end_year: int = 2024,
    output_dir: str = "data/s2s",
    dry_run: bool = False
) -> dict:
    """
    Download S2S reforecasts for all wheat season dates.
    
    For wheat (Oct-Apr), we need forecasts initialized on those dates.
    We'll download by realtime date and collect hindcasts for past years.
    
    Args:
        start_year: First year for hindcast data
        end_year: Last year for hindcast data  
        output_dir: Directory to save files
        dry_run: If True, print requests but don't download
    
    Returns:
        Dict with counts of successful, skipped, and failed downloads
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # We need a recent realtime date to get hindcasts
    # Using recent date that should have all hindcast years available
    realtime_base = datetime(2026, 1, 21)  # Recent date from the web interface
    
    logger.info(f"📥 S2S Reforecast Download Plan")
    logger.info(f"   Hindcast years: {start_year}-{end_year}")
    logger.info(f"   Months: Oct-Apr (wheat season)")
    logger.info(f"   Output: {output_path.absolute()}")
    logger.info("")
    
    results = {"success": 0, "skipped": 0, "failed": 0}
    
    # For each day in the wheat season months
    for month in CONFIG["wheat_months"]:
        dates = get_forecast_dates_for_month(2024, month)  # Year doesn't matter for day generation
        
        for date_str in dates:
            day = int(date_str.split('-')[2])
            # Construct the realtime date using a recent year
            rt_date = f"2026-{month:02d}-{day:02d}"
            
            logger.info(f"Processing {rt_date}")
            
            result = download_s2s_single_date(
                realtime_date=rt_date,
                hindcast_years=CONFIG["hindcast_years"],
                output_dir=output_path,
                dry_run=dry_run,
                target_years=list(range(start_year, end_year + 1))
            )
            
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


def test_download(output_dir: str = "data/s2s", dry_run: bool = False):
    """Test with a single date that's known to work."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("🧪 Test mode: downloading single date (Jan 21)")
    
    # Test with single hindcast year and all 6 parameters
    result = download_s2s_single_date(
        realtime_date="2026-01-21",
        hindcast_years=[2020, 2021, 2022, 2023, 2024],
        output_dir=output_path,
        dry_run=dry_run,
        target_years=[2020]  # Just 1 year for test
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Download S2S ECMWF reforecast data")
    parser.add_argument("--start-year", type=int, default=2017,
                        help="First hindcast year (default: 2017)")
    parser.add_argument("--end-year", type=int, default=2024,
                        help="Last hindcast year (default: 2024)")
    parser.add_argument("--output-dir", type=str, default="data/s2s",
                        help="Output directory (default: data/s2s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print requests without downloading")
    parser.add_argument("--test", action="store_true",
                        help="Test with single date")
    
    args = parser.parse_args()
    
    # Check API configuration
    if not check_ecmwf_api():
        sys.exit(1)
    
    if args.test:
        test_download(args.output_dir, args.dry_run)
    else:
        download_wheat_season(args.start_year, args.end_year, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
