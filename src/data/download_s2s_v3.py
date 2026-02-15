#!/usr/bin/env python3
"""
S2S ECMWF Reforecast Data Download Script (v3)

Key improvement: Separates instantaneous and accumulated parameters since
they have different step availability in the S2S archive.

Downloads S2S reforecast data for wheat yield prediction.
- Period: 2017-2024 (wheat seasons via hindcast dates)
- Variables: Tmax, Tmin, Precip, Solar, 10u, 10v
- Resolution: 0.4° (~36 km)
- Region: Punjab/Haryana/UP (24-32°N, 73-85°E)
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
    
    # Separate parameters by type (they have different step availability)
    "params_instantaneous": {
        "10u": 165,      # 10m U wind (instantaneous)
        "10v": 166,      # 10m V wind (instantaneous)
    },
    "params_6hourly": {
        "mx2t6": 121,    # Max temp 2m (6h window)
        "mn2t6": 122,    # Min temp 2m (6h window)
    },
    "params_accumulated": {
        "tp": 228228,    # Total precipitation (accumulated)
        "ssrd": 169,     # Surface solar radiation downwards (accumulated)
    },
    
    "lead_time_hours": 1104,  # 46 days
    "step_hours": 6,          # 6-hourly data
    "step_start": 6,          # Start at step 6
    
    # S2S reforecast years
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
        return False


def download_s2s_params(
    realtime_date: str,
    hindcast_date: str,
    param_codes: str,
    param_name: str,
    output_dir: Path,
    step_start: int = 6,
    step_end: int = 1104,
    step_interval: int = 6,
    dry_run: bool = False
) -> Optional[Path]:
    """
    Download S2S data for specific parameters.
    
    Args:
        realtime_date: Realtime date in YYYY-MM-DD format
        hindcast_date: Hindcast date in YYYY-MM-DD format
        param_codes: Parameter codes (e.g., "121/122")
        param_name: Name for output file (e.g., "temp_6h")
        output_dir: Directory to save files
        step_start: Starting step hour
        step_end: Ending step hour
        step_interval: Step interval
        dry_run: If True, print request but don't download
    
    Returns:
        Path to downloaded file, or None if failed
    """
    from ecmwfapi import ECMWFDataServer
    
    # Clean up dates for filename
    rt_parts = realtime_date.split('-')
    hd_parts = hindcast_date.split('-')
    
    output_file = output_dir / f"s2s_{param_name}_{hd_parts[0]}_{hd_parts[1]}_{hd_parts[2]}.grib"
    
    # Skip if already exists
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Already exists: {output_file.name} ({file_size:.2f} MB)")
        return output_file
    
    # Build step string
    steps = "/".join(str(h) for h in range(step_start, step_end + 1, step_interval))
    
    # Area: N/W/S/E
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
        "param": param_codes,
        "date": realtime_date,
        "hdate": hindcast_date,
        "time": "00:00:00",
        "step": steps,
        "area": area,
        "grid": CONFIG["grid"],
        "target": str(output_file)
    }
    
    if dry_run:
        logger.info(f"[DRY RUN] Would download: {param_name} for {hindcast_date}")
        logger.info(f"  Request: {request}")
        return None
    
    logger.info(f"⏳ Downloading: {param_name} for {hindcast_date}...")
    
    try:
        server = ECMWFDataServer()
        server.retrieve(request)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Complete: {output_file.name} ({file_size:.2f} MB)")
        return output_file
        
    except Exception as e:
        error_str = str(e)
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)
            if file_size > 0.01:  # Keep if > 10 KB
                logger.warning(f"⚠️ Partial: {output_file.name} ({file_size:.2f} MB)")
                return output_file
        logger.error(f"✗ Failed: {param_name} - {error_str}")
        return None


def test_download(output_dir: str = "data/s2s", dry_run: bool = False):
    """Test download with separate parameter types."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("🧪 Test mode: downloading separate parameter sets for Jan 21, 2020")
    
    realtime_date = "2026-01-21"
    hindcast_date = "2020-01-21"
    
    results = []
    
    # Test 1: Temperature 6-hourly parameters (mx2t6, mn2t6)
    temp_params = "/".join(str(v) for v in CONFIG["params_6hourly"].values())
    result = download_s2s_params(
        realtime_date=realtime_date,
        hindcast_date=hindcast_date,
        param_codes=temp_params,
        param_name="temp_6h",
        output_dir=output_path,
        dry_run=dry_run
    )
    results.append(("temp_6h", result))
    
    # Test 2: Wind instantaneous parameters (10u, 10v)
    wind_params = "/".join(str(v) for v in CONFIG["params_instantaneous"].values())
    result = download_s2s_params(
        realtime_date=realtime_date,
        hindcast_date=hindcast_date,
        param_codes=wind_params,
        param_name="wind_inst",
        output_dir=output_path,
        dry_run=dry_run
    )
    results.append(("wind_inst", result))
    
    # Test 3: Accumulated parameters (tp, ssrd)
    accum_params = "/".join(str(v) for v in CONFIG["params_accumulated"].values())
    result = download_s2s_params(
        realtime_date=realtime_date,
        hindcast_date=hindcast_date,
        param_codes=accum_params,
        param_name="accum",
        output_dir=output_path,
        dry_run=dry_run
    )
    results.append(("accum", result))
    
    # Summary
    logger.info("")
    logger.info("📊 Test Results:")
    for name, result in results:
        status = "✓" if result else "✗"
        logger.info(f"   {status} {name}")
    
    return results


def download_wheat_season(
    start_year: int = 2017,
    end_year: int = 2024,
    output_dir: str = "data/s2s",
    dry_run: bool = False
) -> dict:
    """Download S2S reforecasts for wheat season."""
    from calendar import monthrange
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📥 S2S Reforecast Download Plan")
    logger.info(f"   Hindcast years: {start_year}-{end_year}")
    logger.info(f"   Months: Oct-Apr (wheat season)")
    logger.info(f"   Output: {output_path.absolute()}")
    logger.info("")
    
    results = {"success": 0, "partial": 0, "failed": 0}
    
    # All parameters combined
    all_params = {}
    all_params.update(CONFIG["params_6hourly"])
    all_params.update(CONFIG["params_instantaneous"])
    all_params.update(CONFIG["params_accumulated"])
    param_codes = "/".join(str(v) for v in all_params.values())
    
    # For each wheat season month
    for month in CONFIG["wheat_months"]:
        _, last_day = monthrange(2024, month)
        
        # For each odd day (S2S reforecast frequency)
        for day in range(1, last_day + 1, 2):
            realtime_date = f"2026-{month:02d}-{day:02d}"
            
            # For each hindcast year
            for year in range(start_year, end_year + 1):
                hindcast_date = f"{year}-{month:02d}-{day:02d}"
                
                logger.info(f"Processing {hindcast_date}")
                
                result = download_s2s_params(
                    realtime_date=realtime_date,
                    hindcast_date=hindcast_date,
                    param_codes=param_codes,
                    param_name="all",
                    output_dir=output_path,
                    dry_run=dry_run
                )
                
                if result and result.exists():
                    results["success"] += 1
                else:
                    results["failed"] += 1
    
    logger.info("")
    logger.info(f"📊 Download Summary")
    logger.info(f"   Success: {results['success']}")
    logger.info(f"   Failed:  {results['failed']}")
    
    return results


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
    
    if not check_ecmwf_api():
        sys.exit(1)
    
    if args.test:
        test_download(args.output_dir, args.dry_run)
    else:
        download_wheat_season(args.start_year, args.end_year, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
