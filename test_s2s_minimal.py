#!/usr/bin/env python3
"""
Minimal test script to download S2S data with exact parameters that work.
Based on web interface settings from user's screenshot.
"""

from ecmwfapi import ECMWFDataServer
from pathlib import Path

# Create output dir
output_dir = Path("data/s2s")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "test_s2s_jan21.grib"

# Using exact parameters from web interface, with reduced steps (only 1080-1104)
# to match what we saw selected in the screenshot
server = ECMWFDataServer()

# Web interface showed: steps 1080, 1086, 1092, 1098 selected
# and parameters: min temp (122), total precip (228228)
request = {
    "class": "s2",
    "dataset": "s2s",
    "origin": "ecmf",
    "type": "cf",
    "stream": "enfh",
    "expver": "prod",
    "model": "glob",
    "levtype": "sfc",
    "param": "122/228228",  # Just 2 params: min temp + precip
    "date": "2026-01-21",  # Realtime date (model version)
    "hdate": "2024-01-21",  # Single hindcast year
    "time": "00:00:00",
    "step": "6/12/18/24/30/36/42/48",  # Only first 2 days of forecast, 6-hourly
    "area": "32/73/24/85",  # N/W/S/E - our region
    "grid": "0.4/0.4",
    "target": str(output_file)
}

print(f"Request: {request}")
print("Downloading...")

try:
    server.retrieve(request)
    print(f"✓ Success! File: {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")
except Exception as e:
    print(f"Error: {e}")
    if output_file.exists():
        print(f"But file exists: {output_file.stat().st_size / 1024:.1f} KB")
