"""
Landsat Wheat Yield Prediction Pipeline
========================================

A remote sensing data processing pipeline for wheat yield prediction
using Landsat 5/7/8/9 data from 2010 to 2026.

Target Regions:
- Haryana (India)
- Punjab (India)
- Uttar Pradesh (India)

Modules:
- config: Configuration parameters (bands, districts, sensor mapping)
- gee_extractor: Landsat data extraction via Google Earth Engine
- fill_missing_sma: Fill missing values with Simple Moving Average
- process_all_districts: Multi-year processing pipeline orchestrator
"""

__version__ = "1.0.0"
__author__ = "Pipeline Implementation"

from .config import (
    ALL_DISTRICTS,
    LANDSAT_BANDS,
    SENSOR_BAND_MAPPING,
    YEAR_SENSOR_MAPPING,
    PROCESSING_YEARS
)
