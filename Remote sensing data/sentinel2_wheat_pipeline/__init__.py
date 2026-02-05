"""
Wheat Yield Prediction Pipeline
================================

A remote sensing data processing pipeline for wheat yield prediction
using Sentinel-2 and weather data.

Target Regions:
- Haryana (India)
- Punjab (India)
- Western Uttar Pradesh (India)

Modules:
- config: Configuration parameters
- gee_extractor: Sentinel-2 data extraction via Google Earth Engine
- weather_extractor: Weather data extraction via NASA POWER API
- data_preprocessor: Data preprocessing and normalization
- run_pipeline: Main pipeline orchestrator
"""

__version__ = "1.0.0"
__author__ = "Pipeline Implementation"

from .config import (
    ALL_DISTRICTS,
    SENTINEL2_BANDS,
    WEATHER_PARAMETERS,
    PROCESSING_YEARS
)
