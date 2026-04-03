import os
from datetime import datetime

# ─── Directory Configuration ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ─── District Configuration ─────────────────────────────────────────────────

# Haryana Districts (all 22 districts)
HARYANA_DISTRICTS = [
    "Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad",
    "Gurugram", "Hisar", "Jhajjar", "Jind", "Kaithal",
    "Karnal", "Kurukshetra", "Mahendragarh", "Nuh", "Palwal",
    "Panchkula", "Panipat", "Rewari", "Rohtak", "Sirsa",
    "Sonipat", "Yamunanagar"
]

# Punjab Districts (all 23 districts)
PUNJAB_DISTRICTS = [
    "Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib",
    "Fazilka", "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar",
    "Kapurthala", "Ludhiana", "Malerkotla", "Mansa", "Moga",
    "Muktsar", "Pathankot", "Patiala", "Rupnagar", "SAS Nagar",
    "Sangrur", "Shaheed Bhagat Singh Nagar", "Tarn Taran"
]

# Uttar Pradesh Districts (all 75 districts)
UTTAR_PRADESH_DISTRICTS = [
    # Western UP (major wheat region)
    "Saharanpur", "Muzaffarnagar", "Shamli", "Meerut", "Baghpat",
    "Ghaziabad", "Gautam Buddha Nagar", "Bulandshahr", "Hapur", "Amroha",
    "Moradabad", "Bijnor", "Sambhal", "Rampur", "Bareilly",
    "Pilibhit", "Shahjahanpur", "Aligarh", "Hathras", "Mathura",
    "Agra", "Firozabad", "Etah", "Mainpuri", "Kasganj",
    # Central UP
    "Lucknow", "Unnao", "Rae Bareli", "Sitapur", "Hardoi",
    "Lakhimpur Kheri", "Kanpur Nagar", "Kanpur Dehat", "Farrukhabad",
    "Kannauj", "Auraiya", "Etawah", "Fatehpur",
    # Eastern UP
    "Allahabad", "Kaushambi", "Pratapgarh", "Jaunpur", "Varanasi",
    "Ghazipur", "Chandauli", "Mirzapur", "Sonbhadra", "Bhadohi",
    "Ballia", "Mau", "Azamgarh", "Ambedkar Nagar", "Sultanpur",
    "Faizabad", "Amethi", "Barabanki", "Gonda", "Bahraich",
    "Shrawasti", "Balrampur", "Siddharthnagar", "Basti", "Sant Kabir Nagar",
    "Maharajganj", "Gorakhpur", "Kushinagar", "Deoria",
    # Bundelkhand (Southern UP)
    "Jhansi", "Jalaun", "Lalitpur", "Mahoba", "Hamirpur",
    "Banda", "Chitrakoot"
]

ALL_DISTRICTS = {
    "Haryana": HARYANA_DISTRICTS,
    "Punjab": PUNJAB_DISTRICTS,
    "Uttar_Pradesh": UTTAR_PRADESH_DISTRICTS
}

# ─── Landsat Band Configuration ─────────────────────────────────────────────
#
# Landsat sensors have different band numbering but equivalent spectral ranges.
# We harmonize to common output names: NIR, SWIR1, SWIR2, Red.
#
# Cross-sensor band mapping:
#   Output Name  | L5 TM Band  | L7 ETM+ Band | L8 OLI Band | L9 OLI-2 Band
#   -------------|-------------|--------------|-------------|---------------
#   Red          | SR_B3       | SR_B3        | SR_B4       | SR_B4
#   NIR          | SR_B4       | SR_B4        | SR_B5       | SR_B5
#   SWIR1        | SR_B5       | SR_B5        | SR_B6       | SR_B6
#   SWIR2        | SR_B7       | SR_B7        | SR_B7       | SR_B7

LANDSAT_BANDS = {
    "Red": {
        "name": "Red",
        "wavelength": "630-690 nm",
        "resolution": 30,
        "description": "Visible Red band, useful for vegetation discrimination and NDVI"
    },
    "NIR": {
        "name": "Near Infrared",
        "wavelength": "770-900 nm",
        "resolution": 30,
        "description": "NIR band, primary indicator of vegetation vigor (≈ Sentinel-2 B8)"
    },
    "SWIR1": {
        "name": "Shortwave Infrared 1",
        "wavelength": "1550-1750 nm",
        "resolution": 30,
        "description": "SWIR1 band, sensitive to leaf water content (≈ Sentinel-2 B8A/B11)"
    },
    "SWIR2": {
        "name": "Shortwave Infrared 2",
        "wavelength": "2080-2350 nm",
        "resolution": 30,
        "description": "SWIR2 band, soil/mineral discrimination (≈ Sentinel-2 B12)"
    }
}

# Sensor-specific band name mapping (GEE Collection 2 Level-2 SR band names)
SENSOR_BAND_MAPPING = {
    "L5": {  # Landsat 5 TM    (1984–2013)
        "collection": "LANDSAT/LT05/C02/T1_L2",
        "Red": "SR_B3", "NIR": "SR_B4", "SWIR1": "SR_B5", "SWIR2": "SR_B7",
        "qa_band": "QA_PIXEL",
        "scale_factor": 0.0000275,   # C2 L2 scaling
        "offset": -0.2,
    },
    "L7": {  # Landsat 7 ETM+  (1999–present, SLC-off after May 2003)
        "collection": "LANDSAT/LE07/C02/T1_L2",
        "Red": "SR_B3", "NIR": "SR_B4", "SWIR1": "SR_B5", "SWIR2": "SR_B7",
        "qa_band": "QA_PIXEL",
        "scale_factor": 0.0000275,
        "offset": -0.2,
    },
    "L8": {  # Landsat 8 OLI   (2013–present)
        "collection": "LANDSAT/LC08/C02/T1_L2",
        "Red": "SR_B4", "NIR": "SR_B5", "SWIR1": "SR_B6", "SWIR2": "SR_B7",
        "qa_band": "QA_PIXEL",
        "scale_factor": 0.0000275,
        "offset": -0.2,
    },
    "L9": {  # Landsat 9 OLI-2 (2021–present)
        "collection": "LANDSAT/LC09/C02/T1_L2",
        "Red": "SR_B4", "NIR": "SR_B5", "SWIR1": "SR_B6", "SWIR2": "SR_B7",
        "qa_band": "QA_PIXEL",
        "scale_factor": 0.0000275,
        "offset": -0.2,
    },
}

# Which sensors to use for each year range
# Priority is given to sensors with better data quality
YEAR_SENSOR_MAPPING = {
    # 2010-2012: Landsat 5 primary, Landsat 7 fallback
    (2010, 2012): ["L5", "L7"],
    # 2013: Landsat 5 ended, transition to Landsat 8
    (2013, 2013): ["L8", "L7"],
    # 2014-2020: Landsat 8 primary, Landsat 7 fallback
    (2014, 2020): ["L8", "L7"],
    # 2021+: Landsat 8 + Landsat 9
    (2021, 2030): ["L8", "L9"],
}

# ─── Wheat Season Configuration ─────────────────────────────────────────────

WHEAT_SEASON = {
    "sowing_start_month": 10,      # October
    "sowing_end_month": 11,        # November
    "harvest_start_month": 3,      # March
    "harvest_end_month": 4,        # April
    "growing_season_days": 140     # ~140-150 days in India
}

# Time series configuration
TIME_SERIES_CONFIG = {
    "composite_days": 5,           # 5-day mean composite
    "total_time_steps": 42,        # ~42 composites over Oct–Apr
    "start_doy": 274,              # DOY of October 1
    "end_doy": 120                 # DOY of April 30
}

# ─── Quality Control ────────────────────────────────────────────────────────

QUALITY_CONTROL = {
    "max_cloud_cover": 30,         # Max cloud cover % per scene
    "min_valid_pixels": 0.7,       # Min fraction of valid pixels
    "outlier_std_threshold": 3     # Std-devs for outlier detection
}

# ─── Processing Configuration ───────────────────────────────────────────────

# 2010–2025  (wheat season year = sowing year, so 2025 → Oct 2025 – Apr 2026)
PROCESSING_YEARS = list(range(2010, 2026))

EXPORT_CONFIG = {
    "scale": 30,                   # Landsat native resolution
    "crs": "EPSG:4326",
    "format": "GeoTIFF"
}

# ─── Summary ────────────────────────────────────────────────────────────────

print(f"Landsat Pipeline Configuration loaded successfully!")
print(f"Years to process: {PROCESSING_YEARS[0]}-{PROCESSING_YEARS[-1]} ({len(PROCESSING_YEARS)} years)")
print(f"Total districts: {sum(len(d) for d in ALL_DISTRICTS.values())}")
print(f"  - Haryana: {len(HARYANA_DISTRICTS)} districts")
print(f"  - Punjab: {len(PUNJAB_DISTRICTS)} districts")
print(f"  - Uttar Pradesh: {len(UTTAR_PRADESH_DISTRICTS)} districts")
