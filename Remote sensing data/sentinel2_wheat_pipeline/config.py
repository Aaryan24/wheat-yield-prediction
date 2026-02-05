import os
from datetime import datetime

# Data Directory Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Study Region Configuration (Districts of Haryana, Punjab, Uttar Pradesh)

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

# Uttar Pradesh Districts (all 75 districts )
# Organized by region for clarity
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

# Combine all districts
ALL_DISTRICTS = {
    "Haryana": HARYANA_DISTRICTS,
    "Punjab": PUNJAB_DISTRICTS,
    "Uttar_Pradesh": UTTAR_PRADESH_DISTRICTS
}



# Band 7 (773-793nm), Band 8 (785-900nm), Band 8A (855-875nm), Band 12 (2100-2280nm)
SENTINEL2_BANDS = {
    "B7": {
        "name": "Band 7 - Vegetation Red Edge",
        "wavelength": "773-793 nm",
        "resolution": 20,
        "gee_name": "B7"
    },
    "B8": {
        "name": "Band 8 - NIR",
        "wavelength": "785-900 nm",
        "resolution": 10,
        "gee_name": "B8"
    },
    "B8A": {
        "name": "Band 8A - Vegetation Red Edge",
        "wavelength": "855-875 nm",
        "resolution": 20,
        "gee_name": "B8A"
    },
    "B12": {
        "name": "Band 12 - SWIR",
        "wavelength": "2100-2280 nm",
        "resolution": 20,
        "gee_name": "B12"
    }
}

# Additional useful bands for cloud masking
CLOUD_MASK_BANDS = ["QA60"]

# Sentinel-2 Image Collections
SENTINEL2_TOA_COLLECTION = "COPERNICUS/S2"           # Level-1C (TOA)
SENTINEL2_SR_COLLECTION = "COPERNICUS/S2_SR"         # Level-2A (Surface Reflectance)

# Wheat growing season in North India
# Sowing: October-November
# Harvesting: March-April

WHEAT_SEASON = {
    "sowing_start_month": 10,      # October
    "sowing_end_month": 11,        # November
    "harvest_start_month": 3,      # March
    "harvest_end_month": 4,        # April
    "growing_season_days": 140     # Approximately 140-150 days in India
}

# Time series configuration
TIME_SERIES_CONFIG = {
    "composite_days": 5,           # 5-day mean composite 
    "total_time_steps": 50,        # Number of time steps covering growing season
    "start_doy": 274,              # DOY of October 1 (approximate sowing start)
    "end_doy": 120                 # DOY of April 30 (approximate harvest end)
}

# Weather Forecast Data Configuration 

WEATHER_PARAMETERS = {
    "max_temp_day": {
        "name": "Maximum Temperature (Day)",
        "unit": "°C"
    },
    "min_temp_night": {
        "name": "Minimum Temperature (Night)", 
        "unit": "°C"
    },
    "wind_speed_day": {
        "name": "Wind Speed (Day)",
        "unit": "m/s"
    },
    "wind_speed_night": {
        "name": "Wind Speed (Night)",
        "unit": "m/s"
    },
    "precipitation_day": {
        "name": "Precipitation (Day)",
        "unit": "mm"
    },
    "precipitation_night": {
        "name": "Precipitation (Night)",
        "unit": "mm"
    }
}



# Quality Control Parameters

QUALITY_CONTROL = {
    "max_cloud_cover": 20,         # Maximum cloud cover percentage
    "min_valid_pixels": 0.7,       # Minimum fraction of valid pixels (70%)
    "outlier_std_threshold": 3     # Standard deviations for outlier detection
}


# Years to process (adjust based on available data)
PROCESSING_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]



# Export Configuration

EXPORT_CONFIG = {
    "scale": 10,                   # Export resolution in meters
    "crs": "EPSG:4326",           # Coordinate reference system
    "format": "GeoTIFF"           # Export format
}

print(f"Configuration loaded successfully!")
print(f"Total districts to process: {sum(len(d) for d in ALL_DISTRICTS.values())}")
print(f"  - Haryana: {len(HARYANA_DISTRICTS)} districts")
print(f"  - Punjab: {len(PUNJAB_DISTRICTS)} districts") 
print(f"  - Uttar Pradesh: {len(UTTAR_PRADESH_DISTRICTS)} districts")
