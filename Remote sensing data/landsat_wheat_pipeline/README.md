# Wheat Yield Prediction Pipeline
## Landsat Remote Sensing Data Processing (2010–2026)

---

## 📖 Overview

This pipeline extracts and processes **Landsat surface reflectance data** for wheat yield prediction using Google Earth Engine. It provides **seamless coverage from 2010 to 2026** by combining four Landsat sensors.

**Main Components:**
1. **`gee_extractor.py`** – Extract multi-sensor Landsat data (4 bands) at 5-day composites
2. **`fill_missing_sma.py`** – Fill missing values using Simple Moving Average
3. **`process_all_districts.py`** – Orchestrate multi-year extraction and processing for all districts

### Target Regions
- **Haryana** – 22 districts
- **Punjab** – 23 districts
- **Uttar Pradesh** – 75 districts

### Landsat Sensor Coverage

| Sensor | Availability | GEE Collection | Usage in Pipeline |
|--------|-------------|----------------|-------------------|
| Landsat 5 TM | 1984–2013 | `LANDSAT/LT05/C02/T1_L2` | Primary for 2010–2012 |
| Landsat 7 ETM+ | 1999–present | `LANDSAT/LE07/C02/T1_L2` | Gap-filler (SLC-off after 2003) |
| Landsat 8 OLI | 2013–present | `LANDSAT/LC08/C02/T1_L2` | Primary for 2013–2020 |
| Landsat 9 OLI-2 | 2021–present | `LANDSAT/LC09/C02/T1_L2` | Combined with L8 for 2021+ |

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Google Earth Engine account

### Install Dependencies

```bash
pip install numpy pandas requests earthengine-api scikit-learn
```

### Authenticate Google Earth Engine

```bash
earthengine authenticate
```

---

## 📊 Data Specifications

### Harmonized Output Bands

All sensors are harmonized to these common band names:

| Output Name | Wavelength | Sentinel-2 Equivalent | Description |
|-------------|------------|----------------------|-------------|
| Red | 630-690 nm | B4 | Visible red, vegetation discrimination |
| NIR | 770-900 nm | B8 | Near Infrared, vegetation vigor |
| SWIR1 | 1550-1750 nm | B11/B8A | Shortwave IR, leaf water content |
| SWIR2 | 2080-2350 nm | B12 | Shortwave IR, soil/mineral |

### Cross-Sensor Band Mapping

| Output | L5 TM | L7 ETM+ | L8 OLI | L9 OLI-2 |
|--------|-------|---------|--------|----------|
| Red | SR_B3 | SR_B3 | SR_B4 | SR_B4 |
| NIR | SR_B4 | SR_B4 | SR_B5 | SR_B5 |
| SWIR1 | SR_B5 | SR_B5 | SR_B6 | SR_B6 |
| SWIR2 | SR_B7 | SR_B7 | SR_B7 | SR_B7 |

### Time Series Configuration

- **Resolution**: 30m (Landsat native)
- **Composites**: ~42 time steps (5-day composites over ~210 day growing season)
- **Growing Season**: October to April (Indian wheat season)
- **Data Level**: Collection 2 Level-2 Surface Reflectance

---

## 🚀 Usage

### 1. Extract Landsat Data (Single District Test)

```bash
python gee_extractor.py
```

Tests with Karnal, Haryana (2015).

### 2. Fill Missing Values

```bash
python fill_missing_sma.py landsat_Haryana_Karnal_2015.csv
```

Options:
- `-o, --output` – Specify output file
- `-w, --window` – SMA window size (default: 3)
- `-m, --method` – Filling method: sma, linear, polynomial

### 3. Process All Districts (Multi-Year Pipeline)

```bash
# Test mode: Karnal only, years 2015-2016
python process_all_districts.py --test

# Full processing: All districts, years 2010-2025
python process_all_districts.py

# Custom years
python process_all_districts.py --years 2010 2011 2012 2013

# Adjust rate limiting
python process_all_districts.py --rate-limit 15
```

---

## 📁 Project Structure

```
landsat_wheat_pipeline/
├── config.py                    # Configuration (bands, districts, sensor mapping)
├── gee_extractor.py             # Multi-sensor Landsat data extraction from GEE
├── fill_missing_sma.py          # Fill missing values with SMA
├── process_all_districts.py     # Multi-year processing pipeline
├── __init__.py                  # Package initialization
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── output/                      # Output directory (created automatically)
    ├── raw/                     # Raw CSV files per district/year
    ├── filled/                  # SMA-filled CSV files
    ├── merged/                  # Merged multi-year CSV per district
    └── logs/                    # Processing logs and error reports
```

---

## 📐 Output Data Format

### Individual Year CSV (Raw)

**File naming:** `{state}_{district}_{year}_raw.csv`

```csv
state,district,year,time_step,start_date,end_date,Red,NIR,SWIR1,SWIR2,image_count,mean_cloud_cover,quality_flag
Haryana,Karnal,2015,0,2015-10-01,2015-10-06,0.0812,0.2534,0.1876,0.0987,2,12.5,valid
```

### Merged Multi-Year CSV

**File naming:** `{state}_{district}_remote_sensing_data.csv`

Contains all years merged, sorted by year and time_step.

---

## 🔗 Data Sources

1. **Landsat Data**: Google Earth Engine (USGS Collection 2 Level-2)
2. **Administrative Boundaries**: FAO GAUL 2015 dataset on GEE
3. **Cropland Mask**: ESA WorldCover 10m 2021

---

## ⚠️ Notes

1. **Landsat 7 SLC-off**: After May 2003, Landsat 7 ETM+ has scan-line striping. The pipeline uses it only as a fallback sensor, and 5-day compositing helps minimize the artifact impact.
2. **GEE Quota**: Large-scale extraction may take time due to rate limits.
3. **Rate Limiting**: Configurable delays between extractions (default: 10s).
4. **Resume Capability**: The pipeline skips already-processed files on restart.
5. **Data Quality**: All extractions include quality metrics (image count, cloud cover, quality flags).
6. **Band Scaling**: Collection 2 Level-2 surface reflectance scaling (DN × 0.0000275 − 0.2) is applied automatically.
