# Wheat Yield Prediction Pipeline
## Sentinel-2 Remote Sensing Data Processing


---

## 📖 Overview

This pipeline extracts and processes **Sentinel-2 Remote Sensing Data** for wheat yield prediction using Google Earth Engine.

**Main Components:**
1. **`gee_extractor.py`** - Extract Sentinel-2 data (4 spectral bands) at 5-day composites
2. **`fill_missing_sma.py`** - Fill missing values using Simple Moving Average
3. **`process_all_districts.py`** - Orchestrate multi-year data extraction and processing for all districts

### Target Regions
- **Haryana** - 22 districts
- **Punjab** - 23 districts  
- **Uttar Pradesh** - 75 districts

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Google Earth Engine account (for Sentinel-2 data)

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

### Sentinel-2 Bands

| Band | Name | Wavelength | Resolution |
|------|------|------------|------------|
| B7 | Vegetation Red Edge | 773-793 nm | 20m |
| B8 | NIR | 785-900 nm | 10m |
| B8A | Vegetation Red Edge | 855-875 nm | 20m |
| B12 | SWIR | 2100-2280 nm | 20m |

### Time Series Configuration

- **Remote Sensing**: ~42 time steps (5-day composites over ~210 day growing season)
- **Growing Season**: October to April (Indian wheat season)

---

## 🚀 Usage

### 1. Extract Sentinel-2 Data (Single District Test)

```bash
python gee_extractor.py
```

This will extract data for Karnal, Haryana (2022) as a test run.

### 2. Fill Missing Values in Extracted Data

```bash
python fill_missing_sma.py sentinel2_Haryana_Karnal_2022.csv
```

Options:
- `-o, --output` - Specify output file (default: input_filled.csv)
- `-w, --window` - SMA window size (default: 3)
- `-m, --method` - Filling method: sma, linear, polynomial (default: sma)

### 3. Process All Districts (Multi-Year Pipeline)

```bash
# Test mode: Process only Karnal for years 2022-2023
python process_all_districts.py --test

# Full processing: All districts, years 2017-2024
python process_all_districts.py

# Specify custom years
python process_all_districts.py --years 2020 2021 2022

# Adjust rate limiting (delay between extractions in seconds)
python process_all_districts.py --rate-limit 15
```

---

## 📁 Project Structure

```
sentinel2_wheat_pipeline/
├── config.py                    # Configuration parameters
├── gee_extractor.py             # Sentinel-2 data extraction from GEE
├── fill_missing_sma.py          # Fill missing values with Simple Moving Average
├── process_all_districts.py    # Multi-year processing pipeline for all districts
├── __init__.py                  # Python package initialization
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/                        # Data directory (created automatically)
│   ├── raw/                     # Raw extracted data
│   └── processed/               # Preprocessed datasets
└── output/                      # Output directory (created by process_all_districts.py)
    ├── raw/                     # Raw CSV files per district/year
    ├── filled/                  # SMA-filled CSV files per district/year
    ├── merged/                  # Merged multi-year CSV files per district
    └── logs/                    # Processing logs and error reports
```

---

## 📐 Output Data Format

### Individual Year CSV (Raw Extraction)

**File naming:** `{state}_{district}_{year}_raw.csv`

```csv
state,district,year,time_step,start_date,end_date,B7,B8,B8A,B12,image_count,mean_cloud_cover,quality_flag
Haryana,Karnal,2022,0,2022-10-01,2022-10-06,0.1234,0.4567,0.3456,0.0987,3,15.2,valid
Haryana,Karnal,2022,1,2022-10-06,2022-10-11,0.1289,0.4623,0.3501,0.1012,2,18.5,valid
...
```

### Filled CSV (After SMA Processing)

**File naming:** `{state}_{district}_{year}_filled.csv`

Same format as raw, but with missing values filled using Simple Moving Average.

### Merged Multi-Year CSV (Final District Output)

**File naming:** `{state}_{district}_remote_sensing_data.csv`

Contains all years merged into a single file per district, sorted by year and time_step.

---

## 🌾 District List

### Haryana (22 Districts)
Ambala, Bhiwani, Charkhi Dadri, Faridabad, Fatehabad, Gurugram, Hisar, Jhajjar, Jind, Kaithal, Karnal, Kurukshetra, Mahendragarh, Nuh, Palwal, Panchkula, Panipat, Rewari, Rohtak, Sirsa, Sonipat, Yamunanagar

### Punjab (23 Districts)
Amritsar, Barnala, Bathinda, Faridkot, Fatehgarh Sahib, Fazilka, Ferozepur, Gurdaspur, Hoshiarpur, Jalandhar, Kapurthala, Ludhiana, Malerkotla, Mansa, Moga, Muktsar, Pathankot, Patiala, Rupnagar, SAS Nagar, Sangrur, Shaheed Bhagat Singh Nagar, Tarn Taran

### Uttar Pradesh (75 Districts - All districts)
**Western UP:** Saharanpur, Muzaffarnagar, Shamli, Meerut, Baghpat, Ghaziabad, Gautam Buddha Nagar, Bulandshahr, Hapur, Amroha, Moradabad, Bijnor, Sambhal, Rampur, Bareilly, Pilibhit, Shahjahanpur, Aligarh, Hathras, Mathura, Agra, Firozabad, Etah, Mainpuri, Kasganj

**Central UP:** Lucknow, Unnao, Rae Bareli, Sitapur, Hardoi, Lakhimpur Kheri, Kanpur Nagar, Kanpur Dehat, Farrukhabad, Kannauj, Auraiya, Etawah, Fatehpur

**Eastern UP:** Allahabad, Kaushambi, Pratapgarh, Jaunpur, Varanasi, Ghazipur, Chandauli, Mirzapur, Sonbhadra, Bhadohi, Ballia, Mau, Azamgarh, Ambedkar Nagar, Sultanpur, Faizabad, Amethi, Barabanki, Gonda, Bahraich, Shrawasti, Balrampur, Siddharthnagar, Basti, Sant Kabir Nagar, Maharajganj, Gorakhpur, Kushinagar, Deoria

**Bundelkhand (Southern UP):** Jhansi, Jalaun, Lalitpur, Mahoba, Hamirpur, Banda, Chitrakoot

---

## 🔗 Data Sources

1. **Sentinel-2 Data**: Google Earth Engine (`COPERNICUS/S2_HARMONIZED`)
2. **Administrative Boundaries**: FAO GAUL dataset on GEE
3. **Cropland Mask**: ESA WorldCover 10m 2021

---

## ⚠️ Notes

1. **GEE Quota**: Large-scale extraction may take time due to Google Earth Engine rate limits
2. **Rate Limiting**: The `process_all_districts.py` script includes configurable delays between extractions to avoid quota issues
3. **Data Quality**: All extractions include quality metrics (image count, cloud cover, quality flags)
4. **Missing Values**: Use `fill_missing_sma.py` to fill gaps in the time series data
5. **Yield Data**: Historical yield data needs to be collected separately from state agriculture departments for model training

