"""
Landsat Data Extractor for Wheat Yield Prediction
==================================================

Extracts multi-sensor Landsat surface reflectance data via Google Earth Engine.
Combines Landsat 5/7/8/9 for seamless coverage from 2010 to 2026.
"""

import ee
import io
import os
import json
import time
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    _HAS_DRIVE_API = True
except ImportError:
    _HAS_DRIVE_API = False

# Import configuration
try:
    from config import (
        LANDSAT_BANDS, SENSOR_BAND_MAPPING, YEAR_SENSOR_MAPPING,
        ALL_DISTRICTS, WHEAT_SEASON, TIME_SERIES_CONFIG, QUALITY_CONTROL,
        PROCESSING_YEARS, EXPORT_CONFIG, PROCESSED_DATA_DIR, OUTPUT_DIR
    )
except ImportError:
    print("Warning: config.py not found. Using default values.")
    PROCESSING_YEARS = [2020, 2021, 2022]


class LandsatExtractor:
    """
    Extracts harmonized Landsat time-series data for wheat yield prediction.

    Automatically selects the appropriate Landsat sensor(s) for each year
    and harmonizes bands to common names: NIR, SWIR1, SWIR2, Red.
    """

    # Common output band names (harmonized across all sensors)
    OUTPUT_BANDS = ["Red", "NIR", "SWIR1", "SWIR2"]

    # Batch export settings
    EXACT_EXPORT_FLOAT_FORMAT = "%.17g"
    BATCH_EXPORT_DRIVE_FOLDER = "landsat_wheat_pipeline_exports"

    # District coordinates (lat, lon) for fallback geometry
    DISTRICT_COORDS = {
        # Haryana
        "Karnal": (29.6857, 76.9905),
        "Ambala": (30.3782, 76.7767),
        "Bhiwani": (28.7939, 76.1393),
        "Hisar": (29.1492, 75.7217),
        "Jind": (29.3165, 76.3152),
        "Kaithal": (29.8015, 76.3998),
        "Kurukshetra": (29.9695, 76.8783),
        "Panipat": (29.3909, 76.9635),
        "Rohtak": (28.8955, 76.6066),
        "Sirsa": (29.5330, 75.0287),
        "Sonipat": (28.9931, 77.0151),
        "Yamunanagar": (30.1290, 77.2674),
        "Fatehabad": (29.5151, 75.4559),
        "Gurugram": (28.4595, 77.0266),
        "Jhajjar": (28.6063, 76.6566),
        "Mahendragarh": (28.2791, 76.1540),
        "Rewari": (28.1969, 76.6192),
        "Faridabad": (28.4089, 77.3178),
        "Palwal": (28.1487, 77.3320),
        "Nuh": (28.1018, 77.0017),
        "Panchkula": (30.6942, 76.8606),
        "Charkhi Dadri": (28.5921, 76.2711),
        # Punjab
        "Amritsar": (31.6340, 74.8723),
        "Ludhiana": (30.9010, 75.8573),
        "Patiala": (30.3398, 76.3869),
        "Jalandhar": (31.3260, 75.5762),
        "Bathinda": (30.2110, 74.9455),
        "Sangrur": (30.2506, 75.8448),
        "Ferozepur": (30.9331, 74.6225),
        "Moga": (30.8160, 75.1740),
        "Barnala": (30.3819, 75.5472),
        "Mansa": (29.9986, 75.4027),
        "Muktsar": (30.4731, 74.5160),
        "Faridkot": (30.6774, 74.7583),
        "Fazilka": (30.4036, 74.0278),
        "Gurdaspur": (32.0414, 75.4031),
        "Hoshiarpur": (31.5143, 75.9115),
        "Kapurthala": (31.3803, 75.3820),
        "Pathankot": (32.2643, 75.6421),
        "Rupnagar": (31.0392, 76.5266),
        "SAS Nagar": (30.7046, 76.7179),
        "Fatehgarh Sahib": (30.6454, 76.3932),
        "Tarn Taran": (31.4519, 74.9279),
        "Malerkotla": (30.5288, 75.8792),
        "Shaheed Bhagat Singh Nagar": (31.1123, 76.1047),
        # Western UP
        "Saharanpur": (29.9680, 77.5510),
        "Muzaffarnagar": (29.4727, 77.7085),
        "Shamli": (29.4493, 77.3148),
        "Meerut": (28.9845, 77.7064),
        "Baghpat": (28.9449, 77.2195),
        "Ghaziabad": (28.6692, 77.4538),
        "Gautam Buddha Nagar": (28.5706, 77.3262),
        "Bulandshahr": (28.4070, 77.8489),
        "Hapur": (28.7437, 77.7628),
        "Amroha": (28.9044, 78.4672),
        "Moradabad": (28.8389, 78.7768),
        "Bijnor": (29.3724, 78.1313),
        "Sambhal": (28.5839, 78.5699),
        "Rampur": (28.8156, 79.0250),
        "Bareilly": (28.3670, 79.4304),
        "Pilibhit": (28.6382, 79.8039),
        "Shahjahanpur": (27.8805, 79.9117),
        "Aligarh": (27.8974, 78.0880),
        "Hathras": (27.5906, 78.0522),
        "Mathura": (27.4924, 77.6737),
        "Agra": (27.1767, 78.0081),
        "Firozabad": (27.1503, 78.3955),
        "Etah": (27.5587, 78.6656),
        "Mainpuri": (27.2344, 79.0248),
        "Kasganj": (27.8085, 78.6469),
        # Central UP
        "Lucknow": (26.8467, 80.9462),
        "Unnao": (26.5393, 80.4880),
        "Rae Bareli": (26.2345, 81.2440),
        "Sitapur": (27.5726, 80.6828),
        "Hardoi": (27.3954, 80.1311),
        "Lakhimpur Kheri": (27.9462, 80.7801),
        "Kanpur Nagar": (26.4499, 80.3319),
        "Kanpur Dehat": (26.4048, 79.9520),
        "Farrukhabad": (27.3906, 79.5807),
        "Kannauj": (27.0545, 79.9220),
        "Auraiya": (26.4655, 79.5091),
        "Etawah": (26.7856, 79.0159),
        "Fatehpur": (25.9304, 80.8130),
        # Eastern UP
        "Allahabad": (25.4358, 81.8463),
        "Kaushambi": (25.5319, 81.3748),
        "Pratapgarh": (25.8971, 81.9419),
        "Jaunpur": (25.7464, 82.6836),
        "Varanasi": (25.3176, 82.9739),
        "Ghazipur": (25.5878, 83.5883),
        "Chandauli": (25.2604, 83.2639),
        "Mirzapur": (25.1460, 82.5690),
        "Sonbhadra": (24.6884, 83.0643),
        "Bhadohi": (25.3944, 82.5652),
        "Ballia": (25.7607, 84.1487),
        "Mau": (25.9417, 83.5610),
        "Azamgarh": (26.0735, 83.1868),
        "Ambedkar Nagar": (26.4005, 82.6513),
        "Sultanpur": (26.2648, 82.0722),
        "Faizabad": (26.7735, 82.1447),
        "Amethi": (26.1542, 81.8268),
        "Barabanki": (26.9320, 81.1868),
        "Gonda": (27.1361, 81.9610),
        "Bahraich": (27.5747, 81.5960),
        "Shrawasti": (27.5029, 82.0153),
        "Balrampur": (27.4300, 82.1803),
        "Siddharthnagar": (27.2900, 83.0900),
        "Basti": (26.8030, 82.7274),
        "Sant Kabir Nagar": (26.7894, 83.0386),
        "Maharajganj": (27.1191, 83.5616),
        "Gorakhpur": (26.7606, 83.3732),
        "Kushinagar": (26.7397, 83.8844),
        "Deoria": (26.5024, 83.7791),
        # Bundelkhand
        "Jhansi": (25.4484, 78.5685),
        "Jalaun": (26.1469, 79.3358),
        "Lalitpur": (24.6877, 78.4163),
        "Mahoba": (25.2924, 79.8716),
        "Hamirpur": (25.9569, 80.1514),
        "Banda": (25.4754, 80.3355),
        "Chitrakoot": (25.2020, 80.8521),
    }

    def __init__(self, initialize_ee: bool = True):
        """Initialize the Landsat Extractor."""
        if initialize_ee:
            try:
                ee.Initialize(project='ugp-prediction')
                print("Earth Engine initialized successfully!")
            except Exception as e:
                print(f"Error initializing Earth Engine: {e}")
                print("Please run 'earthengine authenticate' first.")
                raise

        # Load India administrative boundaries (FAO GAUL)
        self.india_admin = ee.FeatureCollection("FAO/GAUL/2015/level2")

        # Quality control
        self.max_cloud_percentage = QUALITY_CONTROL.get("max_cloud_cover", 30)

        # Load cropland mask (ESA WorldCover 2021, cropland = 40)
        try:
            self.cropland_mask = ee.ImageCollection('ESA/WorldCover/v200').first().eq(40)
            print("Cropland mask loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load cropland mask: {e}")
            self.cropland_mask = None

        # Cached Drive service for batch exports
        self._drive_service = None

    # ─── Sensor Selection ────────────────────────────────────────────────

    def get_sensors_for_year(self, year: int) -> List[str]:
        """
        Determine which Landsat sensor(s) to use for a given year.

        Returns list of sensor keys (e.g. ["L8", "L9"]) in priority order.
        """
        for (start, end), sensors in YEAR_SENSOR_MAPPING.items():
            if start <= year <= end:
                return sensors
        # Fallback: use Landsat 8
        return ["L8"]

    # ─── Band Harmonization ─────────────────────────────────────────────

    def _harmonize_image(self, image: ee.Image, sensor_key: str) -> ee.Image:
        """
        Rename sensor-specific bands to common output names and apply
        Collection 2 Level-2 surface reflectance scaling.

        Returns image with bands: Red, NIR, SWIR1, SWIR2
        """
        mapping = SENSOR_BAND_MAPPING[sensor_key]
        scale_factor = mapping["scale_factor"]
        offset = mapping["offset"]

        # Select and rename bands
        src_bands = [mapping[b] for b in self.OUTPUT_BANDS]
        harmonized = image.select(src_bands, self.OUTPUT_BANDS)

        # Apply C2 L2 scaling:  reflectance = DN * scale_factor + offset
        harmonized = harmonized.multiply(scale_factor).add(offset)

        # Copy properties for quality tracking
        return harmonized.copyProperties(image, ['system:time_start', 'CLOUD_COVER'])

    # ─── Cloud Masking ───────────────────────────────────────────────────

    def mask_clouds_landsat(self, image: ee.Image) -> ee.Image:
        """
        Cloud masking using QA_PIXEL bitmask (Landsat Collection 2 Level-2).

        QA_PIXEL bit flags:
          Bit 1: Dilated Cloud
          Bit 3: Cloud
          Bit 4: Cloud Shadow
          Bit 5: Snow
        """
        qa = image.select('QA_PIXEL')

        # Create mask where these bits are all zero (clear conditions)
        dilated_cloud = 1 << 1
        cloud = 1 << 3
        cloud_shadow = 1 << 4

        mask = (qa.bitwiseAnd(dilated_cloud).eq(0)
                .And(qa.bitwiseAnd(cloud).eq(0))
                .And(qa.bitwiseAnd(cloud_shadow).eq(0)))

        return image.updateMask(mask)

    # ─── Geometry ────────────────────────────────────────────────────────

    def get_district_geometry(self, state: str, district: str,
                              use_buffer: bool = False) -> ee.Geometry:
        """
        Get district geometry from FAO GAUL boundaries.
        Falls back to 25 km buffered point if boundaries unavailable.
        """
        state_mapping = {
            "Haryana": "Haryana",
            "Punjab": "Punjab",
            "Uttar_Pradesh": "Uttar Pradesh",
            "Uttar Pradesh": "Uttar Pradesh"
        }
        gaul_state = state_mapping.get(state, state)

        try:
            district_fc = self.india_admin.filter(
                ee.Filter.And(
                    ee.Filter.eq('ADM1_NAME', gaul_state),
                    ee.Filter.eq('ADM2_NAME', district)
                )
            )
            size = district_fc.size().getInfo()
            if size > 0 and not use_buffer:
                print(f"  Using actual boundaries for {district}")
                return district_fc.geometry()
            else:
                raise Exception("District not found in GAUL, using fallback")
        except Exception:
            if district in self.DISTRICT_COORDS:
                print(f"  Using 25km buffer fallback for {district}")
                lat, lon = self.DISTRICT_COORDS[district]
                point = ee.Geometry.Point([lon, lat])
                return point.buffer(25000)
            else:
                raise Exception(f"No geometry available for {district}, {state}")

    # ─── Season Dates ────────────────────────────────────────────────────

    def get_wheat_season_dates(self, year: int) -> Tuple[str, str]:
        """
        Get wheat season dates.
        Sowing: October of `year`  →  Harvest: April of `year+1`
        """
        return f"{year}-10-01", f"{year + 1}-04-30"

    # ─── Collection Building ─────────────────────────────────────────────

    def _build_collection(self, geometry: ee.Geometry, start_date: str,
                          end_date: str, year: int) -> ee.ImageCollection:
        """
        Build a merged, harmonized, cloud-masked image collection from
        all applicable Landsat sensors for the given year.
        """
        sensors = self.get_sensors_for_year(year)
        collections = []

        for sensor_key in sensors:
            mapping = SENSOR_BAND_MAPPING[sensor_key]
            collection_id = mapping["collection"]

            # Get the bands we need (sensor-specific names) + QA_PIXEL
            src_bands = [mapping[b] for b in self.OUTPUT_BANDS] + ["QA_PIXEL"]

            try:
                col = (ee.ImageCollection(collection_id)
                       .filterBounds(geometry)
                       .filterDate(start_date, end_date)
                       .filter(ee.Filter.lt('CLOUD_COVER', self.max_cloud_percentage))
                       .select(src_bands))

                # Apply cloud masking then harmonize bands
                sensor = sensor_key  # capture for lambda
                col = col.map(self.mask_clouds_landsat)
                col = col.map(lambda img, sk=sensor: self._harmonize_image(img, sk))

                collections.append(col)
            except Exception as e:
                print(f"  Warning: Could not load {collection_id}: {e}")

        if not collections:
            raise Exception(f"No Landsat data available for year {year}")

        # Merge all sensor collections into one
        merged = collections[0]
        for col in collections[1:]:
            merged = merged.merge(col)

        return merged

    # ─── Compositing ─────────────────────────────────────────────────────

    def create_time_series_composites(
        self,
        geometry: ee.Geometry,
        year: int,
        composite_days: int = 5
    ) -> List[Dict]:
        """
        Create 5-day mean composites for the wheat growing season.
        """
        start_date, end_date = self.get_wheat_season_dates(year)

        # Build merged multi-sensor collection
        collection = self._build_collection(geometry, start_date, end_date, year)

        # Apply cropland mask if available
        if self.cropland_mask is not None:
            collection = collection.map(lambda img: img.updateMask(self.cropland_mask))

        composites = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        time_step = 0
        while current_date < end:
            next_date = current_date + timedelta(days=composite_days)

            period_collection = collection.filterDate(
                current_date.strftime("%Y-%m-%d"),
                next_date.strftime("%Y-%m-%d")
            )

            image_count = period_collection.size()
            composite = period_collection.mean()
            mean_cloud = period_collection.aggregate_mean('CLOUD_COVER')

            composites.append({
                'time_step': time_step,
                'start_date': current_date.strftime("%Y-%m-%d"),
                'end_date': next_date.strftime("%Y-%m-%d"),
                'composite': composite,
                'image_count': image_count,
                'mean_cloud_cover': mean_cloud
            })

            current_date = next_date
            time_step += 1

        return composites

    # ─── Extraction ──────────────────────────────────────────────────────

    def extract_district_time_series(
        self,
        state: str,
        district: str,
        year: int,
        scale: int = 30   # Landsat native resolution
    ) -> Optional[Dict]:
        """
        Extract time series data for a specific district with quality tracking.
        """
        sensors = self.get_sensors_for_year(year)
        print(f"\nExtracting data for {district}, {state} - Year {year} "
              f"(sensors: {', '.join(sensors)})...")

        try:
            geometry = self.get_district_geometry(state, district)
            composites = self.create_time_series_composites(geometry, year)

            time_series_data = []
            valid_composites = 0
            failed_composites = 0

            for comp in composites:
                try:
                    image_count = comp['image_count'].getInfo()
                    mean_cloud = comp['mean_cloud_cover'].getInfo()

                    if image_count == 0:
                        time_series_data.append({
                            'time_step': comp['time_step'],
                            'start_date': comp['start_date'],
                            'end_date': comp['end_date'],
                            'Red': None, 'NIR': None,
                            'SWIR1': None, 'SWIR2': None,
                            'image_count': image_count,
                            'mean_cloud_cover': mean_cloud,
                            'quality_flag': 'insufficient_data'
                        })
                        failed_composites += 1
                        continue

                    # Compute mean over district
                    stats = comp['composite'].reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=scale,
                        maxPixels=1e13,
                        bestEffort=False
                    ).getInfo()

                    red = stats.get('Red')
                    nir = stats.get('NIR')
                    swir1 = stats.get('SWIR1')
                    swir2 = stats.get('SWIR2')

                    if all(v is not None for v in [red, nir, swir1, swir2]):
                        time_series_data.append({
                            'time_step': comp['time_step'],
                            'start_date': comp['start_date'],
                            'end_date': comp['end_date'],
                            'Red': red, 'NIR': nir,
                            'SWIR1': swir1, 'SWIR2': swir2,
                            'image_count': image_count,
                            'mean_cloud_cover': mean_cloud,
                            'quality_flag': 'valid'
                        })
                        valid_composites += 1
                    else:
                        time_series_data.append({
                            'time_step': comp['time_step'],
                            'start_date': comp['start_date'],
                            'end_date': comp['end_date'],
                            'Red': red, 'NIR': nir,
                            'SWIR1': swir1, 'SWIR2': swir2,
                            'image_count': image_count,
                            'mean_cloud_cover': mean_cloud,
                            'quality_flag': 'missing_bands'
                        })
                        failed_composites += 1

                except Exception as e:
                    print(f"  Error in time step {comp['time_step']}: {e}")
                    time_series_data.append({
                        'time_step': comp['time_step'],
                        'start_date': comp['start_date'],
                        'end_date': comp['end_date'],
                        'Red': None, 'NIR': None,
                        'SWIR1': None, 'SWIR2': None,
                        'image_count': 0,
                        'mean_cloud_cover': None,
                        'quality_flag': 'error'
                    })
                    failed_composites += 1

            total_composites = len(time_series_data)
            valid_ratio = valid_composites / total_composites if total_composites > 0 else 0

            print(f"  Valid composites: {valid_composites}/{total_composites} "
                  f"({valid_ratio * 100:.1f}%)")

            return {
                'state': state,
                'district': district,
                'year': year,
                'sensors_used': sensors,
                'time_series': time_series_data,
                'quality_metrics': {
                    'total_composites': total_composites,
                    'valid_composites': valid_composites,
                    'failed_composites': failed_composites,
                    'valid_ratio': valid_ratio,
                    'quality_passed': True  # Accept all (thresholds disabled)
                }
            }

        except Exception as e:
            print(f"   Error processing {district} (interactive): {e}")
            return None

    # ─── Batch Export Helpers ─────────────────────────────────────────────

    def _get_drive_service(self):
        """Build and cache a Google Drive v3 API client."""
        if self._drive_service is not None:
            return self._drive_service
        if not _HAS_DRIVE_API:
            raise ImportError(
                "google-api-python-client is required for batch export. "
                "Install with: pip install google-api-python-client"
            )
        credentials = ee.data.get_persistent_credentials()
        self._drive_service = build('drive', 'v3', credentials=credentials)
        return self._drive_service

    def _build_exact_export_feature(
        self, comp: Dict, geometry: ee.Geometry, scale: int
    ) -> ee.Feature:
        """
        Build one ee.Feature per composite for batch table export.
        All numeric values are formatted as strings server-side to preserve
        exact float representation across the export round-trip.
        """
        fmt = self.EXACT_EXPORT_FLOAT_FORMAT

        image_count = comp['image_count']
        mean_cloud = comp['mean_cloud_cover']
        composite = comp['composite']

        # Reduce region exactly like the interactive path
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e13,
            bestEffort=False
        )

        def maybe_format(value):
            """Format a value as string, guarding against None."""
            return ee.Algorithms.If(
                ee.Algorithms.IsEqual(value, None),
                None,
                ee.Number(value).format(fmt)
            )

        # When image_count == 0, bands and cloud should be None
        has_images = image_count.gt(0)

        red_val = ee.Algorithms.If(has_images, maybe_format(stats.get('Red')), None)
        nir_val = ee.Algorithms.If(has_images, maybe_format(stats.get('NIR')), None)
        swir1_val = ee.Algorithms.If(has_images, maybe_format(stats.get('SWIR1')), None)
        swir2_val = ee.Algorithms.If(has_images, maybe_format(stats.get('SWIR2')), None)
        cloud_val = ee.Algorithms.If(has_images, maybe_format(mean_cloud), None)

        props = {
            'time_step': ee.Number(comp['time_step']).format('%.0f'),
            'start_date': comp['start_date'],
            'end_date': comp['end_date'],
            'Red': red_val,
            'NIR': nir_val,
            'SWIR1': swir1_val,
            'SWIR2': swir2_val,
            'image_count': image_count.format('%.0f'),
            'mean_cloud_cover': cloud_val,
        }

        return ee.Feature(None, props)

    def _wait_for_export_task(self, task, poll_seconds: int = 5) -> Dict:
        """Poll an EE export task until it completes."""
        while True:
            status = task.status()
            state = status.get('state', '')
            if state in ('COMPLETED', 'FAILED', 'CANCELLED'):
                return status
            time.sleep(poll_seconds)

    def _download_drive_file_bytes(self, filename: str) -> bytes:
        """Download a file from Drive by exact name, then delete it."""
        service = self._get_drive_service()

        # Find the file
        results = service.files().list(
            q=f"name='{filename}'",
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        files = results.get('files', [])
        if not files:
            raise FileNotFoundError(f"File '{filename}' not found on Drive")

        file_id = files[0]['id']

        # Download
        request = service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        # Delete from Drive
        try:
            service.files().delete(fileId=file_id).execute()
        except Exception:
            pass  # best-effort cleanup

        return buffer.getvalue()

    @staticmethod
    def _normalize_export_float(text: str) -> str:
        """Normalize an EE-exported float string to Python's repr."""
        if not text:
            return ''
        return str(float(text))

    @staticmethod
    def _normalize_export_int(text: str) -> str:
        """Normalize an EE-exported int string."""
        if not text:
            return '0'
        return str(int(float(text)))

    def _rewrite_exported_csv_exactly(
        self, export_bytes: bytes, state: str, district: str, year: int
    ) -> pd.DataFrame:
        """
        Rewrite EE-exported CSV bytes into the exact schema and numeric
        formatting produced by the interactive extraction path.
        """
        df = pd.read_csv(io.BytesIO(export_bytes), dtype=str).fillna('')

        # Normalize floats
        for col in ['Red', 'NIR', 'SWIR1', 'SWIR2', 'mean_cloud_cover']:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_export_float)

        # Normalize integer
        if 'image_count' in df.columns:
            df['image_count'] = df['image_count'].apply(self._normalize_export_int)

        # Sort key
        df['time_step_num'] = df['time_step'].apply(lambda v: int(float(v)))
        df['time_step'] = df['time_step_num'].astype(str)

        # Add metadata
        df['state'] = state
        df['district'] = district
        df['year'] = str(year)

        # Recompute quality_flag
        band_cols = ['Red', 'NIR', 'SWIR1', 'SWIR2']

        def compute_flag(row):
            if row['image_count'] == '0':
                return 'insufficient_data'
            elif all(row[b] != '' for b in band_cols):
                return 'valid'
            else:
                return 'missing_bands'

        df['quality_flag'] = df.apply(compute_flag, axis=1)

        # Sort and select columns
        df = df.sort_values('time_step_num').reset_index(drop=True)

        final_cols = [
            'state', 'district', 'year', 'time_step',
            'start_date', 'end_date',
            'Red', 'NIR', 'SWIR1', 'SWIR2',
            'image_count', 'mean_cloud_cover', 'quality_flag'
        ]
        return df[final_cols]

    # ─── Public Batch Export Method ──────────────────────────────────────

    def export_district_time_series_to_csv(
        self,
        state: str,
        district: str,
        year: int,
        output_dir: str,
        filename: str,
        scale: int = 30
    ) -> Optional[str]:
        """
        Export a district-year time series via EE batch table export to Drive,
        then download, normalize, and save locally as a raw CSV.

        Falls back to the interactive extraction path on any failure.
        """
        sensors = self.get_sensors_for_year(year)
        print(f"\nExtracting data for {district}, {state} - Year {year} "
              f"(sensors: {', '.join(sensors)}) [batch export]...")

        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Build geometry and composites (same as interactive path)
            geometry = self.get_district_geometry(state, district)
            composites = self.create_time_series_composites(geometry, year)

            # Build features for batch export
            features = []
            for comp in composites:
                feat = self._build_exact_export_feature(comp, geometry, scale)
                features.append(feat)

            feature_collection = ee.FeatureCollection(features)

            # Launch export
            export_id = f"landsat_{state}_{district}_{year}_{uuid.uuid4().hex[:8]}"
            task = ee.batch.Export.table.toDrive(
                collection=feature_collection,
                description=export_id,
                folder=self.BATCH_EXPORT_DRIVE_FOLDER,
                fileNamePrefix=export_id,
                fileFormat='CSV',
                selectors=[
                    'time_step', 'start_date', 'end_date',
                    'Red', 'NIR', 'SWIR1', 'SWIR2',
                    'image_count', 'mean_cloud_cover'
                ]
            )
            task.start()
            print(f"  ▶ Export task started: {export_id}")

            # Wait for completion
            status = self._wait_for_export_task(task)
            if status.get('state') != 'COMPLETED':
                raise RuntimeError(
                    f"Export task {status.get('state')}: "
                    f"{status.get('error_message', 'unknown error')}"
                )
            print(f"  ✓ Export task completed")

            # Download from Drive
            drive_filename = f"{export_id}.csv"
            export_bytes = self._download_drive_file_bytes(drive_filename)
            print(f"  ✓ Downloaded {len(export_bytes)} bytes from Drive")

            # Rewrite to exact schema
            df = self._rewrite_exported_csv_exactly(export_bytes, state, district, year)

            # Count valid composites
            valid_count = (df['quality_flag'] == 'valid').sum()
            total_count = len(df)
            valid_ratio = valid_count / total_count if total_count > 0 else 0
            print(f"  Valid composites: {valid_count}/{total_count} "
                  f"({valid_ratio * 100:.1f}%)")

            # Save
            df.to_csv(output_path, index=False)
            print(f"  Data saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"  ⚠ Batch export failed: {e}")
            print(f"  → Falling back to interactive extraction...")

            # Fallback to interactive path
            try:
                result = self.extract_district_time_series(
                    state=state, district=district, year=year, scale=scale
                )
                if result is not None:
                    return self.save_to_csv(
                        result, output_dir=output_dir, filename=filename
                    )
            except Exception as fallback_err:
                print(f"  ✗ Fallback also failed: {fallback_err}")

            return None

    # ─── CSV Export ──────────────────────────────────────────────────────

    def save_to_csv(
        self,
        result: Dict,
        output_dir: str = None,
        filename: str = None
    ) -> Optional[str]:
        """Save extracted time series data to CSV file."""
        if result is None:
            print("Warning: No data to save")
            return None

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(result['time_series'])
        df['state'] = result['state']
        df['district'] = result['district']
        df['year'] = result['year']

        base_cols = ['state', 'district', 'year', 'time_step',
                     'start_date', 'end_date',
                     'Red', 'NIR', 'SWIR1', 'SWIR2']
        quality_cols = ['image_count', 'mean_cloud_cover', 'quality_flag']
        available_cols = [c for c in base_cols + quality_cols if c in df.columns]
        df = df[available_cols]

        if filename is None:
            filename = f"landsat_{result['state']}_{result['district']}_{result['year']}.csv"

        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Data saved to: {output_path}")
        return output_path

    def save_multiple_to_csv(
        self,
        results: List[Dict],
        output_path: str = "all_districts_landsat.csv"
    ) -> Optional[str]:
        """Save multiple district results to a single CSV file."""
        all_dfs = []

        for result in results:
            if result is None:
                continue
            df = pd.DataFrame(result['time_series'])
            df['state'] = result['state']
            df['district'] = result['district']
            df['year'] = result['year']
            all_dfs.append(df)

        if not all_dfs:
            print("Warning: No data to save")
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True)

        base_cols = ['state', 'district', 'year', 'time_step',
                     'start_date', 'end_date',
                     'Red', 'NIR', 'SWIR1', 'SWIR2']
        quality_cols = ['image_count', 'mean_cloud_cover', 'quality_flag']
        available_cols = [c for c in base_cols + quality_cols if c in combined_df.columns]
        combined_df = combined_df[available_cols]

        combined_df.to_csv(output_path, index=False)
        print(f"Combined data saved to: {output_path}")
        print(f"Total records: {len(combined_df)}")
        return output_path


# ─── Standalone helper ───────────────────────────────────────────────────────

def process_all_districts(
    extractor: LandsatExtractor,
    years: List[int] = None,
    output_dir: str = None
) -> List[Dict]:
    """Process all districts and extract time series data."""
    if years is None:
        years = PROCESSING_YEARS
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for state, districts in ALL_DISTRICTS.items():
        print(f"\n{'=' * 50}")
        print(f"Processing {state} ({len(districts)} districts)")
        print('=' * 50)

        for district in districts:
            for year in years:
                result = extractor.extract_district_time_series(
                    state=state, district=district, year=year
                )
                if result:
                    all_results.append(result)
                    output_file = os.path.join(
                        output_dir, f"{state}_{district}_{year}.json"
                    )
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)

    combined_output = os.path.join(output_dir, "all_districts_data.json")
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nData saved to: {combined_output}")
    return all_results


def main():
    """Main function — quick test run."""
    print("=" * 60)
    print("Landsat Wheat Yield Prediction Pipeline")
    print("Target: Haryana, Punjab, Uttar Pradesh  |  2010–2026")
    print("=" * 60)

    try:
        extractor = LandsatExtractor(initialize_ee=True)
    except Exception as e:
        print(f"\nFailed to initialize Earth Engine: {e}")
        print("\nPlease ensure you have:")
        print("1. Installed earthengine-api: pip install earthengine-api")
        print("2. Authenticated: earthengine authenticate")
        print("3. Have an approved GEE account")
        return

    # Test with one district
    print("\n--- Testing with Karnal, Haryana (2015) ---")
    test_result = extractor.extract_district_time_series(
        state="Haryana", district="Karnal", year=2015
    )

    if test_result:
        print(f"\nExtracted {len(test_result['time_series'])} time steps")
        print(f"Sensors used: {test_result['sensors_used']}")
        print("\nSample data (first 5 time steps):")
        for ts in test_result['time_series'][:5]:
            print(f"  {ts}")
        csv_path = extractor.save_to_csv(test_result)
        print(f"\n✓ CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
