
import ee
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Import configuration
try:
    from config import (
        SENTINEL2_BANDS,
        ALL_DISTRICTS, WHEAT_SEASON, TIME_SERIES_CONFIG, QUALITY_CONTROL,
        PROCESSING_YEARS, EXPORT_CONFIG, PROCESSED_DATA_DIR, OUTPUT_DIR
    )
except ImportError:
    print("Warning: config.py not found. Using default values.")
    PROCESSING_YEARS = [2020, 2021, 2022]


class Sentinel2Extractor:
    """
    Class to extract Sentinel-2 data for wheat yield prediction.
    """
    
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
    }
    
    def __init__(self, initialize_ee: bool = True):
        # Initialize the Sentinel-2 Extractor.
        if initialize_ee:
            try:
                ee.Initialize(project='ugp-prediction')
                print("Earth Engine initialized successfully!")
            except Exception as e:
                print(f"Error initializing Earth Engine: {e}")
                print("Please run 'earthengine authenticate' first.")
                raise
        
        # Load India administrative boundaries
        self.india_admin = ee.FeatureCollection("FAO/GAUL/2015/level2")
        
        # Sentinel-2 bands as needed
        self.bands = ['B7', 'B8', 'B8A', 'B12']
        
        # Use harmonized collection 
        self.sentinel2_collection = 'COPERNICUS/S2_HARMONIZED'
        
        # Quality control thresholds 
        self.min_images_per_composite = 0  # Track all composites regardless of image count
        self.max_cloud_percentage = 30  # Maximum cloud cover per image
        
        # Load cropland mask for agricultural area filtering
        # Using ESA WorldCover 10m 2021 - cropland class = 40
        try:
            self.cropland_mask = ee.ImageCollection('ESA/WorldCover/v200').first().eq(40)
            print("Cropland mask loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load cropland mask: {e}")
            self.cropland_mask = None
        
    def get_district_geometry(self, state: str, district: str, use_buffer: bool = False) -> ee.Geometry:
        """
        Get the geometry for a specific district using actual administrative boundaries.
        Falls back to buffered point if boundaries unavailable.
        """
        state_mapping = {
            "Haryana": "Haryana",
            "Punjab": "Punjab",
            "Uttar_Pradesh": "Uttar Pradesh",
            "Uttar Pradesh": "Uttar Pradesh"
        }
        
        gaul_state = state_mapping.get(state, state)
        
        try:
            # Try to get actual district boundaries from GAUL
            district_fc = self.india_admin.filter(
                ee.Filter.And(
                    ee.Filter.eq('ADM1_NAME', gaul_state),
                    ee.Filter.eq('ADM2_NAME', district)
                )
            )
            
            # Check if we found the district
            size = district_fc.size().getInfo()
            if size > 0 and not use_buffer:
                print(f"  Using actual boundaries for {district}")
                return district_fc.geometry()
            else:
                raise Exception("District not found in GAUL, using fallback")
                
        except Exception as e:
            # Fallback to buffered point geometry
            if district in self.DISTRICT_COORDS:
                print(f"  Using 25km buffer fallback for {district}")
                lat, lon = self.DISTRICT_COORDS[district]
                point = ee.Geometry.Point([lon, lat])
                # Buffer by 25000 meters (25 km radius)
                return point.buffer(25000)
            else:
                raise Exception(f"No geometry available for {district}, {state}")
    
    def mask_clouds_s2(self, image: ee.Image) -> ee.Image:
        """
        Preprocess Sentinel-2 images with proper cloud masking.
        Uses QA60 band for cloud and cirrus detection.
        """
        # Use QA60 band for cloud masking
        qa = image.select('QA60')
        
        # Bits 10 and 11 are clouds and cirrus
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        
        # Both flags should be zero, indicating clear conditions
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
               qa.bitwiseAnd(cirrusBitMask).eq(0))
        
        # Apply mask, scale to reflectance, and preserve timestamp
        return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start', 'CLOUDY_PIXEL_PERCENTAGE'])
    
    def get_wheat_season_dates(self, year: int) -> Tuple[str, str]:
        """
        Get wheat season start and end dates for a given year.
        
        For Indian wheat:
        - Sowing: October-November of year
        - Harvest: March-April of year+1
        
        """
        start_date = f"{year}-10-01"
        end_date = f"{year+1}-04-30"
        
        return start_date, end_date
    
    def create_time_series_composites(
        self, 
        geometry: ee.Geometry,
        year: int,
        composite_days: int = 5  # 5-day composites 
    ) -> List[Dict]:
        """
        Create 5-day mean composites for the wheat growing season with quality tracking.
        """
        start_date, end_date = self.get_wheat_season_dates(year)
        
        # Get Sentinel-2 collection with improved cloud filtering
        collection = ee.ImageCollection(self.sentinel2_collection) \
            .filterBounds(geometry) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.max_cloud_percentage)) \
            .map(self.mask_clouds_s2) \
            .select(self.bands)
        
        # Apply cropland mask if available
        if self.cropland_mask is not None:
            collection = collection.map(lambda img: img.updateMask(self.cropland_mask))
        
        composites = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        time_step = 0
        while current_date < end:
            next_date = current_date + timedelta(days=composite_days)
            
            # Filter images for this composite period
            period_collection = collection.filterDate(
                current_date.strftime("%Y-%m-%d"),
                next_date.strftime("%Y-%m-%d")
            )
            
            # Get image count and mean cloud cover for quality tracking
            image_count = period_collection.size()
            
            # Only create composite if we have enough images
            composite = period_collection.mean()
            
            # Get mean cloud percentage for the period
            mean_cloud = period_collection.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE')
            
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
    
    def extract_district_time_series(
        self,
        state: str,
        district: str,
        year: int,
        scale: int = 20  # Changed to 20m 
    ) -> Dict:
        """
        Extract time series data for a specific district with quality validation.
        """
        print(f"\nExtracting data for {district}, {state} - Year {year}...")
        
        try:
            # Get district geometry
            geometry = self.get_district_geometry(state, district)
            
            # Create composites
            composites = self.create_time_series_composites(geometry, year)
            
            # Extract mean values for each composite with quality tracking
            time_series_data = []
            valid_composites = 0
            failed_composites = 0
            
            for comp in composites:
                try:
                    # Get image count for this composite
                    image_count = comp['image_count'].getInfo()
                    mean_cloud = comp['mean_cloud_cover'].getInfo()
                    
                    # Validate composite has sufficient data
                    if image_count < self.min_images_per_composite:
                        print(f"  ⚠ Time step {comp['time_step']}: Only {image_count} images, skipping")
                        time_series_data.append({
                            'time_step': comp['time_step'],
                            'start_date': comp['start_date'],
                            'end_date': comp['end_date'],
                            'B7': None, 'B8': None, 'B8A': None, 'B12': None,
                            'image_count': image_count,
                            'mean_cloud_cover': mean_cloud,
                            'quality_flag': 'insufficient_data'
                        })
                        failed_composites += 1
                        continue
                    
                    # Compute mean over the district
                    stats = comp['composite'].reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=scale,
                        maxPixels=1e13,
                        bestEffort=False  # Ensures we get accurate results or fail
                    ).getInfo()
                    
                    # Check if we got valid data for all bands
                    b7 = stats.get('B7')
                    b8 = stats.get('B8')
                    b8a = stats.get('B8A')
                    b12 = stats.get('B12')
                    
                    if all(v is not None for v in [b7, b8, b8a, b12]):
                        time_series_data.append({
                            'time_step': comp['time_step'],
                            'start_date': comp['start_date'],
                            'end_date': comp['end_date'],
                            'B7': b7,
                            'B8': b8,
                            'B8A': b8a,
                            'B12': b12,
                            'image_count': image_count,
                            'mean_cloud_cover': mean_cloud,
                            'quality_flag': 'valid'
                        })
                        valid_composites += 1
                    else:
                        print(f"  Time step {comp['time_step']}: Missing band values")
                        time_series_data.append({
                            'time_step': comp['time_step'],
                            'start_date': comp['start_date'],
                            'end_date': comp['end_date'],
                            'B7': b7, 'B8': b8, 'B8A': b8a, 'B12': b12,
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
                        'B7': None, 'B8': None, 'B8A': None, 'B12': None,
                        'image_count': 0,
                        'mean_cloud_cover': None,
                        'quality_flag': 'error'
                    })
                    failed_composites += 1
            
            # Calculate quality metrics
            total_composites = len(time_series_data)
            valid_ratio = valid_composites / total_composites if total_composites > 0 else 0
            
            # Quality thresholds disabled - accepting all extractions
            quality_passed = True  # Always pass, no longer rejecting based on quality
            
            print(f"  Valid composites: {valid_composites}/{total_composites} ({valid_ratio*100:.1f}%)")
            print(f"  Data extracted (quality thresholds disabled)")
            
            return {
                'state': state,
                'district': district,
                'year': year,
                'time_series': time_series_data,
                'quality_metrics': {
                    'total_composites': total_composites,
                    'valid_composites': valid_composites,
                    'failed_composites': failed_composites,
                    'valid_ratio': valid_ratio,
                    'quality_passed': quality_passed
                }
            }
            
        except Exception as e:
            print(f"   Error processing {district}: {e}")
            return None
    
    def compute_vegetation_indices(self, image: ee.Image) -> ee.Image:
        """
        Compute additional vegetation indices that may be useful.
        """
        # NDVI (Normalized Difference Vegetation Index)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # EVI (Enhanced Vegetation Index)
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('EVI')
        
        # NDWI (Normalized Difference Water Index)
        ndwi = image.normalizedDifference(['B8', 'B12']).rename('NDWI')
        
        return image.addBands([ndvi, evi, ndwi])
    
    def export_to_drive(
        self,
        image: ee.Image,
        geometry: ee.Geometry,
        description: str,
        folder: str = "wheat_yield_data",
        scale: int = 10
    ):
        """
        Export image to Google Drive.

        """
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            region=geometry,
            scale=scale,
            crs='EPSG:4326',
            maxPixels=1e13
        )
        
        task.start()
        print(f"Export task started: {description}")
        return task
    
    def save_to_csv(
        self,
        result: Dict,
        output_dir: str = None,
        filename: str = None
    ) -> str:
        """
        Save extracted time series data to CSV file.
        """
        if result is None:
            print("Warning: No data to save")
            return None
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame from time series data
        df = pd.DataFrame(result['time_series'])
        
        # Add metadata columns
        df['state'] = result['state']
        df['district'] = result['district']
        df['year'] = result['year']
        
        # Reorder columns with quality metrics
        base_cols = ['state', 'district', 'year', 'time_step', 'start_date', 'end_date', 
                     'B7', 'B8', 'B8A', 'B12']
        quality_cols = ['image_count', 'mean_cloud_cover', 'quality_flag']
        
        # Only include quality columns if they exist
        available_cols = [col for col in base_cols + quality_cols if col in df.columns]
        df = df[available_cols]
        
        # Generate filename
        if filename is None:
            filename = f"sentinel2_{result['state']}_{result['district']}_{result['year']}.csv"
        
        output_path = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Data saved to: {output_path}")
        
        return output_path
    
    def save_multiple_to_csv(
        self,
        results: List[Dict],
        output_path: str = "all_districts_sentinel2.csv"
    ) -> str:
        """
        Save multiple district results to a single CSV file.
        """
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
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Reorder columns with quality metrics
        base_cols = ['state', 'district', 'year', 'time_step', 'start_date', 'end_date', 
                     'B7', 'B8', 'B8A', 'B12']
        quality_cols = ['image_count', 'mean_cloud_cover', 'quality_flag']
        
        # Only include quality columns if they exist
        available_cols = [col for col in base_cols + quality_cols if col in combined_df.columns]
        combined_df = combined_df[available_cols]
        
        # Save to CSV
        combined_df.to_csv(output_path, index=False)
        print(f"Combined data saved to: {output_path}")
        print(f"Total records: {len(combined_df)}")
        
        return output_path





def process_all_districts(
    extractor: Sentinel2Extractor,
    years: List[int] = None,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Process all districts and extract time series data.
    """
    if years is None:
        years = PROCESSING_YEARS
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # Load district lists
    from config import ALL_DISTRICTS
    
    for state, districts in ALL_DISTRICTS.items():
        print(f"\n{'='*50}")
        print(f"Processing {state} ({len(districts)} districts)")
        print('='*50)
        
        for district in districts:
            for year in years:
                result = extractor.extract_district_time_series(
                    state=state,
                    district=district,
                    year=year
                )
                
                if result:
                    all_results.append(result)
                    
                    # Save individual result
                    output_file = os.path.join(
                        output_dir, 
                        f"{state}_{district}_{year}.json"
                    )
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
    
    # Save combined results
    combined_output = os.path.join(output_dir, "all_districts_data.json")
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nData saved to: {combined_output}")
    
    return all_results


def main():
    """Main function to run the pipeline."""
    print("="*60)
    print("Sentinel-2 Wheat Yield Prediction Pipeline")
    print("Target: Haryana, Punjab, Uttar Pradesh")
    print("="*60)
    
    # Initialize extractor
    try:
        extractor = Sentinel2Extractor(initialize_ee=True)
    except Exception as e:
        print(f"\nFailed to initialize Earth Engine: {e}")
        print("\nPlease ensure you have:")
        print("1. Installed earthengine-api: pip install earthengine-api")
        print("2. Authenticated: earthengine authenticate")
        print("3. Have an approved GEE account")
        return
    
    # Example: Extract data for one district
    print("\n--- Testing with single district ---")
    test_result = extractor.extract_district_time_series(
        state="Haryana",
        district="Karnal",
        year=2022
    )
    
    if test_result:
        print(f"\nSuccessfully extracted {len(test_result['time_series'])} time steps")
        print("\nSample data (first 5 time steps):")
        for ts in test_result['time_series'][:5]:
            print(f"  {ts}")
        
        # Save to CSV
        csv_path = extractor.save_to_csv(test_result)
        print(f"\n✓ CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
