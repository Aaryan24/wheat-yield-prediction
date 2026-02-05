import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

# Import local modules
from gee_extractor import Sentinel2Extractor
from fill_missing_sma import fill_missing_with_sma
from config import ALL_DISTRICTS


class DistrictProcessor:
    """Orchestrates multi-year data processing for all districts."""
    
    def __init__(
        self,
        years: List[int] = None,
        output_base_dir: str = "output",
        rate_limit_delay: int = 10  # seconds between extractions
    ):
        """
        Initialize the district processor.
        
        Args:
            years: List of years to process (default: 2017-2024)
            output_base_dir: Base directory for all outputs
            rate_limit_delay: Delay in seconds between extractions to avoid rate limits
        """
        self.years = years or list(range(2017, 2025))
        self.output_base_dir = output_base_dir
        self.rate_limit_delay = rate_limit_delay
        
        # Create output directories
        self.raw_dir = os.path.join(output_base_dir, "raw")
        self.filled_dir = os.path.join(output_base_dir, "filled")
        self.merged_dir = os.path.join(output_base_dir, "merged")
        self.logs_dir = os.path.join(output_base_dir, "logs")
        
        for directory in [self.raw_dir, self.filled_dir, self.merged_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize extractor
        self.extractor = Sentinel2Extractor(initialize_ee=True)
        
        # Tracking
        self.processing_log = []
        self.errors = []
        
    def extract_and_fill_year(
        self,
        state: str,
        district: str,
        year: int
    ) -> Optional[pd.DataFrame]:
        """
        Extract data for one year and apply SMA filling.
        
        Args:
            state: State name
            district: District name
            year: Year to process
            
        Returns:
            Filled DataFrame, or None if extraction failed
        """
        # Generate filenames
        raw_filename = f"{state}_{district}_{year}_raw.csv"
        filled_filename = f"{state}_{district}_{year}_filled.csv"
        
        raw_path = os.path.join(self.raw_dir, raw_filename)
        filled_path = os.path.join(self.filled_dir, filled_filename)
        
        # Check if already processed (for resume capability)
        if os.path.exists(filled_path):
            print(f"  ↻ Year {year} already processed, loading from file")
            return pd.read_csv(filled_path)
        
        try:
            print(f"  → Extracting year {year}...")
            
            # Extract Sentinel-2 data
            result = self.extractor.extract_district_time_series(
                state=state,
                district=district,
                year=year,
                scale=20
            )
            
            if result is None:
                self.errors.append({
                    'state': state,
                    'district': district,
                    'year': year,
                    'error': 'Extraction returned None'
                })
                return None
            
            # Save raw data
            self.extractor.save_to_csv(result, output_dir=self.raw_dir, filename=raw_filename)
            
            # Load and fill missing values
            df = pd.read_csv(raw_path)
            band_columns = ['B7', 'B8', 'B8A', 'B12']
            
            missing_before = df[band_columns].isna().sum().sum()
            
            if missing_before > 0:
                print(f"  → Filling {missing_before} missing values with SMA...")
                df_filled = fill_missing_with_sma(df, band_columns, window=3)
            else:
                print(f"  ✓ No missing values to fill")
                df_filled = df
            
            missing_after = df_filled[band_columns].isna().sum().sum()
            
            # Save filled data
            df_filled.to_csv(filled_path, index=False)
            print(f"  ✓ Year {year} processed ({missing_before - missing_after} values filled)")
            
            # Rate limiting delay
            print(f"  ⏸ Waiting {self.rate_limit_delay}s for rate limiting...")
            time.sleep(self.rate_limit_delay)
            
            return df_filled
            
        except Exception as e:
            error_msg = f"Error processing {state}/{district}/{year}: {str(e)}"
            print(f"  ✗ {error_msg}")
            self.errors.append({
                'state': state,
                'district': district,
                'year': year,
                'error': str(e)
            })
            return None
    
    def merge_years(
        self,
        state: str,
        district: str,
        year_dataframes: Dict[int, pd.DataFrame]
    ) -> Optional[str]:
        """
        Merge all years into a single district CSV.
        
        Args:
            state: State name
            district: District name
            year_dataframes: Dictionary mapping year -> DataFrame
            
        Returns:
            Path to merged CSV, or None if merge failed
        """
        if not year_dataframes:
            print(f"  ✗ No data to merge for {state}/{district}")
            return None
        
        try:
            print(f"  → Merging {len(year_dataframes)} years...")
            
            # Concatenate all years
            dfs = []
            for year in sorted(year_dataframes.keys()):
                df = year_dataframes[year].copy()
                # Ensure year column exists
                if 'year' not in df.columns:
                    df['year'] = year
                dfs.append(df)
            
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # Sort by year and time_step
            merged_df = merged_df.sort_values(['year', 'time_step']).reset_index(drop=True)
            
            # Save merged data
            merged_filename = f"{state}_{district}_remote_sensing_data.csv"
            merged_path = os.path.join(self.merged_dir, merged_filename)
            merged_df.to_csv(merged_path, index=False)
            
            print(f"  ✓ Merged {len(merged_df)} records → {merged_filename}")
            return merged_path
            
        except Exception as e:
            error_msg = f"Error merging {state}/{district}: {str(e)}"
            print(f"  ✗ {error_msg}")
            self.errors.append({
                'state': state,
                'district': district,
                'year': 'merge',
                'error': str(e)
            })
            return None
    
    def process_district(
        self,
        state: str,
        district: str,
        district_num: int,
        total_districts: int
    ) -> bool:
        """
        Process complete workflow for one district.

        """
        print(f"\n{'='*80}")
        print(f"[{district_num}/{total_districts}] Processing: {state} - {district}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Check if already fully processed
        merged_filename = f"{state}_{district}_remote_sensing_data.csv"
        merged_path = os.path.join(self.merged_dir, merged_filename)
        
        if os.path.exists(merged_path):
            print(f"✓ District already processed, skipping")
            return True
        
        # Extract and fill each year
        year_dataframes = {}
        
        for year in self.years:
            df = self.extract_and_fill_year(state, district, year)
            if df is not None:
                year_dataframes[year] = df
        
        # Merge all years
        if year_dataframes:
            merged_path = self.merge_years(state, district, year_dataframes)
            success = merged_path is not None
        else:
            print(f"  ✗ No successful extractions for {state}/{district}")
            success = False
        
        elapsed = time.time() - start_time
        
        # Log processing
        self.processing_log.append({
            'state': state,
            'district': district,
            'years_processed': len(year_dataframes),
            'success': success,
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"\nDistrict {district_num}/{total_districts} completed in {elapsed/60:.1f} minutes")
        
        return success
    
    def process_all_districts(self):
        """Process all districts from config."""
        print("="*80)
        print("MULTI-YEAR SENTINEL-2 DATA PROCESSING PIPELINE")
        print("="*80)
        print(f"Years: {self.years[0]}-{self.years[-1]}")
        print(f"Rate limit delay: {self.rate_limit_delay}s per extraction")
        print(f"Output directory: {self.output_base_dir}")
        print("="*80)
        
        # Count total districts
        total_districts = sum(len(districts) for districts in ALL_DISTRICTS.values())
        
        print(f"\nTotal districts to process: {total_districts}")
        for state, districts in ALL_DISTRICTS.items():
            print(f"  - {state}: {len(districts)} districts")
        
        input(f"\nPress Enter to start processing {total_districts} districts...")
        
        start_time = time.time()
        district_num = 0
        successful = 0
        failed = 0
        
        # Process each state/district
        for state, districts in ALL_DISTRICTS.items():
            for district in districts:
                district_num += 1
                
                success = self.process_district(
                    state=state,
                    district=district,
                    district_num=district_num,
                    total_districts=total_districts
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
        
        # Save logs and summary
        self.save_summary(total_districts, successful, failed, start_time)
    
    def save_summary(
        self,
        total_districts: int,
        successful: int,
        failed: int,
        start_time: float
    ):
        """Save processing summary and logs."""
        elapsed = time.time() - start_time
        
        # Summary report
        summary = {
            'total_districts': total_districts,
            'successful': successful,
            'failed': failed,
            'years_processed': self.years,
            'total_elapsed_seconds': elapsed,
            'total_elapsed_hours': elapsed / 3600,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(self.output_base_dir, "summary_report.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Processing log
        log_path = os.path.join(self.logs_dir, "processing_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.processing_log, f, indent=2)
        
        # Error log
        if self.errors:
            error_path = os.path.join(self.logs_dir, "errors.json")
            with open(error_path, 'w') as f:
                json.dump(self.errors, f, indent=2)
        
        # Print final summary
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Total districts: {total_districts}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {elapsed/3600:.2f} hours")
        print(f"\nResults saved to: {self.output_base_dir}")
        print(f"  - Merged data: {self.merged_dir}")
        print(f"  - Raw data: {self.raw_dir}")
        print(f"  - Filled data: {self.filled_dir}")
        print(f"  - Logs: {self.logs_dir}")
        
        if self.errors:
            print(f"\n⚠ {len(self.errors)} errors occurred. Check {self.logs_dir}/errors.json")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process Sentinel-2 data for all districts across multiple years"
    )
    
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=list(range(2017, 2025)),
        help='Years to process (default: 2017-2024)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Base output directory (default: output)'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=10,
        help='Delay in seconds between extractions (default: 10)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only Karnal, Haryana for years 2022-2023'
    )
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        print("\n*** TEST MODE ***")
        print("Processing: Karnal, Haryana - Years 2022, 2023\n")
        
        processor = DistrictProcessor(
            years=[2022, 2023],
            output_base_dir=args.output_dir,
            rate_limit_delay=args.rate_limit
        )
        
        processor.process_district(
            state="Haryana",
            district="Karnal",
            district_num=1,
            total_districts=1
        )
        
        processor.save_summary(1, 1, 0, time.time())
        return
    
    # Full processing
    processor = DistrictProcessor(
        years=args.years,
        output_base_dir=args.output_dir,
        rate_limit_delay=args.rate_limit
    )
    
    processor.process_all_districts()


if __name__ == "__main__":
    main()
