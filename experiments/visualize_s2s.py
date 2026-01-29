#!/usr/bin/env python3
"""
S2S ECMWF GRIB Data Visualization Script

This script explores and visualizes the downloaded S2S GRIB file.

Data specs from download:
- class: s2 (S2S project)
- origin: ecmf (ECMWF model)
- param: 122 (Momentum flux, north-south) / 228228 (Total precipitation)
- step: 1080/1086/1092/1098 hours (45-46 days lead time)
- hdate: 2024-01-21 (hindcast date)
- type: cf (control forecast)
- stream: enfh (ensemble forecast hindcast)
"""

import sys
from pathlib import Path

# Check dependencies
try:
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import cfgrib
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nInstall required packages:")
    print("pip install xarray cfgrib matplotlib numpy eccodes")
    sys.exit(1)


def explore_grib(filepath: str):
    """Explore the structure of the GRIB file."""
    print("=" * 60)
    print("EXPLORING GRIB FILE STRUCTURE")
    print("=" * 60)
    
    # Try to open with different filter combinations
    # GRIB files often have multiple "messages" that need different filters
    
    datasets = []
    
    # Method 1: Open all datasets in the file
    try:
        print("\n📦 Opening GRIB file with cfgrib...")
        ds_list = cfgrib.open_datasets(filepath)
        print(f"   Found {len(ds_list)} dataset(s) in the file\n")
        
        for i, ds in enumerate(ds_list):
            print(f"--- Dataset {i+1} ---")
            print(f"Variables: {list(ds.data_vars)}")
            print(f"Dimensions: {dict(ds.dims)}")
            print(f"Coordinates: {list(ds.coords)}")
            print()
            datasets.append(ds)
            
    except Exception as e:
        print(f"Error with cfgrib.open_datasets: {e}")
        
        # Fallback: try xarray directly
        try:
            ds = xr.open_dataset(filepath, engine='cfgrib')
            datasets.append(ds)
            print(f"Variables: {list(ds.data_vars)}")
        except Exception as e2:
            print(f"Error with xarray: {e2}")
    
    return datasets


def analyze_dataset(ds: xr.Dataset, idx: int):
    """Analyze a single dataset in detail."""
    print("=" * 60)
    print(f"DETAILED ANALYSIS - Dataset {idx}")
    print("=" * 60)
    
    print("\n📊 Full Dataset Info:")
    print(ds)
    
    print("\n📍 Coordinate Details:")
    for coord_name, coord in ds.coords.items():
        print(f"\n  {coord_name}:")
        print(f"    Shape: {coord.shape}")
        print(f"    Dtype: {coord.dtype}")
        if coord.size < 20:
            print(f"    Values: {coord.values}")
        else:
            print(f"    Range: {coord.values.min()} to {coord.values.max()}")
            print(f"    First 5: {coord.values[:5]}")
    
    print("\n🌡️ Variable Details:")
    for var_name, var in ds.data_vars.items():
        print(f"\n  {var_name}:")
        print(f"    Dimensions: {var.dims}")
        print(f"    Shape: {var.shape}")
        print(f"    Dtype: {var.dtype}")
        
        # Get attributes (units, long_name, etc.)
        if var.attrs:
            print(f"    Attributes:")
            for k, v in var.attrs.items():
                print(f"      {k}: {v}")
        
        # Statistics
        data = var.values
        if not np.isnan(data).all():
            print(f"    Statistics:")
            print(f"      Min: {np.nanmin(data):.4f}")
            print(f"      Max: {np.nanmax(data):.4f}")
            print(f"      Mean: {np.nanmean(data):.4f}")
            print(f"      Std: {np.nanstd(data):.4f}")
            print(f"      NaN count: {np.isnan(data).sum()}")


def plot_data(datasets: list, output_dir: Path):
    """Create visualizations of the data."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    output_dir.mkdir(exist_ok=True)
    
    for ds_idx, ds in enumerate(datasets):
        for var_name in ds.data_vars:
            var = ds[var_name]
            
            # Get the data
            data = var.values
            
            # Handle different dimensions
            if 'latitude' in var.dims and 'longitude' in var.dims:
                lat = ds.coords.get('latitude', ds.coords.get('lat'))
                lon = ds.coords.get('longitude', ds.coords.get('lon'))
                
                # If there are multiple steps, plot each
                if 'step' in var.dims:
                    steps = ds.coords['step'].values
                    
                    fig, axes = plt.subplots(1, len(steps), figsize=(5*len(steps), 4))
                    if len(steps) == 1:
                        axes = [axes]
                    
                    for i, step in enumerate(steps):
                        step_data = var.sel(step=step).values
                        
                        # Handle potential extra dimensions
                        if step_data.ndim > 2:
                            step_data = step_data.squeeze()
                        
                        im = axes[i].pcolormesh(
                            lon.values, lat.values, step_data,
                            cmap='viridis', shading='auto'
                        )
                        
                        # Convert step to hours/days
                        if isinstance(step, np.timedelta64):
                            step_hours = step / np.timedelta64(1, 'h')
                        else:
                            step_hours = step
                        
                        axes[i].set_title(f'Step: {step_hours}h ({step_hours/24:.1f}d)')
                        axes[i].set_xlabel('Longitude')
                        axes[i].set_ylabel('Latitude')
                        plt.colorbar(im, ax=axes[i], shrink=0.8)
                    
                    plt.suptitle(f'{var_name} - {var.attrs.get("long_name", "")}')
                    plt.tight_layout()
                    
                else:
                    # Single timestep
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    plot_data = data.squeeze()
                    im = ax.pcolormesh(
                        lon.values, lat.values, plot_data,
                        cmap='viridis', shading='auto'
                    )
                    
                    ax.set_title(f'{var_name} - {var.attrs.get("long_name", "")}')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    plt.colorbar(im, ax=ax, label=var.attrs.get('units', ''))
                
                # Save figure
                fig_path = output_dir / f'ds{ds_idx}_{var_name}.png'
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                print(f"   Saved: {fig_path}")
                plt.close()
            
            else:
                print(f"   Skipping {var_name}: no lat/lon dimensions found")
                print(f"     Dimensions: {var.dims}")


def print_geographic_info(datasets: list):
    """Print geographic coverage info."""
    print("\n" + "=" * 60)
    print("GEOGRAPHIC COVERAGE")
    print("=" * 60)
    
    for ds_idx, ds in enumerate(datasets):
        lat = ds.coords.get('latitude', ds.coords.get('lat'))
        lon = ds.coords.get('longitude', ds.coords.get('lon'))
        
        if lat is not None and lon is not None:
            print(f"\nDataset {ds_idx}:")
            print(f"  Latitude range:  {lat.values.min():.2f}° to {lat.values.max():.2f}°")
            print(f"  Longitude range: {lon.values.min():.2f}° to {lon.values.max():.2f}°")
            
            # Calculate grid spacing
            if len(lat) > 1:
                lat_spacing = abs(lat.values[1] - lat.values[0])
                print(f"  Latitude spacing: {lat_spacing:.4f}° (~{lat_spacing * 111:.1f} km)")
            if len(lon) > 1:
                lon_spacing = abs(lon.values[1] - lon.values[0])
                print(f"  Longitude spacing: {lon_spacing:.4f}° (~{lon_spacing * 111:.1f} km)")
            
            print(f"  Grid size: {len(lat)} × {len(lon)} = {len(lat) * len(lon)} points")


def main():
    # Find the GRIB file
    # Look in common locations
    possible_paths = [
        Path("output"),
        Path("output.grib"),
        Path("output.grib2"),
        Path("../output"),
        Path("../output.grib"),
        Path("/Users/aaryan/Downloads/ugp/output"),
        Path("/Users/aaryan/Downloads/output"),
    ]
    
    grib_file = None
    for p in possible_paths:
        if p.exists():
            grib_file = p
            break
    
    if grib_file is None:
        print("❌ Could not find GRIB file!")
        print("\nPlease provide the path to your GRIB file:")
        print("  python visualize_s2s.py /path/to/your/output.grib")
        print("\nOr copy the file to this directory as 'output'")
        
        if len(sys.argv) > 1:
            grib_file = Path(sys.argv[1])
            if not grib_file.exists():
                print(f"\n❌ File not found: {grib_file}")
                sys.exit(1)
        else:
            sys.exit(1)
    
    print(f"\n📁 Using GRIB file: {grib_file}")
    print(f"   File size: {grib_file.stat().st_size / 1024:.1f} KB")
    
    # Explore and analyze
    datasets = explore_grib(str(grib_file))
    
    if not datasets:
        print("\n❌ Could not read any datasets from the GRIB file")
        sys.exit(1)
    
    for i, ds in enumerate(datasets):
        analyze_dataset(ds, i + 1)
    
    print_geographic_info(datasets)
    
    # Create visualizations
    output_dir = Path(__file__).parent / "plots"
    plot_data(datasets, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
