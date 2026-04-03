"""
Fill missing values in Landsat remote sensing time-series data
using Simple Moving Average (SMA) or interpolation.

Adapted for Landsat bands: Red, NIR, SWIR1, SWIR2.
"""

import pandas as pd
import numpy as np
import argparse
import os


def fill_missing_with_sma(df: pd.DataFrame, columns: list, window: int = 3) -> pd.DataFrame:
    """
    Fill missing values using Simple Moving Average (SMA).

    For each missing value, calculates the average of surrounding valid values
    within the specified window.
    """
    df_filled = df.copy()

    for col in columns:
        if col not in df_filled.columns:
            continue

        values = df_filled[col].values.astype(float)
        filled_values = values.copy()

        missing_idx = np.where(pd.isna(values))[0]

        for idx in missing_idx:
            start = max(0, idx - window)
            end = min(len(values), idx + window + 1)

            window_values = []
            for i in range(start, end):
                if i != idx and not pd.isna(values[i]):
                    window_values.append(values[i])

            if window_values:
                filled_values[idx] = np.mean(window_values)

        df_filled[col] = filled_values

    return df_filled


def fill_missing_with_interpolation(df: pd.DataFrame, columns: list,
                                     method: str = 'linear') -> pd.DataFrame:
    """
    Fill missing values using pandas interpolation (alternative method).
    """
    df_filled = df.copy()

    for col in columns:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].interpolate(method=method)
            df_filled[col] = df_filled[col].ffill().bfill()

    return df_filled


def process_landsat_csv(
    input_file: str,
    output_file: str = None,
    window: int = 3,
    method: str = 'sma'
) -> pd.DataFrame:
    """
    Process Landsat CSV file and fill missing values.
    """
    print(f"Reading: {input_file}")
    df = pd.read_csv(input_file)

    # Landsat band columns
    band_columns = ['Red', 'NIR', 'SWIR1', 'SWIR2']

    missing_before = df[band_columns].isna().sum().sum()
    print(f"\nMissing values before filling: {missing_before}")

    if method == 'sma':
        print(f"Filling with Simple Moving Average (window={window})...")
        df_filled = fill_missing_with_sma(df, band_columns, window=window)
    else:
        print(f"Filling with {method} interpolation...")
        df_filled = fill_missing_with_interpolation(df, band_columns, method=method)

    missing_after = df_filled[band_columns].isna().sum().sum()
    print(f"Missing values after filling: {missing_after}")
    print(f"Values filled: {missing_before - missing_after}")

    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_filled{ext}"

    df_filled.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    print("\n--- Summary Statistics ---")
    print(df_filled[band_columns].describe())

    return df_filled


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Fill missing values in Landsat CSV using Simple Moving Average"
    )

    parser.add_argument(
        'input_file',
        nargs='?',
        default='landsat_Haryana_Karnal_2015.csv',
        help='Input CSV file'
    )
    parser.add_argument('-o', '--output', default=None, help='Output CSV file')
    parser.add_argument('-w', '--window', type=int, default=3, help='SMA window size (default: 3)')
    parser.add_argument(
        '-m', '--method',
        choices=['sma', 'linear', 'polynomial'],
        default='sma',
        help='Filling method (default: sma)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        return

    df_filled = process_landsat_csv(
        input_file=args.input_file,
        output_file=args.output,
        window=args.window,
        method=args.method
    )

    print("\n--- Sample of Filled Data (first 10 rows) ---")
    print(df_filled[['time_step', 'start_date', 'Red', 'NIR', 'SWIR1', 'SWIR2']].head(10))


if __name__ == "__main__":
    main()
