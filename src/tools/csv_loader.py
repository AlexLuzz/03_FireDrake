import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Literal, Union

def load_csv(filepath: str, datetime_col: Optional[str] = None) -> pd.DataFrame:
    """Load CSV with auto-delimiter detection and smart datetime parsing."""
    # Auto-detect delimiter (checking the first line)
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        first_line = f.readline()
        delimiter = ';' if ';' in first_line else ','
    
    # Load and clean column names
    df = pd.read_csv(filepath, delimiter=delimiter, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    
    if datetime_col and datetime_col in df.columns:
        # Strategy 1: Fast pandas inference
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True)
        except (ValueError, TypeError):
            # Strategy 2: Explicit common formats
            formats = ['%Y-%m-%d', '%d/%m/%Y', '%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M']
            for fmt in formats:
                try:
                    df[datetime_col] = pd.to_datetime(df[datetime_col], format=fmt)
                    break
                except (ValueError, TypeError):
                    continue
        
        df = df.set_index(datetime_col).sort_index()
    
    return df

def filter_dates(df: pd.DataFrame, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
    """Filter DataFrame by a date range."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= (df.index >= start)
    if end:
        mask &= (df.index <= end)
        
    filtered_df = df[mask]
    if filtered_df.empty:
        print(f"Warning: No data found between {start} and {end}")
    return filtered_df

def align_data(df: pd.DataFrame, freq: str, method: Literal['ffill', 'bfill', 'interpolate'] = 'interpolate') -> pd.DataFrame:
    """Resample data to a fixed frequency (e.g., '1h', '15min')."""
    resampler = df.resample(freq)
    
    if method == 'interpolate':
        return resampler.interpolate(method='linear')
    return getattr(resampler, method)()

def smooth_data(df: pd.DataFrame, window: int, method: Literal['mean', 'median'] = 'mean') -> pd.DataFrame:
    """Apply a rolling window smoothing."""
    rolling = df.rolling(window=window, center=True, min_periods=1)
    return getattr(rolling, method)()

def apply_hampel_filter(series: pd.Series, window_size: int = 5, n_sigma: float = 3.0) -> pd.Series:
    """Detect and replace outliers with the local median."""
    # Convert to numeric if it's a string/comma-decimal format
    if series.dtype == object:
        series = pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')
        
    filtered = series.copy()
    L = 1.4826 # Scale factor for Gaussian distribution
    
    rolling_median = series.rolling(window=window_size, center=True).median()
    rolling_mad = series.rolling(window=window_size, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    lower_bound = rolling_median - (n_sigma * L * rolling_mad)
    upper_bound = rolling_median + (n_sigma * L * rolling_mad)
    
    outliers = (series < lower_bound) | (series > upper_bound)
    filtered[outliers] = rolling_median[outliers]
    return filtered

def get_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract a column as a clean numpy float array."""
    values = df[col].astype(str).str.replace(',', '.', regex=False)
    return pd.to_numeric(values, errors='coerce').values

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # 1. Load data
    # df = load_csv("data.csv", datetime_col="Timestamp")
    
    # Dummy data for demonstration
    data = {
        'Timestamp': ['01/01/2024 10:00', '01/01/2024 10:05', '01/01/2024 10:10'],
        'Value': ['10,5', '100,0', '11,2'] # Includes an outlier and comma decimals
    }
    df_demo = pd.DataFrame(data)
    df_demo['Timestamp'] = pd.to_datetime(df_demo['Timestamp'])
    df_demo = df_demo.set_index('Timestamp')

    print("Original Data:\n", df_demo)

    # 2. Extracting values (get_... equivalents)
    vals = get_numeric(df_demo, 'Value')
    times = df_demo.index.values
    print(f"\nExtracted Values: {vals}")
    print(f"Extracted Times: {times[:2]}...")

    # 3. Clean and Filter
    df_demo['Value'] = apply_hampel_filter(df_demo['Value'])
    df_filtered = filter_dates(df_demo, start=datetime(2024, 1, 1, 10, 0))
    
    print("\nProcessed Data (Outlier removed):\n", df_filtered)