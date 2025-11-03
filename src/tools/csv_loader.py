"""
Dead simple CSV loader with datetime handling
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Literal


class CSVLoader:
    """Load CSV with smart delimiter/datetime detection"""
    
    def __init__(self, filepath: str, datetime_col: Optional[str] = None):
        self.filepath = filepath
        self.datetime_col = datetime_col
        self.df = self._load(datetime_col)
    
    def _load(self, datetime_col: Optional[str]) -> pd.DataFrame:
        # Auto-detect delimiter
        with open(self.filepath, 'r', encoding='utf-8-sig') as f:
            delimiter = ';' if ';' in f.readline() else ','
        
        # Load and clean
        df = pd.read_csv(self.filepath, delimiter=delimiter, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        
        # Parse datetime if specified
        if datetime_col and datetime_col in df.columns:
            # Try different datetime parsing strategies
            try:
                # First try automatic inference (pandas 2.0+ handles this automatically)
                df[datetime_col] = pd.to_datetime(df[datetime_col])
            except:
                try:
                    # Try with dayfirst=True for European formats
                    df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True)
                except:
                    try:
                        # Try common formats explicitly
                        formats_to_try = [
                            '%Y-%m-%d',        # 2024-01-01
                            '%d/%m/%Y',        # 01/01/2024  
                            '%d/%m/%Y %H:%M',  # 01/01/2024 12:00
                            '%Y-%m-%d %H:%M'   # 2024-01-01 12:00
                        ]
                        parsed = False
                        for fmt in formats_to_try:
                            try:
                                df[datetime_col] = pd.to_datetime(df[datetime_col], format=fmt)
                                parsed = True
                                break
                            except:
                                continue
                        if not parsed:
                            raise ValueError(f"Could not parse datetime column '{datetime_col}'")
                    except Exception as e:
                        raise ValueError(f"Failed to parse datetime column '{datetime_col}': {e}")
            
            df = df.set_index(datetime_col).sort_index()
        
        return df
    
    def filter_dates(self, start: Optional[datetime] = None, 
                     end: Optional[datetime] = None):
        """Filter by date range (requires datetime_col set at init)"""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index (set datetime_col at init)")
        
        if start:
            self.df = self.df[self.df.index >= start]
        if end:
            self.df = self.df[self.df.index <= end]
        
        if self.df.empty:
            raise ValueError(f"No data between {start} and {end}")
        
        return self
    
    def align(self, freq: str, method: Literal['ffill', 'bfill', 'interpolate'] = 'interpolate'):
        """
        Resample to regular intervals
        
        Args:
            freq: pandas frequency string ('h'=hourly, 'D'=daily, '15min', etc.)
            method: 'ffill' (forward fill), 'bfill' (backward fill), 'interpolate' (linear)
        
        Example:
            loader.align('h')  # Hourly intervals with linear interpolation
            loader.align('D', 'ffill')  # Daily with forward fill
        """
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index (set datetime_col at init)")
        
        # Resample to regular frequency
        df_resampled = self.df.resample(freq)
        
        if method == 'interpolate':
            self.df = df_resampled.interpolate(method='linear')
        elif method == 'ffill':
            self.df = df_resampled.ffill()
        elif method == 'bfill':
            self.df = df_resampled.bfill()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self
    
    def smooth(self, window: int, method: Literal['mean', 'median'] = 'mean'):
        """
        Smooth data with rolling window
        
        Args:
            window: number of points in rolling window
            method: 'mean' or 'median'
        
        Example:
            loader.smooth(3)  # 3-point moving average
            loader.smooth(5, 'median')  # 5-point median filter
        """
        if method == 'mean':
            self.df = self.df.rolling(window, center=True, min_periods=1).mean()
        elif method == 'median':
            self.df = self.df.rolling(window, center=True, min_periods=1).median()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self
    
    def hampel_filter(self, col: str, window_size: int = 5, n_sigma: float = 3.0):
        """
        Remove outliers using Hampel filter (median absolute deviation)
        
        Args:
            col: Column name to filter
            window_size: Moving window size (must be odd)
            n_sigma: Threshold in standard deviations
        
        Returns:
            self for method chaining
        """
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found")
        
        # Get numeric data
        values = self.get_numeric(col)
        filtered = values.copy()
        half_window = window_size // 2
        
        for i in range(half_window, len(values) - half_window):
            window = values[i - half_window : i + half_window + 1]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            
            if mad > 0:  # Avoid division by zero
                threshold = n_sigma * 1.4826 * mad  # 1.4826 converts MAD to std
                if np.abs(values[i] - median) > threshold:
                    filtered[i] = median
        
        # Update the dataframe
        self.df[col] = filtered
        return self
    
    def get_numeric(self, col: str):
        """Get column as numeric (handles comma decimals)"""
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found")
        
        values = self.df[col].astype(str).str.replace(',', '.', regex=False)
        return pd.to_numeric(values, errors='coerce').values
    
    def get_column(self, col: str):
        """Get column as-is"""
        return self.df[col].values
    
    def get_datetimes(self):
        """Get datetime index as array"""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index (set datetime_col at init)")
        return self.df.index.values
    
    @property
    def columns(self):
        return self.df.columns.tolist()
    
    @property
    def length(self):
        return len(self.df)