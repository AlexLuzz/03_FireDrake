"""
Simple data import utilities for simulation results
No classes, just clean functions with sensible defaults
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime
from scipy.interpolate import interp1d

from .csv_loader import CSVLoader
from .time_converter import TimeConverter


# Default piezometer offsets (edit these as needed)
DEFAULT_OFFSETS = {
    'LTC 101': 0.0,
    'LTC 102': 0.0,
    'LTC 103': 0.0,
}


def hampel_filter(data: np.ndarray, window_size: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """
    Remove outliers using Hampel filter (median absolute deviation)
    
    Args:
        data: Input array
        window_size: Moving window size (must be odd)
        n_sigma: Threshold in standard deviations
    """
    filtered = data.copy()
    half_window = window_size // 2
    
    for i in range(half_window, len(data) - half_window):
        window = data[i - half_window : i + half_window + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        
        if mad > 0:  # Avoid division by zero
            threshold = n_sigma * 1.4826 * mad  # 1.4826 converts MAD to std
            if np.abs(data[i] - median) > threshold:
                filtered[i] = median
    
    return filtered


def load_comsol_data(csv_path: Union[str, Path], 
                     start_from_days: float = 0.0, 
                     sim_duration_days: Optional[float] = None) -> Dict:
    """
    Load COMSOL simulation results
    
    Returns dict with 'times' (days) and LTC columns
    """
    loader = CSVLoader(str(csv_path))
    
    # Find time column
    time_col = next((col for col in loader.columns 
                    if any(t in col.lower() for t in ['time', 't'])), None)
    if not time_col:
        raise ValueError(f"No time column in {csv_path}")
    
    time_values = loader.get_numeric(time_col)
    
    # Filter to simulation period
    if sim_duration_days:
        end_time = start_from_days + sim_duration_days
        mask = (time_values >= start_from_days) & (time_values <= end_time)
        if not mask.any():
            raise ValueError(f"No data between {start_from_days} and {end_time} days")
        loader.df = loader.df[mask]
        time_values = time_values[mask]
    
    # Extract data
    result = {'times': time_values - start_from_days}
    for col in loader.columns:
        if col != time_col and col.upper().startswith('LTC'):
            result[col] = loader.get_numeric(col)
    
    print(f"✓ COMSOL: {list(result.keys())} ({len(result['times'])} points)")
    return result


def load_measured_data(csv_path: Union[str, Path],
                      time_converter: TimeConverter,
                      start_datetime: Optional[datetime] = None,
                      end_datetime: Optional[datetime] = None,
                      offsets: Optional[Dict[str, float]] = None,
                      smooth_window: int = 4,
                      hampel_window: int = 5,
                      hampel_sigma: float = 3.0) -> Dict:
    """
    Load field measurement data with filtering and smoothing
    
    Args:
        smooth_window: Sliding window for smoothing (0 to disable)
        hampel_window: Window for Hampel filter (0 to disable)
        hampel_sigma: Outlier threshold in sigmas
        offsets: Custom offsets dict (uses DEFAULT_OFFSETS if None)
    
    Returns dict with 'times' (days), 'datetimes', and LTC columns
    """
    if offsets is None:
        offsets = DEFAULT_OFFSETS
    
    # Find datetime column
    loader_temp = CSVLoader(str(csv_path))
    datetime_col = next((col for col in loader_temp.columns 
                        if any(t in col.lower() for t in ['time', 'date'])), None)
    if not datetime_col:
        raise ValueError(f"No datetime column in {csv_path}")
    
    # Load with datetime index
    loader = CSVLoader(str(csv_path), datetime_col)
    if start_datetime or end_datetime:
        loader.filter_dates(start_datetime, end_datetime)
    
    # Convert times
    datetimes = loader.get_datetimes()
    sim_times = np.array([time_converter.to_seconds(dt) / 86400.0 for dt in datetimes])
    
    result = {'times': sim_times, 'datetimes': datetimes}
    
    # Process each data column
    for col in loader.columns:
        if col == datetime_col:
            continue
        
        # Standardize names: "Level 101" -> "LTC 101"
        if 'level' in col.lower():
            import re
            match = re.search(r'(\d+)', col)
            name = f"LTC {match.group(1)}" if match else col
        elif col.upper().startswith('LTC'):
            name = col
        else:
            continue
        
        values = loader.get_numeric(col)
        
        # 1. Hampel filter (remove outliers)
        if hampel_window > 0:
            values = hampel_filter(values, window_size=hampel_window, n_sigma=hampel_sigma)
        
        # 2. Smooth (reduce noise)
        if smooth_window > 1:
            values = pd.Series(values).rolling(
                smooth_window, center=True, min_periods=1
            ).mean().values
        
        # 3. Apply offset
        if name in offsets and offsets[name] != 0:
            values = values + offsets[name]
            print(f"  {name}: {offsets[name]:+.3f}m offset")
        
        result[name] = values
    
    print(f"✓ Measured: {[k for k in result if k not in ['times', 'datetimes']]} ({len(sim_times)} points)")
    return result


def calculate_residuals(simulation_data: Dict, reference_data: Dict) -> Dict:
    """Calculate residuals: simulation - reference (interpolated to sim times)"""
    residuals = {'times': simulation_data['times']}
    
    for key in simulation_data.keys():
        if key != 'times' and key in reference_data:
            # Interpolate reference to simulation times
            interp_func = interp1d(
                reference_data['times'], reference_data[key],
                kind='linear', bounds_error=False, fill_value=np.nan
            )
            ref_interp = interp_func(simulation_data['times'])
            residuals[key] = simulation_data[key] - ref_interp
    
    return residuals


def preview_data(comsol_data: Optional[Dict] = None,
                measured_data: Optional[Dict] = None,
                time_converter: Optional[TimeConverter] = None,
                figsize=(12, 8)) -> plt.Figure:
    """Quick visualization of loaded data"""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    
    # Plot COMSOL
    if comsol_data:
        for i, (key, values) in enumerate(comsol_data.items()):
            if key != 'times':
                ax.plot(comsol_data['times'], values, 
                       color=colors[i % len(colors)], linestyle='--', 
                       linewidth=2, label=f'{key} (COMSOL)', alpha=0.8)
    
    # Plot measured
    if measured_data:
        use_datetime = 'datetimes' in measured_data and time_converter
        
        for i, (key, values) in enumerate(measured_data.items()):
            if key in ['times', 'datetimes']:
                continue
            
            if use_datetime:
                x = [time_converter.to_datetime(t * 86400) for t in measured_data['times']]
            else:
                x = measured_data['times']
            
            ax.plot(x, values, color=colors[i % len(colors)], 
                   linestyle=':', linewidth=1.5, marker='o', markersize=2,
                   label=f'{key} (Measured)', alpha=0.7)
    
    # Format
    ax.set_ylabel('Water Table Elevation (m)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if measured_data and 'datetimes' in measured_data and time_converter:
        ax.set_xlabel('Date', fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_xlabel('Time (days)', fontweight='bold')
    
    plt.title('Data Preview', fontweight='bold', fontsize=14)
    plt.tight_layout()
    return fig