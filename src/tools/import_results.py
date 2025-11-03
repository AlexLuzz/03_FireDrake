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

# Default data file paths (edit these as needed)
DEFAULT_COMSOL_FILE = Path("./data_input/RAF_COMSOL_PZ_CG.csv")
DEFAULT_MEASURED_FILE = Path("./data_input/MEASURED_PZ_CG.csv")
DEFAULT_COMSOL_REF_DATE = datetime(2024, 2, 22)  # COMSOL t=0 reference date
DEFAULT_MEASURED_OFFSETS = {
    'LTC 101': 0.60,
    'LTC 102': 0.70, 
    'LTC 103': 0.35,
}


def load_comsol_data(csv_path: Union[str, Path] = None, 
                     start_from_days: float = 0.0, 
                     sim_duration_days: Optional[float] = None) -> Dict:
    """Load COMSOL simulation results"""
    csv_path = csv_path or DEFAULT_COMSOL_FILE
    loader = CSVLoader(str(csv_path))
    
    # Find time column and load data
    time_col = next((col for col in loader.columns if 'time' in col.lower() or col.lower() == 't'), None)
    if not time_col:
        raise ValueError(f"No time column in {csv_path}")
    
    times = loader.get_numeric(time_col) - start_from_days
    
    # Filter time range if specified
    if sim_duration_days:
        mask = (times >= 0) & (times <= sim_duration_days)
        if not mask.any():
            raise ValueError(f"No data in range 0 to {sim_duration_days} days")
        times = times[mask]
        loader.df = loader.df[mask]
    
    # Extract LTC columns
    result = {'times': times}
    result.update({col: loader.get_numeric(col) for col in loader.columns 
                   if col != time_col and col.upper().startswith('LTC')})
    
    print(f"✓ COMSOL: {list(result.keys())} ({len(times)} points)")
    return result


def load_measured_data(csv_path: Union[str, Path] = None,
                      time_converter: TimeConverter = None,
                      start_datetime: Optional[datetime] = None,
                      end_datetime: Optional[datetime] = None,
                      offsets: Optional[Dict[str, float]] = None,
                      smooth_window: int = 6,
                      hampel_window: int = 120,
                      align: bool = False,
                      align_freq: int = 6) -> Dict:
    """Load field measurement data with filtering and smoothing"""
    csv_path = csv_path or DEFAULT_MEASURED_FILE
    offsets = offsets or DEFAULT_MEASURED_OFFSETS
    
    # Find and load datetime column
    temp_loader = CSVLoader(str(csv_path))
    datetime_col = next((col for col in temp_loader.columns if 'time' in col.lower() or 'date' in col.lower()), None)
    if not datetime_col:
        raise ValueError(f"No datetime column in {csv_path}")
    
    loader = CSVLoader(str(csv_path), datetime_col)
    if start_datetime or end_datetime:
        loader.filter_dates(start_datetime, end_datetime)
    
    # Apply Hampel filter to data columns first (on raw CSV data)
    data_columns = []
    for col in loader.columns:
        # Standardize column names
        import re
        if 'level' in col.lower():
            match = re.search(r'(\d+)', col)
            name = f"LTC {match.group(1)}" if match else col
        elif col.upper().startswith('LTC'):
            name = col
        else:
            continue
        data_columns.append((col, name))
        
        # Apply Hampel filter on raw data
        if hampel_window > 0:
            loader.hampel_filter(col, window_size=hampel_window)
    
    # Apply alignment if requested
    if align and align_freq > 0:
        freq_str = f"{align_freq}h"  # Convert hours to frequency string
        loader.align(freq_str, method='interpolate')
    
    # Convert times after potential alignment
    datetimes = loader.get_datetimes()
    sim_times = np.array([time_converter.to_seconds(dt) / 86400.0 for dt in datetimes])
    result = {'times': sim_times, 'datetimes': datetimes}
    
    # Process data columns and apply remaining filters
    for col, name in data_columns:
        values = loader.get_numeric(col)
        
        # Apply smoothing filter
        if smooth_window > 1:
            values = pd.Series(values).rolling(smooth_window, center=True, min_periods=1).mean().values
        
        # Apply offset
        if name in offsets and offsets[name] != 0:
            values += offsets[name]
            print(f"  {name}: {offsets[name]:+.3f}m offset")
        
        result[name] = values
    
    print(f"✓ Measured: {[k for k in result if k not in ['times', 'datetimes']]} ({len(sim_times)} points)")
    return result


def preview_data(comsol_data: Optional[Dict],
                measured_data: Optional[Dict] = None,
                time_converter: Optional[TimeConverter] = None,
                use_datetime: bool = True,
                figsize=(12, 8)) -> plt.Figure:
    """Quick visualization of loaded data"""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    
    # Helper function to get x-axis data
    def get_x_data(data_dict, is_comsol=False):
        if not use_datetime or not time_converter:
            return data_dict['times']
        
        if is_comsol:
            # Convert COMSOL times (days) to datetime
            return [time_converter.to_datetime(t * 86400) for t in data_dict['times']]
        else:
            # Use measured datetimes, filter out pre-2020 data
            if 'datetimes' in data_dict:
                cutoff = datetime(2020, 1, 1)
                return [dt for dt in pd.to_datetime(data_dict['datetimes']) if dt >= cutoff]
            return data_dict['times']
    
    # Helper function to filter y-data for measured data
    def filter_y_data(x_data, y_data, original_times):
        if not use_datetime or len(x_data) == len(y_data):
            return y_data
        # Match filtered x_data length
        cutoff = datetime(2020, 1, 1)
        mask = np.array([dt >= cutoff for dt in pd.to_datetime(original_times)])
        return y_data[mask] if len(y_data) == len(mask) else y_data
    
    # Plot COMSOL data
    if comsol_data:
        x_comsol = get_x_data(comsol_data, is_comsol=True)
        for i, (key, values) in enumerate(comsol_data.items()):
            if key != 'times':
                ax.plot(x_comsol, values, color=colors[i % len(colors)], 
                       linestyle='--', linewidth=2, label=f'{key} (COMSOL)', alpha=0.8)
    
    # Plot measured data
    if measured_data:
        x_measured = get_x_data(measured_data, is_comsol=False)
        for i, (key, values) in enumerate(measured_data.items()):
            if key not in ['times', 'datetimes']:
                y_filtered = filter_y_data(x_measured, values, 
                                         measured_data.get('datetimes', measured_data['times']))
                ax.plot(x_measured[:len(y_filtered)], y_filtered, 
                       color=colors[i % len(colors)], linestyle=':', linewidth=1.5, 
                       marker='o', markersize=2, label=f'{key} (Measured)', alpha=0.7)
    
    # Format axes
    ax.set_ylabel('Water Table Elevation (m)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if use_datetime and time_converter:
        ax.set_xlabel('Date', fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_xlabel('Time (days)', fontweight='bold')
    
    plt.title('Data Preview', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig("data_preview.png", dpi=300, bbox_inches='tight')
    return fig


