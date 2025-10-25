"""
Data import utilities for simulation results comparison
Handles COMSOL, measured data, and other external datasets
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .csv_loader import CSVLoader
from .time_converter import TimeConverter


@dataclass
class PiezometerOffset:
    """Individual piezometer offset configuration"""
    name: str
    vertical_offset: float = 0.0  # Vertical offset in meters
    y_position: float = None      # Y position if different from default
    display_name: str = None      # Custom display name
    
    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.name


class ImportResults:
    """
    Centralized data import and management for simulation results comparison
    
    Handles:
    - COMSOL simulation results
    - Field measurement data  
    - Data alignment and filtering
    - Individual piezometer offsets
    - Basic visualization
    """
    
    def __init__(self, time_converter: Optional[TimeConverter] = None):
        """
        Initialize ImportResults
        
        Args:
            time_converter: TimeConverter for datetime handling (optional)
        """
        self.time_converter = time_converter
        self.comsol_data = None
        self.measured_data = None
        self.piezometer_offsets = {}
        
    def set_time_converter(self, time_converter: TimeConverter):
        """Set or update the time converter"""
        self.time_converter = time_converter
        
    def add_piezometer_offset(self, piezometer_id: str, vertical_offset: float = 0.0, 
                             y_position: float = None, display_name: str = None):
        """
        Add individual offset for a specific piezometer
        
        Args:
            piezometer_id: Piezometer identifier (e.g., "LTC 101", "LTC 1")
            vertical_offset: Vertical offset in meters (positive = up)
            y_position: Y position if different from default
            display_name: Custom display name for plots
        """
        self.piezometer_offsets[piezometer_id] = PiezometerOffset(
            name=piezometer_id,
            vertical_offset=vertical_offset,
            y_position=y_position,
            display_name=display_name
        )
        
    def load_comsol_data(self, csv_path: Union[str, Path], 
                        start_from_days: float = 0.0, 
                        sim_duration_days: Optional[float] = None,
                        ref_date: Optional[datetime] = None) -> Dict:
        """
        Load COMSOL simulation results
        
        Args:
            csv_path: Path to COMSOL CSV file
            start_from_days: Time offset in days to align with simulation
            sim_duration_days: Duration of simulation in days
            ref_date: Reference datetime for alignment
            
        Returns:
            Dictionary with 'times' and LTC data columns
        """
        try:
            # Load CSV data
            loader = CSVLoader(str(csv_path))
            
            # Find time column (should be numeric for COMSOL)
            time_col = None
            for col in loader.columns:
                if any(term in col.lower() for term in ['time', 't']):
                    time_col = col
                    break
                    
            if not time_col:
                raise ValueError(f"No time column found in {csv_path}")
            
            # Get time values (in days)
            time_values = loader.get_numeric(time_col)
            
            # Filter to simulation period if specified
            if sim_duration_days:
                end_time = start_from_days + sim_duration_days
                mask = (time_values >= start_from_days) & (time_values <= end_time)
                if not mask.any():
                    raise ValueError(f"No COMSOL data in time range {start_from_days:.1f} to {end_time:.1f} days")
                
                # Filter the data
                loader.df = loader.df[mask]
                time_values = time_values[mask]
            
            # Convert to simulation time (days relative to start_from_days)
            sim_times = time_values - start_from_days
            
            # Load LTC columns
            result = {'times': sim_times}
            for col in loader.columns:
                if col != time_col and (col.startswith('LTC') or col.startswith('ltc')):
                    result[col] = loader.get_numeric(col)
            
            self.comsol_data = result
            print(f"✓ Loaded COMSOL data: {list(result.keys())} ({len(sim_times)} points)")
            return result
            
        except Exception as e:
            print(f"⚠️  Error loading COMSOL data from {csv_path}: {e}")
            return None
    
    def load_measured_data(self, csv_path: Union[str, Path],
                          start_datetime: Optional[datetime] = None,
                          end_datetime: Optional[datetime] = None,
                          apply_offsets: bool = True) -> Dict:
        """
        Load field measurement data with individual piezometer offsets
        
        Args:
            csv_path: Path to measured data CSV file
            start_datetime: Start of data period
            end_datetime: End of data period
            apply_offsets: Whether to apply individual piezometer offsets
            
        Returns:
            Dictionary with 'times', 'datetimes' and LTC data columns
        """
        try:
            # Find datetime column
            loader_temp = CSVLoader(str(csv_path))
            datetime_col = None
            for col in loader_temp.columns:
                if any(term in col.lower() for term in ['time', 'date', 'datetime']):
                    datetime_col = col
                    break
                    
            if not datetime_col:
                raise ValueError(f"No datetime column found in {csv_path}")
            
            # Load with datetime index
            loader = CSVLoader(str(csv_path), datetime_col)
            
            # Filter by date range if specified
            if start_datetime or end_datetime:
                loader.filter_dates(start_datetime, end_datetime)
            
            # Get datetimes
            datetimes = loader.get_datetimes()
            
            # Convert to simulation time if time_converter available
            if self.time_converter:
                sim_times = np.array([self.time_converter.to_seconds(dt) / 86400.0 
                                    for dt in datetimes])
            else:
                # Use hours from start as fallback
                start_dt = pd.Timestamp(datetimes[0])
                sim_times = np.array([(pd.Timestamp(dt) - start_dt).total_seconds() / 86400.0 
                                    for dt in datetimes])
            
            # Prepare result dictionary
            result = {
                'times': sim_times,
                'datetimes': datetimes
            }
            
            # Load data columns with individual offsets
            for col in loader.columns:
                if col != datetime_col and ('level' in col.lower() or col.startswith('LTC')):
                    # Standardize column names: "Level 101" -> "LTC 101"
                    if 'level' in col.lower():
                        import re
                        match = re.search(r'(\d+)', col)
                        if match:
                            num = match.group(1)
                            standardized_name = f"LTC {num}"
                        else:
                            standardized_name = col
                    else:
                        standardized_name = col
                    
                    # Get raw values
                    values = loader.get_numeric(col)
                    
                    # Apply individual offset if configured and requested
                    if apply_offsets and standardized_name in self.piezometer_offsets:
                        offset_config = self.piezometer_offsets[standardized_name]
                        values = values + offset_config.vertical_offset
                        print(f"  Applied {offset_config.vertical_offset:.3f}m offset to {standardized_name}")
                    
                    result[standardized_name] = values
            
            self.measured_data = result
            print(f"✓ Loaded measured data: {list(result.keys())} ({len(sim_times)} points)")
            return result
            
        except Exception as e:
            print(f"⚠️  Error loading measured data from {csv_path}: {e}")
            return None
    
    def preview_data(self, figsize: Tuple[float, float] = (12, 8), 
                    show_offsets: bool = True) -> plt.Figure:
        """
        Create a simple visualization of loaded data
        
        Args:
            figsize: Figure size (width, height)
            show_offsets: Whether to show offset information in legend
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot COMSOL data
        if self.comsol_data:
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            for i, (key, values) in enumerate(self.comsol_data.items()):
                if key != 'times':
                    color = colors[i % len(colors)]
                    ax.plot(self.comsol_data['times'], values, 
                           color=color, linestyle='--', linewidth=2,
                           label=f'{key} - COMSOL', alpha=0.8)
        
        # Plot measured data
        if self.measured_data:
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            for i, (key, values) in enumerate(self.measured_data.items()):
                if key not in ['times', 'datetimes']:
                    color = colors[i % len(colors)]
                    
                    # Prepare label with offset info if requested
                    label = f'{key} - Measured'
                    if show_offsets and key in self.piezometer_offsets:
                        offset = self.piezometer_offsets[key].vertical_offset
                        if offset != 0:
                            label += f' (+{offset:.3f}m)'
                    
                    # Use datetime axis if available
                    if 'datetimes' in self.measured_data and self.time_converter:
                        x_data = [self.time_converter.to_datetime(t * 86400) 
                                for t in self.measured_data['times']]
                        ax.plot(x_data, values, color=color, linestyle=':', 
                               linewidth=1.5, marker='o', markersize=2,
                               label=label, alpha=0.7)
                    else:
                        ax.plot(self.measured_data['times'], values, 
                               color=color, linestyle=':', linewidth=1.5,
                               marker='o', markersize=2, label=label, alpha=0.7)
        
        # Formatting
        ax.set_ylabel('Water Table Elevation (m)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Handle datetime axis
        if (self.measured_data and 'datetimes' in self.measured_data and 
            self.time_converter):
            ax.set_xlabel('Date', fontweight='bold')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.set_xlabel('Time (days)', fontweight='bold')
        
        plt.title('Imported Data Preview', fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def get_data_summary(self) -> Dict:
        """
        Get summary information about loaded datasets
        
        Returns:
            Dictionary with data summary information
        """
        summary = {
            'comsol_loaded': self.comsol_data is not None,
            'measured_loaded': self.measured_data is not None,
            'piezometer_offsets': len(self.piezometer_offsets),
            'has_time_converter': self.time_converter is not None
        }
        
        if self.comsol_data:
            summary['comsol_columns'] = [k for k in self.comsol_data.keys() if k != 'times']
            summary['comsol_time_range'] = (
                float(self.comsol_data['times'][0]),
                float(self.comsol_data['times'][-1])
            )
            summary['comsol_points'] = len(self.comsol_data['times'])
        
        if self.measured_data:
            summary['measured_columns'] = [k for k in self.measured_data.keys() 
                                         if k not in ['times', 'datetimes']]
            summary['measured_time_range'] = (
                float(self.measured_data['times'][0]),
                float(self.measured_data['times'][-1])
            )
            summary['measured_points'] = len(self.measured_data['times'])
            
            if 'datetimes' in self.measured_data:
                summary['measured_date_range'] = (
                    pd.Timestamp(self.measured_data['datetimes'][0]).strftime('%Y-%m-%d'),
                    pd.Timestamp(self.measured_data['datetimes'][-1]).strftime('%Y-%m-%d')
                )
        
        return summary
    
    def calculate_residuals(self, simulation_data: Dict, data_type: str = 'comsol') -> Dict:
        """
        Calculate residuals between simulation and reference data
        
        Args:
            simulation_data: Dictionary with simulation results
            data_type: 'comsol' or 'measured' - which reference to use
            
        Returns:
            Dictionary with residual data
        """
        reference_data = self.comsol_data if data_type == 'comsol' else self.measured_data
        
        if not reference_data:
            raise ValueError(f"No {data_type} data loaded for residual calculation")
        
        # Interpolate reference data to simulation time points
        from scipy.interpolate import interp1d
        
        residuals = {'times': simulation_data['times']}
        
        for key in simulation_data.keys():
            if key != 'times' and key in reference_data:
                # Create interpolation function
                interp_func = interp1d(reference_data['times'], reference_data[key],
                                     kind='linear', bounds_error=False, fill_value=np.nan)
                
                # Interpolate reference to simulation times
                ref_interp = interp_func(simulation_data['times'])
                
                # Calculate residuals
                residuals[key] = simulation_data[key] - ref_interp
        
        return residuals