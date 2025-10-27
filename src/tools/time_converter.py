"""
Time conversion utilities for datetime ↔ seconds conversions
"""
from datetime import datetime, timedelta
from typing import Union, Optional, List
import numpy as np
import pandas as pd
from .csv_loader import CSVLoader

class TimeConverter:
    """
    Simple converter between datetime and simulation seconds
    
    Example:
        converter = TimeConverter(datetime(2024, 6, 1))
        t_sec = converter.to_seconds(datetime(2024, 6, 3, 12, 0))  # 2.5 days
        dt = converter.to_datetime(216000)  # back to datetime
    """
    
    def __init__(self, start_datetime: datetime):
        self.start = start_datetime
    
    def to_seconds(self, dt: Union[datetime, pd.Timestamp, np.datetime64]) -> float:
        """Convert datetime to seconds from simulation start"""
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        elif isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt).to_pydatetime()
        return (dt - self.start).total_seconds()
    
    def to_datetime(self, seconds: Union[float, int]) -> datetime:
        """Convert simulation seconds to datetime"""
        return self.start + timedelta(seconds=float(seconds))
    
    def to_hours(self, dt_or_seconds: Union[datetime, pd.Timestamp, np.datetime64, float]) -> float:
        """Convert datetime or seconds to hours from simulation start"""
        if isinstance(dt_or_seconds, (datetime, pd.Timestamp, np.datetime64)):
            return self.to_seconds(dt_or_seconds) / 3600.0
        else:
            return float(dt_or_seconds) / 3600.0
    
    def to_days(self, dt_or_seconds: Union[datetime, pd.Timestamp, np.datetime64, float]) -> float:
        """Convert datetime or seconds to days from simulation start"""
        if isinstance(dt_or_seconds, (datetime, pd.Timestamp, np.datetime64)):
            return self.to_seconds(dt_or_seconds) / 86400.0
        else:
            return float(dt_or_seconds) / 86400.0

    def load_datetime_csv(self, csv_path: str, datetime_column: str = 'Date',
                         value_columns: Optional[List[str]] = None,
                         start_datetime: Optional[datetime] = None,
                         end_datetime: Optional[datetime] = None) -> dict:
        """Load CSV with datetime column and convert to simulation seconds"""
        
        loader = CSVLoader(csv_path, datetime_column)
        
        if start_datetime or end_datetime:
            loader.filter_dates(datetime_column, start_datetime, end_datetime)
        
        # Convert to simulation seconds
        datetimes = loader.get_column(datetime_column)
        times_seconds = np.array([self.to_seconds(dt) for dt in datetimes])
        
        # Build result dictionary
        result = {'times': times_seconds, 'datetimes': datetimes}
        
        # Add value columns
        if value_columns is None:
            value_columns = [col for col in loader.columns if col != datetime_column]
        
        for col in value_columns:
            result[col] = loader.get_numeric(col)
        
        return result
