"""
Time conversion utilities for datetime ↔ seconds conversions
"""
from datetime import datetime, timedelta
from typing import Union
import numpy as np
import pandas as pd

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

