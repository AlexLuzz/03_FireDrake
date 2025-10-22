"""
Time conversion utilities for datetime ↔ seconds conversions
Handles CSV datetime parsing and plotting time axes
"""
from datetime import datetime, timedelta
from typing import Union, Optional, List
import numpy as np
import csv


class TimeConverter:
    """
    Handles conversions between datetime and seconds for simulation
    
    Example:
        converter = TimeConverter(start=datetime(2024, 6, 1, 0, 0))
        
        # Convert datetime to simulation seconds
        t_sec = converter.to_seconds(datetime(2024, 6, 3, 12, 0))  # 2.5 days = 216000 s
        
        # Convert simulation seconds back to datetime
        dt = converter.to_datetime(216000)  # 2024-06-03 12:00:00
    """
    
    def __init__(self, start_datetime: datetime):
        """
        Initialize converter with simulation start time
        
        Args:
            start_datetime: Reference datetime (t=0 in simulation)
        """
        self.start = start_datetime
    
    def to_seconds(self, dt: Union[datetime, timedelta]) -> float:
        """
        Convert datetime or timedelta to seconds from simulation start
        
        Args:
            dt: Either a datetime or timedelta object
            
        Returns:
            Time in seconds from simulation start
        """
        if isinstance(dt, datetime):
            return (dt - self.start).total_seconds()
        elif isinstance(dt, timedelta):
            return dt.total_seconds()
        else:
            raise TypeError(f"Expected datetime or timedelta, got {type(dt)}")
    
    def to_datetime(self, seconds: float) -> datetime:
        """
        Convert simulation seconds to datetime
        
        Args:
            seconds: Time in seconds from simulation start
            
        Returns:
            Corresponding datetime object
        """
        return self.start + timedelta(seconds=seconds)
    
    def to_hours(self, seconds: float) -> float:
        """Convert seconds to hours"""
        return seconds / 3600.0
    
    def to_days(self, seconds: float) -> float:
        """Convert seconds to days"""
        return seconds / 86400.0
    
    def format_time(self, seconds: float, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format simulation time as string
        
        Args:
            seconds: Simulation time in seconds
            format_str: Datetime format string
            
        Returns:
            Formatted datetime string
        """
        return self.to_datetime(seconds).strftime(format_str)
    
    def parse_datetime(self, date_str: str, format_str: Optional[str] = None) -> datetime:
        """
        Parse datetime string with automatic format detection
        
        Args:
            date_str: String representation of date/datetime
            format_str: Optional format string. If None, tries common formats
            
        Returns:
            Parsed datetime object
        """
        if format_str:
            return datetime.strptime(date_str, format_str)
        
        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Could not parse datetime string: '{date_str}'. "
                        f"Try specifying format_str explicitly.")
    
    def load_datetime_csv(self, 
                         csv_path: str,
                         datetime_column: str = 'Date',
                         value_columns: Optional[List[str]] = None,
                         datetime_format: Optional[str] = None,
                         delimiter: Optional[str] = None,
                         encoding: str = 'utf-8-sig',
                         start_datetime: Optional[datetime] = None,
                         end_datetime: Optional[datetime] = None) -> dict:
        """
        Load CSV with datetime column and convert to simulation seconds
        
        Args:
            csv_path: Path to CSV file
            datetime_column: Name of datetime column
            value_columns: Names of value columns to load (None = all except datetime)
            datetime_format: Format string for datetime parsing (None = auto-detect)
            delimiter: CSV delimiter (None = auto-detect from ;, or \t)
            encoding: File encoding
            start_datetime: Filter data >= this datetime (None = no filter)
            end_datetime: Filter data <= this datetime (None = no filter)
            
        Returns:
            Dictionary with 'times' (in seconds) and value columns
            
        Example:
            # For BB_METEO.csv with format:
            # Date/Heure;Pluie tot. (mm);Neige tot. (cm)
            # 2024-01-01;0;0
            # 2024-01-02;1.5;0
            
            converter = TimeConverter(start_datetime=datetime(2024, 1, 1))
            data = converter.load_datetime_csv(
                'BB_METEO.csv',
                datetime_column='Date/Heure',
                value_columns=['Pluie tot. (mm)'],
                delimiter=';',
                start_datetime=datetime(2024, 8, 1),
                end_datetime=datetime(2024, 8, 31)
            )
            
            # Returns: {'times': [0, 86400, ...], 'Pluie tot. (mm)': [0, 1.5, ...]}
        """
        # Auto-detect delimiter
        if delimiter is None:
            with open(csv_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                delimiter = ';' if ';' in first_line else ',' if ',' in first_line else '\t'
        
        # Read CSV
        with open(csv_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = list(reader)
        
        if not rows:
            raise ValueError(f"Empty CSV file: {csv_path}")
        
        # Find datetime column (case-insensitive partial match)
        header = list(rows[0].keys())
        datetime_col = None
        for col in header:
            if datetime_column.lower() in col.lower() or col.lower() in datetime_column.lower():
                datetime_col = col
                break
        
        if datetime_col is None:
            raise KeyError(f"Datetime column '{datetime_column}' not found. Available: {header}")
        
        # Determine value columns
        if value_columns is None:
            value_cols = [col for col in header if col != datetime_col]
        else:
            # Find matching columns (partial, case-insensitive)
            value_cols = []
            for target in value_columns:
                for col in header:
                    if target.lower() in col.lower() or col.lower() in target.lower():
                        value_cols.append(col)
                        break
        
        # Parse data
        datetimes = []
        data_dict = {col: [] for col in value_cols}
        
        for row in rows:
            # Parse datetime
            date_str = row[datetime_col].strip()
            try:
                dt = self.parse_datetime(date_str, datetime_format)
                
                # Apply datetime filters
                if start_datetime and dt < start_datetime:
                    continue
                if end_datetime and dt > end_datetime:
                    continue
                
                datetimes.append(dt)
                
                # Parse values (handle comma decimal separator)
                for col in value_cols:
                    val_str = row[col].strip().replace(',', '.')
                    try:
                        data_dict[col].append(float(val_str))
                    except ValueError:
                        data_dict[col].append(np.nan)
            except (ValueError, KeyError) as e:
                # Skip rows that can't be parsed
                continue
        
        if not datetimes:
            raise ValueError(f"No valid datetime entries found in {csv_path}")
        
        # Convert to simulation seconds
        times_seconds = np.array([self.to_seconds(dt) for dt in datetimes])
        
        # Build result dictionary
        result = {'times': times_seconds, 'datetimes': np.array(datetimes)}
        for col in value_cols:
            result[col] = np.array(data_dict[col])
        
        print(f"✓ Loaded {csv_path}")
        print(f"  Date range: {datetimes[0]} to {datetimes[-1]}")
        if start_datetime or end_datetime:
            print(f"  Filtered: {start_datetime or 'any'} to {end_datetime or 'any'}")
        print(f"  Simulation time: 0s to {times_seconds[-1]:.0f}s ({times_seconds[-1]/86400:.1f} days)")
        print(f"  Columns: {value_cols}")
        
        return result
    
    def create_datetime_array(self, times_seconds: np.ndarray) -> np.ndarray:
        """
        Convert array of simulation seconds to datetime array for plotting
        
        Args:
            times_seconds: Array of times in seconds from simulation start
            
        Returns:
            Array of datetime objects
        """
        return np.array([self.to_datetime(t) for t in times_seconds])
