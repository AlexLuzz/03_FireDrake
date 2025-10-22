"""
Data import utilities for CSV files with standardized column naming
"""
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union


class CSVImporter:
    """
    Base class for importing CSV files with various formats
    
    Handles:
    - Different delimiters (comma, semicolon, tab)
    - Different decimal separators (. or ,)
    - Different encodings (UTF-8, UTF-8-sig, latin1)
    - Header detection
    - Column name mapping
    """
    
    @staticmethod
    def detect_format(filepath: Union[str, Path]) -> Dict[str, str]:
        """
        Auto-detect CSV format
        
        Returns:
            Dictionary with 'delimiter', 'decimal', 'encoding'
        """
        filepath = Path(filepath)
        
        # Try different encodings
        for encoding in ['utf-8-sig', 'utf-8', 'latin1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    first_lines = [f.readline() for _ in range(5)]
                    
                # Detect delimiter (count occurrences in first lines)
                delimiters = [',', ';', '\t', '|']
                delimiter_counts = {d: sum(line.count(d) for line in first_lines) 
                                   for d in delimiters}
                delimiter = max(delimiter_counts, key=delimiter_counts.get)
                
                # Detect decimal separator (look for numbers)
                text = ''.join(first_lines)
                uses_comma_decimal = ',' in text and any(c.isdigit() for c in text.split(',')[0][-3:])
                decimal = ',' if uses_comma_decimal else '.'
                
                return {
                    'delimiter': delimiter,
                    'decimal': decimal,
                    'encoding': encoding
                }
            except:
                continue
        
        # Default fallback
        return {'delimiter': ',', 'decimal': '.', 'encoding': 'utf-8'}
    
    @staticmethod
    def load_csv(filepath: Union[str, Path], 
                 delimiter: Optional[str] = None,
                 decimal: Optional[str] = None,
                 encoding: Optional[str] = None,
                 skip_rows: int = 0) -> Tuple[List[str], np.ndarray]:
        """
        Load CSV file with automatic format detection
        
        Args:
            filepath: Path to CSV file
            delimiter: Column delimiter (auto-detect if None)
            decimal: Decimal separator (auto-detect if None)
            encoding: File encoding (auto-detect if None)
            skip_rows: Number of rows to skip before header
            
        Returns:
            (header_names, data_array) tuple
        """
        filepath = Path(filepath)
        
        # Auto-detect format if not specified
        if delimiter is None or encoding is None:
            format_info = CSVImporter.detect_format(filepath)
            delimiter = delimiter or format_info['delimiter']
            decimal = decimal or format_info['decimal']
            encoding = encoding or format_info['encoding']
        
        # Read CSV
        with open(filepath, 'r', encoding=encoding) as f:
            # Skip initial rows
            for _ in range(skip_rows):
                f.readline()
            
            reader = csv.reader(f, delimiter=delimiter)
            
            # Read header
            header = next(reader)
            header = [col.strip() for col in header]
            
            # Read data
            rows = []
            for row in reader:
                if not row or all(cell.strip() == '' for cell in row):
                    continue  # Skip empty rows
                
                # Convert to float, handling decimal separator
                numeric_row = []
                for cell in row:
                    cell = cell.strip()
                    if decimal == ',':
                        cell = cell.replace(',', '.')
                    try:
                        numeric_row.append(float(cell))
                    except ValueError:
                        numeric_row.append(np.nan)
                
                rows.append(numeric_row)
        
        data = np.array(rows)
        
        return header, data


class RainDataImporter:
    """
    Import rain event data from CSV files
    
    Expected formats:
    1. Simple format: time, intensity
    2. Multi-zone format: time, zone1_intensity, zone2_intensity, ...
    3. Datetime format: datetime, intensity
    
    Standardizes to: times (hours or datetime), intensities (mm/hr)
    """
    
    @staticmethod
    def load_rain_csv(filepath: Union[str, Path],
                     time_column: str = 'time',
                     intensity_columns: Optional[List[str]] = None,
                     time_unit: str = 'hours',
                     datetime_format: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Load rain event data from CSV
        
        Args:
            filepath: Path to CSV file
            time_column: Name of time column
            intensity_columns: Names of intensity columns (auto-detect if None)
            time_unit: 'hours', 'days', 'seconds', or 'datetime'
            datetime_format: Format string for parsing datetime (e.g., '%Y-%m-%d %H:%M:%S')
            
        Returns:
            Dictionary with 'times' and intensity columns
        """
        header, data = CSVImporter.load_csv(filepath)
        
        # Map header names (case-insensitive)
        header_lower = [h.lower() for h in header]
        
        # Find time column
        time_idx = None
        for i, h in enumerate(header_lower):
            if time_column.lower() in h or h in ['time', 'date', 'datetime', 't']:
                time_idx = i
                break
        
        if time_idx is None:
            time_idx = 0  # Assume first column
        
        times = data[:, time_idx]
        
        # Parse datetime if needed
        if time_unit == 'datetime':
            # TODO: Implement datetime parsing from text
            raise NotImplementedError("Datetime parsing not yet implemented")
        
        # Find intensity columns
        if intensity_columns is None:
            # Auto-detect: all columns except time
            intensity_indices = [i for i in range(data.shape[1]) if i != time_idx]
            intensity_columns = [header[i] for i in intensity_indices]
        else:
            # Find specified columns
            intensity_indices = []
            for col_name in intensity_columns:
                for i, h in enumerate(header):
                    if col_name.lower() in h.lower():
                        intensity_indices.append(i)
                        break
        
        # Build result dictionary
        result = {'times': times}
        for i, col_idx in enumerate(intensity_indices):
            col_name = header[col_idx]
            result[col_name] = data[:, col_idx]
        
        print(f"✓ Loaded rain data from {filepath.name}")
        print(f"  Time range: {times[0]:.2f} to {times[-1]:.2f} {time_unit}")
        print(f"  Intensity columns: {list(result.keys())[1:]}")
        
        return result


class MaterialDataImporter:
    """
    Import material property curves from CSV files
    
    Expected formats:
    - Pressure head vs. water content: pressure_head, theta
    - Pressure head vs. relative permeability: pressure_head, kr
    - Full characteristic curves: pressure_head, theta, kr
    
    Standardizes to: pressure_head (m), theta (-), kr (-)
    """
    
    STANDARD_COLUMNS = {
        'pressure_head': ['pressure_head', 'pressure', 'head', 'h', 'p', 'psi'],
        'theta': ['theta', 'water_content', 'moisture', 'wc', 'volumetric_water_content'],
        'kr': ['kr', 'relative_permeability', 'k_rel', 'krel', 'perm']
    }
    
    @staticmethod
    def load_material_curve(filepath: Union[str, Path],
                           column_mapping: Optional[Dict[str, str]] = None) -> Dict[str, np.ndarray]:
        """
        Load material property curves from CSV
        
        Args:
            filepath: Path to CSV file
            column_mapping: Manual mapping of standard names to CSV columns
                          e.g., {'pressure_head': 'Psi', 'theta': 'Moisture'}
        
        Returns:
            Dictionary with standardized column names
        """
        header, data = CSVImporter.load_csv(filepath)
        
        # Map columns
        result = {}
        header_lower = [h.lower() for h in header]
        
        for std_name, possible_names in MaterialDataImporter.STANDARD_COLUMNS.items():
            # Check manual mapping first
            if column_mapping and std_name in column_mapping:
                search_name = column_mapping[std_name].lower()
                for i, h in enumerate(header_lower):
                    if search_name in h:
                        result[std_name] = data[:, i]
                        break
            else:
                # Auto-detect from possible names
                for i, h in enumerate(header_lower):
                    if any(pn in h for pn in possible_names):
                        result[std_name] = data[:, i]
                        break
        
        if result:
            print(f"✓ Loaded material curve from {filepath.name}")
            print(f"  Found columns: {list(result.keys())}")
        else:
            print(f"⚠️  No standard columns found in {filepath.name}")
            print(f"  Available columns: {header}")
        
        return result


# Convenience functions
def load_rain_data(filepath: Union[str, Path], **kwargs) -> Dict[str, np.ndarray]:
    """Convenience function to load rain data"""
    return RainDataImporter.load_rain_csv(filepath, **kwargs)


def load_material_data(filepath: Union[str, Path], **kwargs) -> Dict[str, np.ndarray]:
    """Convenience function to load material data"""
    return MaterialDataImporter.load_material_curve(filepath, **kwargs)


def smooth_data(times: np.ndarray, data: np.ndarray, window_hours: float = 1.0) -> np.ndarray:
    """
    Apply sliding window smoothing to noisy data
    
    Args:
        times: Time array in days
        data: Data array with potential NaN values
        window_hours: Window size in hours for smoothing (default: 1 hour)
    
    Returns:
        Smoothed data array
    """
    # Remove NaN values for smoothing
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return data
    
    times_valid = times[valid_mask]
    data_valid = data[valid_mask]
    
    # Convert window to days
    window_days = window_hours / 24.0
    
    # Apply sliding window average
    smoothed = np.zeros_like(data_valid)
    for i, t in enumerate(times_valid):
        # Find points within window
        window_mask = np.abs(times_valid - t) <= window_days / 2.0
        if np.any(window_mask):
            smoothed[i] = np.mean(data_valid[window_mask])
        else:
            smoothed[i] = data_valid[i]
    
    # Put smoothed data back into full array with NaNs
    result = np.full_like(data, np.nan)
    result[valid_mask] = smoothed
    
    return result


def load_and_align_data(csv_path: Union[str, Path], 
                        start_from: float, 
                        sim_duration_days: float, 
                        data_type: str = 'Data', 
                        ref_date: Optional[datetime] = None, 
                        offset: float = 0.0,
                        apply_smoothing: bool = False,
                        smoothing_window_hours: float = 1.0) -> Optional[Dict[str, np.ndarray]]:
    """
    Load data (COMSOL or measured) and align it with simulation time
    
    Args:
        csv_path: Path to CSV file
        start_from: Time (days) to map to simulation t=0
        sim_duration_days: Simulation duration in days
        data_type: 'COMSOL' or 'Measured' for logging
        ref_date: Reference datetime for t=0 (required if CSV has datetime strings)
        offset: Vertical offset to add to all data values (e.g., 0.6 for 60cm)
        apply_smoothing: Whether to apply smoothing (auto-enabled for Measured data)
        smoothing_window_hours: Window size in hours for smoothing
    
    Returns:
        Dictionary with aligned time and data arrays, or None if error
    """
    try:
        # Load raw data
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline()
            delimiter = ';' if ';' in first_line else ','
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = [{k.strip(): v.strip() for k, v in row.items()} 
                    for row in reader]
        
        # Find time/date column
        time_col = None
        for k in rows[0].keys():
            if any(term in k.lower() for term in ['time', 'date', 'jour', 'day']):
                time_col = k
                break
        
        if not time_col:
            print(f"⚠️  No time/date column found in {csv_path}")
            return None
        
        # Try to parse times - could be floats or datetime strings
        times_raw = []
        is_datetime_col = False
        
        for row in rows:
            time_str = row[time_col]
            try:
                # Try parsing as float first
                times_raw.append(float(time_str.replace(',', '.')))
            except ValueError:
                # Must be a datetime string
                is_datetime_col = True
                break
        
        # If datetime strings, need to parse them
        if is_datetime_col:
            if ref_date is None:
                print(f"⚠️  {data_type} data has datetime strings but no ref_date provided")
                return None
            
            times_raw = []
            for row in rows:
                date_str = row[time_col]
                # Try common European date formats
                for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        # Convert to days from ref_date
                        days_diff = (dt - ref_date).total_seconds() / 86400.0
                        times_raw.append(days_diff)
                        break
                    except ValueError:
                        continue
                else:
                    print(f"⚠️  Could not parse date: {date_str}")
                    return None
            
            times_raw = np.array(times_raw)
            print(f"✓ Parsed {len(times_raw)} datetime strings from {data_type} data")
        else:
            times_raw = np.array(times_raw)
        
        # Parse data columns - look for LTC or Level columns
        data_raw = {}
        column_mapping = {}  # Map original column names to standardized LTC names
        
        for col in rows[0].keys():
            if col.startswith('LTC') or 'ltc' in col.lower() or 'level' in col.lower():
                try:
                    # Parse values, replacing empty strings with NaN
                    values = []
                    for row in rows:
                        val_str = row[col].replace(',', '.').strip()
                        if val_str == '' or val_str == 'nan':
                            values.append(np.nan)
                        else:
                            values.append(float(val_str))
                    
                    data_raw[col] = np.array(values)
                    
                    # Standardize column names: "Level 101" -> "LTC 101", "Level 102" -> "LTC 102", etc.
                    if 'level' in col.lower():
                        # Extract number from "Level 101 (m)" or "Level 101"
                        import re
                        match = re.search(r'(\d+)', col)
                        if match:
                            num = match.group(1)
                            standardized_name = f"LTC {num}"
                            column_mapping[col] = standardized_name
                    else:
                        column_mapping[col] = col
                        
                except ValueError as e:
                    print(f"⚠️  Could not parse column {col}: {e}")
                    continue
        
        if not data_raw:
            print(f"⚠️  No LTC/Level data columns found in {csv_path}")
            return None
        
        # Apply column name mapping
        data_raw = {column_mapping[k]: v for k, v in data_raw.items()}
        
        print(f"✓ Loaded {data_type} data: {list(data_raw.keys())}")
        print(f"  Raw time range: {times_raw[0]:.2f} to {times_raw[-1]:.2f} days ({len(times_raw)} points)")
        
        # Filter, shift, and clip
        mask_start = times_raw >= start_from
        if not np.any(mask_start):
            print(f"⚠️  No {data_type} data found at or after start_from={start_from:.2f} days")
            return None
        
        times_filtered = times_raw[mask_start]
        data_filtered = {k: v[mask_start] for k, v in data_raw.items()}
        times_shifted = times_filtered - start_from
        
        mask_clip = times_shifted <= sim_duration_days
        if not np.any(mask_clip):
            print(f"⚠️  No {data_type} data in simulation range after shifting")
            return None
        
        times_aligned = times_shifted[mask_clip]
        data_aligned = {k: v[mask_clip] for k, v in data_filtered.items()}
        
        # Apply smoothing if requested or if measured data
        if apply_smoothing or data_type == 'Measured':
            print(f"  Applying sliding window smoothing ({smoothing_window_hours}-hour window)...")
            data_aligned = {k: smooth_data(times_aligned, v, window_hours=smoothing_window_hours) 
                           for k, v in data_aligned.items()}
        
        # Apply vertical offset if specified
        if offset != 0.0:
            print(f"  Applying vertical offset: {offset:+.2f} m to {data_type} data")
            data_aligned = {k: v + offset for k, v in data_aligned.items()}
        
        # Create final aligned dictionary
        aligned_data = {'times': times_aligned}
        aligned_data.update(data_aligned)
        
        print(f"  Aligned {data_type} data: {len(times_aligned)} points in range [0, {sim_duration_days:.2f}] days")
        
        return aligned_data
        
    except Exception as e:
        print(f"⚠️  Error loading {data_type} data: {e}")
        return None
