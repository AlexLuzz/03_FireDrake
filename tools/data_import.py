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
