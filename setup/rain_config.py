# physics/rain_config.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RainZone:
    """Define a spatial zone where rain is applied"""
    x_min: float  # Start of zone (m)
    x_max: float  # End of zone (m)
    multiplier: float = 1.0  # Intensity multiplier relative to base intensity
    name: str = "rain_zone"  # Optional name for zone


@dataclass
class RainEvent:
    """Define a rain event with timing and intensity"""
    start_time: float  # Start time (hours)
    end_time: float    # End time (hours)
    intensity: float   # Rain intensity (mm/hour)
    zones: Optional[List[RainZone]] = None  # Spatial zones (None = entire domain)
    name: str = "rain_event"  # Optional name for event


class RainScenario:
    """Manage multiple rain events over a simulation"""
    
    def __init__(self, events: List[RainEvent]):
        self.events = sorted(events, key=lambda e: e.start_time)
    
    @classmethod
    def single_event(cls, start_hours: float, end_hours: float, 
                     intensity_mm_hr: float, zones: Optional[List[RainZone]] = None):
        """Create scenario with single rain event"""
        event = RainEvent(start_hours, end_hours, intensity_mm_hr, zones)
        return cls([event])
    
    def get_flux_expression(self, t_hours: float, mesh):
        """
        Get UFL expression for spatially-varying rain flux at given time
        
        Returns:
            UFL expression for rain flux (m/s, negative = into domain)
        """
        from firedrake import SpatialCoordinate, Constant, conditional, And
        
        # Remove finished events
        while self.events and t_hours > self.events[0].end_time:
            self.events.pop(0)
        
        # Check if we have an active event
        if self.events and self.events[0].start_time <= t_hours <= self.events[0].end_time:
            event = self.events[0]
            
            # If no zones specified, apply uniformly everywhere
            if event.zones is None:
                flux = event.intensity / 3600000.0  # mm/hr to m/s
                return Constant(-flux)
            
            # Build UFL expression for zones
            coords = SpatialCoordinate(mesh)
            x = coords[0]
            flux_expr = Constant(0.0)
            for zone in event.zones:
                zone_flux = event.intensity * zone.multiplier / 3600000.0
                flux_expr = conditional(
                    And(x >= zone.x_min, x <= zone.x_max),
                    Constant(-zone_flux),
                    flux_expr
                )
            return flux_expr
        
        return Constant(0.0)
        
    @classmethod
    def from_csv(cls, csv_path: str, 
             start_from: float = 0.0,
             time_col: str = "time_hours", 
             rain_col: str = "intensity_mm_hr",
             time_unit: str = "hours",
             rain_unit: str = "mm/hour",
             zones: List[RainZone] = None,
             delimiter: str = None):
        """
        Load rain events from CSV file
        
        CSV format example:
            time_hours, intensity_mm_hr
            0.0, 0.0
            3.0, 0.0
            3.0, 20.0
            5.0, 20.0
            5.0, 0.0
        
        Args:
            csv_path: Path to CSV file
            start_from: Start simulation from this time value (in time_unit)
                        Example: start_from=100 with time_unit="days" → starts at day 100
            time_col: Name of time column in CSV
            rain_col: Name of rain column in CSV
            time_unit: "hours" or "days"
            rain_unit: "mm/hour" or "mm/day" (rain intensity)
            zones: Optional spatial zones (default: entire domain)
            delimiter: CSV delimiter (default: auto-detect)
        
        Returns:
            RainScenario object with time shifted to start at t=0
        
        Examples:
            # Standard hourly data
            scenario = RainScenario.from_csv("rain.csv")
            
            # Daily data, start from day 100
            scenario = RainScenario.from_csv(
                "rain.csv",
                start_from=100.0,
                time_col="day",
                rain_col="rain_mm_day",
                time_unit="days",
                rain_unit="mm/day"
            )
        """
        import csv
        
        # Auto-detect delimiter
        if delimiter is None:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline()
                delimiter = ';' if ';' in first_line else ','
        
        # Read CSV
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = [{k.strip(): v for k, v in row.items()} for row in reader]
        
        if not rows:
            raise ValueError(f"Empty CSV file: {csv_path}")
        
        # Helper: parse float with comma or dot decimal separator
        def parse_float(s):
            return float(s.strip().replace(',', '.'))
        
        # Extract data with error handling
        try:
            times_raw = np.array([parse_float(row[time_col]) for row in rows])
            rain_raw = np.array([parse_float(row[rain_col]) for row in rows])
        except KeyError as e:
            available = list(rows[0].keys())
            raise KeyError(f"Column {e} not found. Available: {available}")
        
        # Convert time to hours
        time_unit = time_unit.lower()
        if time_unit in ["hours", "hour", "h"]:
            times_hours = times_raw
        elif time_unit in ["days", "day", "d"]:
            times_hours = times_raw * 24.0
            start_from_hours = start_from * 24.0
        else:
            raise ValueError(f"Invalid time_unit '{time_unit}'. Use 'hours' or 'days'")
        
        # Filter data: only keep times >= start_from
        if time_unit in ["days", "day", "d"]:
            start_from_hours = start_from * 24.0
        else:
            start_from_hours = start_from
        
        mask = times_hours >= start_from_hours
        times_hours = times_hours[mask]
        rain_raw = rain_raw[mask]
        
        if len(times_hours) == 0:
            raise ValueError(f"No data after start_from={start_from} {time_unit}")
        
        # Shift time to start at t=0
        times_hours = times_hours - times_hours[0]
        
        # Convert rain to mm/hour
        rain_unit = rain_unit.lower().replace('/', '')  # "mm/hour" → "mmhour"
        if rain_unit in ["mmhour", "mm_hour", "mmh"]:
            intensities = rain_raw  # Already mm/hour
        elif rain_unit in ["mmday", "mm_day", "mmd"]:
            intensities = rain_raw / 24.0  # mm/day → mm/hour
        else:
            raise ValueError(f"Invalid rain_unit '{rain_unit}'. Use 'mm/hour' or 'mm/day'")
        
        # Build rain events (periods with constant non-zero intensity)
        events = []
        in_event = False
        event_start = None
        current_intensity = 0.0
        
        for i in range(len(times_hours)):
            t = times_hours[i]
            intensity = intensities[i]
            
            if intensity > 0 and not in_event:
                # Start new event
                in_event = True
                event_start = t
                current_intensity = intensity
            
            elif intensity <= 0 and in_event:
                # End event
                events.append(RainEvent(
                    start_time=event_start,
                    end_time=t,
                    intensity=current_intensity,
                    zones=zones,
                    name=f"event_{len(events)+1}"
                ))
                in_event = False
            
            elif in_event and abs(intensity - current_intensity) > 1e-6:
                # Intensity changed - close old event, start new
                events.append(RainEvent(
                    start_time=event_start,
                    end_time=t,
                    intensity=current_intensity,
                    zones=zones,
                    name=f"event_{len(events)+1}"
                ))
                event_start = t
                current_intensity = intensity
        
        # Close final event if still active
        if in_event:
            events.append(RainEvent(
                start_time=event_start,
                end_time=times_hours[-1],
                intensity=current_intensity,
                zones=zones,
                name=f"event_{len(events)+1}"
            ))
        
        print(f"Loaded rain scenario from {csv_path}")
        print(f"  Start time: {start_from} {time_unit} (shifted to t=0)")
        print(f"  Duration: {times_hours[-1]/24:.1f} days")
        print(f"  Events found: {len(events)}")
        
        return cls(events)
    
    @classmethod
    def from_datetime_csv(cls, csv_path: str,
                          time_converter,
                          datetime_column: str = 'Date',
                          rain_column: str = 'Pluie',
                          rain_unit: str = 'mm/day',
                          zones: List[RainZone] = None,
                          delimiter: Optional[str] = None,
                          start_datetime: Optional['datetime'] = None,
                          end_datetime: Optional['datetime'] = None):
        """
        Load rain events from CSV with datetime column
        
        Designed for files like BB_METEO.csv:
            Date/Heure;Pluie tot. (mm);Neige tot. (cm)
            2024-01-01;0;0
            2024-01-02;1.5;0
            2024-01-03;0;0
        
        Args:
            csv_path: Path to CSV file
            time_converter: TimeConverter instance (defines simulation start)
            datetime_column: Name of datetime column
            rain_column: Name of rain column
            rain_unit: "mm/day" or "mm/hour"
            zones: Optional spatial zones
            delimiter: CSV delimiter (auto-detect if None)
            start_datetime: Only load data >= this datetime (default: use all data)
            end_datetime: Only load data <= this datetime (default: use all data)
        
        Returns:
            RainScenario with events starting at t=0
        
        Example:
            from tools import TimeConverter
            from setup import RainScenario
            from datetime import datetime
            
            converter = TimeConverter(start_datetime=datetime(2024, 8, 15))
            rain = RainScenario.from_datetime_csv(
                'BB_METEO.csv',
                time_converter=converter,
                start_datetime=datetime(2024, 8, 15),
                end_datetime=datetime(2024, 8, 21)
            )
            
            converter = TimeConverter(start_datetime=datetime(2024, 1, 1))
            rain = RainScenario.from_datetime_csv(
                'BB_METEO.csv',
                time_converter=converter,
                datetime_column='Date/Heure',
                rain_column='Pluie tot. (mm)',
                rain_unit='mm/day'
        """
        # Load data using TimeConverter with optional datetime filtering
        data = time_converter.load_datetime_csv(
            csv_path,
            datetime_column=datetime_column,
            value_columns=[rain_column],
            delimiter=delimiter,
            start_datetime=start_datetime,
            end_datetime=end_datetime
        )
        
        times_seconds = data['times']
        # Find the actual rain column name (partial match)
        rain_col_key = [k for k in data.keys() if rain_column.lower() in k.lower()][0]
        rain_values = data[rain_col_key]
        
        # Convert to hours
        times_hours = times_seconds / 3600.0
        
        # Convert rain to mm/hour
        rain_unit_lower = rain_unit.lower().replace('/', '')
        if 'day' in rain_unit_lower:
            intensities = rain_values / 24.0  # mm/day → mm/hour
        elif 'hour' in rain_unit_lower:
            intensities = rain_values  # Already mm/hour
        else:
            raise ValueError(f"Invalid rain_unit: {rain_unit}. Use 'mm/day' or 'mm/hour'")
        
        # Build rain events (constant intensity periods)
        events = []
        in_event = False
        event_start = None
        current_intensity = 0.0
        
        for i in range(len(times_hours)):
            t = times_hours[i]
            intensity = intensities[i]
            
            if intensity > 0 and not in_event:
                # Start new event
                in_event = True
                event_start = t
                current_intensity = intensity
            
            elif intensity <= 0 and in_event:
                # End event
                events.append(RainEvent(
                    start_time=event_start,
                    end_time=t,
                    intensity=current_intensity,
                    zones=zones,
                    name=f"event_{len(events)+1}"
                ))
                in_event = False
            
            elif in_event and abs(intensity - current_intensity) > 1e-6:
                # Intensity changed
                events.append(RainEvent(
                    start_time=event_start,
                    end_time=t,
                    intensity=current_intensity,
                    zones=zones,
                    name=f"event_{len(events)+1}"
                ))
                event_start = t
                current_intensity = intensity
        
        # Close final event
        if in_event:
            events.append(RainEvent(
                start_time=event_start,
                end_time=times_hours[-1],
                intensity=current_intensity,
                zones=zones,
                name=f"event_{len(events)+1}"
            ))
        
        print(f"  Created {len(events)} rain event(s)")
        
        return cls(events)
    
# ==========================================
# EXAMPLES
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAIN CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Simple uniform rain
    print("\n1. Simple uniform rain event")
    print("-" * 40)
    scenario1 = RainScenario.single_event(
        start_hours=3.0,
        end_hours=5.0,
        intensity_mm_hr=20.0
    )
    scenario1.print_summary()
    
    # Example 2: Rain with different zones
    print("\n2. Rain with spatial zones (green infrastructure gets 6x more)")
    print("-" * 40)
    zones = [
        RainZone(x_min=0.0, x_max=9.0, multiplier=1.0, name="till_area"),
        RainZone(x_min=9.0, x_max=11.0, multiplier=6.0, name="green_infrastructure"),
        RainZone(x_min=11.0, x_max=20.0, multiplier=1.0, name="till_area_right")
    ]
    scenario2 = RainScenario.single_event(
        start_hours=3.0,
        end_hours=5.0,
        intensity_mm_hr=20.0,
        zones=zones
    )
    scenario2.print_summary()
    
    # Example 3: Multiple events
    print("\n3. Multiple rain events")
    print("-" * 40)
    events = [
        RainEvent(start_time=3.0, end_time=5.0, intensity=20.0, name="morning_rain"),
        RainEvent(start_time=10.0, end_time=12.0, intensity=40.0, name="afternoon_storm"),
        RainEvent(start_time=20.0, end_time=21.0, intensity=10.0, name="evening_drizzle")
    ]
    scenario3 = RainScenario(events)
    scenario3.print_summary()
    
    