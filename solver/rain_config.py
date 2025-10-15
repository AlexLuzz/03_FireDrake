# physics/rain_config.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import csv

@dataclass
class RainZone:
    """
    Define a spatial zone where rain is applied
    """
    x_min: float  # Start of zone (m)
    x_max: float  # End of zone (m)
    multiplier: float = 1.0  # Intensity multiplier relative to base intensity
    name: str = "rain_zone"  # Optional name for zone
    
    def contains_x(self, x: float) -> bool:
        """Check if x coordinate is in this zone"""
        return self.x_min <= x <= self.x_max


@dataclass
class RainEvent:
    """
    Define a rain event with timing and intensity
    """
    start_time: float  # Start time (hours)
    end_time: float    # End time (hours)
    intensity: float   # Rain intensity (mm/hour)
    zones: List[RainZone] = None  # Spatial zones (None = entire domain)
    name: str = "rain_event"  # Optional name for event
    
    def __post_init__(self):
        if self.zones is None:
            # Default: apply to entire domain
            self.zones = [RainZone(x_min=-np.inf, x_max=np.inf, multiplier=1.0, name="full_domain")]
    
    @property
    def duration_hours(self) -> float:
        """Duration in hours"""
        return self.end_time - self.start_time
    
    @property
    def total_depth_mm(self) -> float:
        """Total rain depth in mm"""
        return self.intensity * self.duration_hours
    
    def is_active(self, t_hours: float) -> bool:
        """Check if event is active at given time"""
        return self.start_time <= t_hours <= self.end_time
    
    def get_flux_at_x(self, x: float, t_hours: float) -> float:
        """
        Get rain flux at position x and time t
        
        Returns:
            Flux in m/s (negative = into domain)
        """
        if not self.is_active(t_hours):
            return 0.0
        
        # Check which zone x is in
        for zone in self.zones:
            if zone.contains_x(x):
                # Convert mm/hour to m/s: mm/hour * (1m/1000mm) * (1hour/3600s)
                flux_m_s = self.intensity * zone.multiplier / 3600000.0
                return -flux_m_s  # Negative = into domain
        
        return 0.0


class RainScenario:
    """
    Manage multiple rain events over a simulation
    """
    
    def __init__(self, events: List[RainEvent]):
        """
        Initialize rain scenario
        
        Args:
            events: List of RainEvent objects
        """
        self.events = sorted(events, key=lambda e: e.start_time)
    
    @classmethod
    def single_event(cls, start_hours: float, end_hours: float, 
                     intensity_mm_hr: float, zones: List[RainZone] = None):
        """
        Create scenario with single rain event
        
        Example:
            # Uniform 20mm/hour rain from 3h to 5h
            scenario = RainScenario.single_event(3.0, 5.0, 20.0)
        """
        event = RainEvent(start_hours, end_hours, intensity_mm_hr, zones)
        return cls([event])
    
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
    
    def get_flux_at_x(self, x: float, t_hours: float) -> float:
        """
        Get total rain flux at position x and time t
        
        Returns:
            Flux in m/s (negative = into domain)
        """
        total_flux = 0.0
        for event in self.events:
            total_flux += event.get_flux_at_x(x, t_hours)
        return total_flux
    
    def print_summary(self):
        """Print summary of rain events"""
        print("Rain Scenario Summary:")
        print(f"  Total events: {len(self.events)}")
        
        total_duration = sum(e.duration_hours for e in self.events)
        avg_intensity = np.mean([e.intensity for e in self.events])
        total_depth = sum(e.total_depth_mm for e in self.events)
        
        print(f"  Total rain duration: {total_duration:.1f} hours")
        print(f"  Average intensity: {avg_intensity:.1f} mm/hour")
        print(f"  Total rain depth: {total_depth:.1f} mm")
        print()
        
        for i, event in enumerate(self.events, 1):
            print(f"  Event {i}: {event.name}")
            print(f"    Time: {event.start_time:.1f}h to {event.end_time:.1f}h ({event.duration_hours:.1f}h)")
            print(f"    Intensity: {event.intensity:.1f} mm/hour")
            print(f"    Total depth: {event.total_depth_mm:.1f} mm")
            print(f"    Zones: {len(event.zones)}")
            for zone in event.zones:
                if zone.multiplier != 1.0:
                    print(f"      - {zone.name}: x∈[{zone.x_min:.1f}, {zone.x_max:.1f}]m, "
                          f"multiplier={zone.multiplier:.1f}x")
            print()


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
    
    # Example 4: Create example CSV
    print("\n4. Creating example CSV file")
    print("-" * 40)
    
    # Create example CSV without pandas
    with open('rain_events_example.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_hours', 'intensity_mm_hr'])
        writer.writerow([0.0, 0.0])
        writer.writerow([3.0, 0.0])
        writer.writerow([3.0, 20.0])
        writer.writerow([5.0, 20.0])
        writer.writerow([5.0, 0.0])
        writer.writerow([10.0, 0.0])
        writer.writerow([10.0, 40.0])
        writer.writerow([12.0, 40.0])
        writer.writerow([12.0, 0.0])
        writer.writerow([24.0, 0.0])
    
    print("Created rain_events_example.csv")
    
    # Display contents
    with open('rain_events_example.csv', 'r') as f:
        print(f.read())
    
    # Load from CSV
    scenario4 = RainScenario.from_csv('rain_events_example.csv')
    scenario4.print_summary()
    
    print("=" * 60)