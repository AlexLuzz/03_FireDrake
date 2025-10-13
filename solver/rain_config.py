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
                 time_col: str = "time_hours", 
                 rain_col: str = "intensity_mm_hr",
                 time_unit: str = "hours",
                 rain_type: str = "rate",
                 zones: List[RainZone] = None):
        """
        Load rain events from CSV file with flexible column names and units
        
        CSV format example:
            time_hours, intensity_mm_hr
            0.0, 0.0
            3.0, 0.0
            3.0, 20.0
            5.0, 20.0
            5.0, 0.0
        
        Args:
            csv_path: Path to CSV file
            time_col: Name of time column in CSV
            rain_col: Name of rain column in CSV (always in mm)
            time_unit: Unit of time column - "hours" or "days"
            rain_type: Type of rain data - "rate" (mm/hour) or "depth" (cumulative mm)
            zones: Optional spatial zones (default: entire domain)
        
        Returns:
            RainScenario object
        
        Examples:
            # Default: time in hours, rain rate in mm/hr
            scenario = RainScenario.from_csv("rain.csv")
            
            # Time in days, rain rate in mm/day
            scenario = RainScenario.from_csv(
                "rain.csv", 
                time_col="Day", 
                rain_col="Rainfall_mm_day",
                time_unit="days",
                rain_type="rate"
            )
            
            # Time in hours, cumulative rain depth in mm
            scenario = RainScenario.from_csv(
                "rain.csv",
                time_col="Time_h",
                rain_col="Cumulative_mm",
                time_unit="hours",
                rain_type="depth"
            )
        """
        # Read CSV using standard library
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Extract raw data
        times_raw = np.array([float(row[time_col]) for row in rows])
        rain_raw = np.array([float(row[rain_col]) for row in rows])
        
        # Convert time to hours
        if time_unit.lower() in ["hours", "hour", "h"]:
            times = times_raw
        elif time_unit.lower() in ["days", "day", "d"]:
            times = times_raw * 24.0
        else:
            raise ValueError(f"Unknown time unit '{time_unit}'. Supported: 'hours', 'days'")
        
        # Convert rain data to intensity (mm/hr)
        if rain_type.lower() == "rate":
            # Already a rate, just need to convert from mm/time_unit to mm/hr
            if time_unit.lower() in ["hours", "hour", "h"]:
                intensities = rain_raw  # Already mm/hr
            elif time_unit.lower() in ["days", "day", "d"]:
                intensities = rain_raw / 24.0  # mm/day to mm/hr
        elif rain_type.lower() == "depth":
            # Cumulative depth - calculate rate from differences
            intensities = np.zeros_like(rain_raw)
            for i in range(1, len(rain_raw)):
                dt = times[i] - times[i-1]  # in hours
                if dt > 0:
                    depth_increment = rain_raw[i] - rain_raw[i-1]
                    intensities[i] = depth_increment / dt  # mm/hr
            intensities[0] = 0.0  # First point has no rate
        else:
            raise ValueError(f"Unknown rain_type '{rain_type}'. Supported: 'rate', 'depth'")
        
        # Find rain events (periods with non-zero intensity)
        events = []
        in_event = False
        event_start = None
        current_intensity = 0.0
        
        for i in range(len(times)):
            t = times[i]
            intensity = intensities[i]
            
            if intensity > 0 and not in_event:
                # Start of new event
                in_event = True
                event_start = t
                current_intensity = intensity
            
            elif intensity <= 0 and in_event:
                # End of event
                events.append(RainEvent(
                    start_time=event_start,
                    end_time=t,
                    intensity=current_intensity,
                    zones=zones,
                    name=f"event_{len(events)+1}"
                ))
                in_event = False
            
            elif in_event and intensity != current_intensity:
                # Intensity changed - end old event, start new one
                events.append(RainEvent(
                    start_time=event_start,
                    end_time=t,
                    intensity=current_intensity,
                    zones=zones,
                    name=f"event_{len(events)+1}"
                ))
                event_start = t
                current_intensity = intensity
        
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