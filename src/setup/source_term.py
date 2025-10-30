"""
Simple source/sink term manager for FEM simulations
"""
from dataclasses import dataclass
from typing import List, Optional, Callable
from datetime import datetime
import pandas as pd
from firedrake import Constant, SpatialCoordinate, conditional, And
from ..tools.csv_loader import CSVLoader
from ..tools.time_converter import TimeConverter
from ..tools.fetch_rainfall_meteostat import fetch_and_aggregate_weather

@dataclass
class Zone:
    """
    Spatial zone where sources/sinks are applied.
    
    Attributes:
        name: Zone identifier (e.g., "road", "left_half", "domain")
        x_min, x_max: Horizontal extent [m]
        y_min, y_max: Vertical extent [m] (optional for 1D problems)
                      ⚠️ IMPORTANT: If using y-boundaries in 2D, be careful of the surface you 
                      are defining as the entry in the equations: 
                      - for vertical boundaries: left=1 (x=0), right=2 (x=Lx)
        multiplier: Spatial intensity factor (default=1.0)
                   Example: multiplier=0.5 → half the base rate in this zone
    
    Examples:
        # 1D zone (rainfall across entire domain width)
        Zone("domain", x_min=0, x_max=100)
        
        # 2D zone (rainfall only on left half)
        Zone("left_half", x_min=0, x_max=50, y_min=0, y_max=5)
        # Zone with reduced intensity (e.g., curb vs road)
        Zone("curb", x_min=10, x_max=15, multiplier=0.3)
    """
    name: str
    x_min: float
    x_max: float
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    multiplier: float = 1.0
    
    def contains(self, x: float, y: Optional[float] = None) -> bool:
        in_x = self.x_min <= x <= self.x_max
        if self.y_min is not None and y is not None:
            return in_x and (self.y_min <= y <= self.y_max)
        return in_x
    
    def ufl_condition(self, coords):
        x = coords[0]
        cond = And(x >= self.x_min, x <= self.x_max)
        if self.y_min is not None:
            y = coords[1]
            cond = And(cond, And(y >= self.y_min, y <= self.y_max))
        return cond


@dataclass
class Event:
    """
    Represents a source/sink event with spatial and temporal extent.
    
    Attributes:
        name: Event identifier
        start: Start time (seconds since simulation start)
        end: End time (seconds since simulation start)
        rate: Intensity in original units (before rate_conv)
        zones: List of spatial zones where event applies
        
        profile: Optional temporal profile function τ ∈ [0,1] → multiplier
                Allows time-varying intensity within the event duration.
                Example: lambda tau: 4*tau*(1-tau)  # Parabolic: 0→peak→0
                Example: lambda tau: tau            # Linear ramp-up
                Example: lambda tau: np.sin(π*tau)  # Sinusoidal
                If None, intensity is constant during event.
        
        start_dt, end_dt: Optional datetime for logging/debugging
                 Example: datetime(2024, 1, 15, 10, 0)
                 Useful for: "Event 'storm_1' occurred on Jan 15, 2024 at 10:00"
                 Not used in calculations, purely metadata.
    
    Examples:
        # Simple constant event (no profile, no datetimes)
        Event("rain_constant", start=3600, end=7200, rate=10, zones=[zone])
        
        # Storm with intensity profile (builds up, peaks, weakens)
        Event("storm", start=0, end=21600, rate=20, zones=[zone],
              profile=lambda tau: 4*tau*(1-tau))
        
        # Event with datetime metadata for logging
        Event("rain_jan15", start=36000, end=57600, rate=15, zones=[zone],
              start_dt=datetime(2024, 1, 15, 10, 0),
              end_dt=datetime(2024, 1, 15, 16, 0))
        
        # Chemical application with decay profile
        Event("chloride", start=0, end=7200, rate=100, zones=[road_zone],
              profile=lambda tau: np.exp(-2*tau),  # Exponential decay
              start_dt=datetime(2024, 2, 1, 6, 0))
    """
    name: str
    start: float
    end: float
    rate: float
    zones: List[Zone]
    profile: Optional[Callable] = None  # Optional time-varying profile: f(t_normalized) -> multiplier
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    
    def is_active(self, t: float) -> bool:
        return self.start <= t < self.end
    
    def get_rate(self, t: float) -> float:
        """
        Get rate at time t.
        
        Returns constant rate (self.rate) throughout event duration.
        Profile feature available but not used by rainfall_scenario.
        """
        if not self.is_active(t):
            return 0.0
        
        # For rainfall_scenario: profile is always None, so rate is constant
        if self.profile is None:
            return self.rate
        
        # Optional time-varying profile (not used in rainfall_scenario)
        t_norm = (t - self.start) / (self.end - self.start)
        return self.rate * self.profile(t_norm)


class SourceScenario:
    """
    Manages spatial-temporal source/sink distributions.
    
    IMPORTANT: Internally, all times are stored and used in SECONDS.
    The time_unit parameter only affects how add_event() interprets its inputs.
    """
    
    def __init__(self, time_unit: str = "hours", rate_unit_conversion: float = 1.0):
        """
        Initialize scenario manager.
        
        Args:
            time_unit: Unit for time inputs in add_event() (NOT internal storage)
                      Options: "s", "min", "h", "d"/"day"
            rate_conv: Conversion factor for rates (e.g., mm/h → m/s = 1/3600000)
        """
        self.zones: List[Zone] = []
        self.events: List[Event] = []
        self.time_unit = time_unit
        self.rate_unit_conversion = rate_unit_conversion
        self._units = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}
    
    def _to_seconds(self, value: float) -> float:
        return value * self._units[self.time_unit]
    
    def add_zone(self, name: str, x_min: float, x_max: float, 
                 y_min: float = None, y_max: float = None, multiplier: float = 1.0) -> Zone:
        zone = Zone(name, x_min, x_max, y_min, y_max, multiplier)
        self.zones.append(zone)
        return zone
    
    def get_zone(self, name: str) -> Optional[Zone]:
        return next((z for z in self.zones if z.name == name), None)
    
    def add_event(self, name: str, start: float, end: float, rate: float,
                  zones, profile: Callable = None, 
                  start_datetime: datetime = None, end_datetime: datetime = None):
        """Add event with optional datetime tracking"""
        if not isinstance(zones, list):
            zones = [zones]
        
        zone_objs = []
        for z in zones:
            if isinstance(z, str):
                zone_obj = self.get_zone(z)
                if not zone_obj:
                    raise ValueError(f"Zone '{z}' not found")
                zone_objs.append(zone_obj)
            else:
                zone_objs.append(z)
        
        event = Event(name, self._to_seconds(start), self._to_seconds(end), 
                     rate, zone_objs, profile, start_datetime, end_datetime)
        self.events.append(event)
        return event
    
    def get_flux(self, x: float, y: float, t: float) -> float:
        """Get flux at point (x, y) and time t (seconds)"""
        total = 0.0
        for event in self.events:
            if event.is_active(t):
                rate = event.get_rate(t)
                for zone in event.zones:
                    if zone.contains(x, y):
                        total += rate * zone.multiplier
        return total
    
    def get_flux_expression(self, t: float, mesh):
        """Get UFL expression at time t (seconds) - USE THIS IN WEAK FORMS"""
        coords = SpatialCoordinate(mesh)
        flux = Constant(0.0)
        
        for event in self.events:
            if not event.is_active(t):
                continue
            
            rate = event.get_rate(t)
            
            for zone in event.zones:
                zone_flux = rate * zone.multiplier * self.rate_unit_conversion
                flux = conditional(
                    zone.ufl_condition(coords),
                    Constant(zone_flux),
                    flux
                )
        
        return flux


def rainfall_scenario(domain_length: float = None, events: list = None, time_unit: str = "days",
                     csv_path: str = None, time_col: str = "Date", rain_col: str = "Pluie tot. (mm)",
                     from_date: datetime = None, to_date: datetime = None, 
                     rain_unit: str = "mm/day", zones: list = None,
                     meteostat_station: str = None, meteostat_agg_hours: int = 6):
    """
    Create rainfall scenario from events, CSV, or Meteostat.
    
    Args:
        domain_length: Domain length (m), creates single zone [0, domain_length]
        events: List of event dicts with 'start', 'end', 'rate' keys
        csv_path: Path to CSV file with rainfall data
        time_col: Column name for time data
        rain_col: Column name for rainfall data
                  - For datetime: accumulated mm over each time step (auto-computed to mm/hr)
                  - For numeric time: intensity in rain_unit
        rain_unit: Unit ONLY for numeric time (e.g., "mm/hour" or "mm/day")
                   Not needed for datetime (auto-computed from time differences)
        zones: List of zone dicts with 'name', 'x_min', 'x_max', 'multiplier'
        meteostat_station: Station ID (e.g., 'SOK6B' for Montreal)
        meteostat_agg_hours: Aggregation period in hours (default: 6)
    
    Examples:
        # From Meteostat (datetime, auto-computed)
        scenario = rainfall_scenario(
            domain_length=100,
            meteostat_station='SOK6B',
            from_date=datetime(2024, 5, 1),
            to_date=datetime(2024, 5, 15),
            meteostat_agg_hours=6
        )
        
        # From CSV with datetime (accumulated mm → auto-computed to mm/hr)
        scenario = rainfall_scenario(
            domain_length=100,
            csv_path="rain.csv",
            time_col='Date',
            rain_col='Precipitation (mm)'  # Accumulated over time step
        )
        
        # From CSV with numeric time (must specify rain_unit)
        scenario = rainfall_scenario(
            domain_length=100,
            csv_path="rain_numeric.csv",
            time_col='time_hours',
            rain_col='intensity',
            rain_unit='mm/hour'  # Required for numeric time!
        )
    """
    # Create scenario with mm/hr → m/s conversion
    scenario = SourceScenario(time_unit, rate_unit_conversion=1.0/3600000.0)
    
    # Setup zones
    if zones:
        for z in zones:
            scenario.add_zone(z['name'], z['x_min'], z['x_max'], 
                            y_min=z.get('y_min'), y_max=z.get('y_max'),
                            multiplier=z.get('multiplier', 1.0))
    elif domain_length:
        scenario.add_zone("domain", 0, domain_length)
    else:
        # If no zones specified and using CSV, we need domain_length
        if csv_path or meteostat_station:
            raise ValueError("Must specify either 'zones' or 'domain_length'")
    
    # Fetch from Meteostat if requested
    if meteostat_station:
        if not from_date or not to_date:
            raise ValueError("Must specify from_date and to_date for Meteostat")
        
        # Use the existing fetch_and_aggregate_weather function
        try:
            precip_df, _ = fetch_and_aggregate_weather(
                start_date=from_date,
                end_date=to_date,
                station=meteostat_station,
                precip_step=meteostat_agg_hours
            )
        except ImportError:
            raise ImportError("Install meteostat: pip install meteostat")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Meteostat data: {e}")
        
        # Check if we got any data
        if precip_df.empty:
            print(f"⚠️  No Meteostat data found for station {meteostat_station} between {from_date} and {to_date}")
            return scenario
        
        # Calculate intensity in mm/hr
        precip_df['intensity_mm_hr'] = precip_df['precipitation'] / meteostat_agg_hours
        
        # Use this data directly
        times = precip_df['date'].values
        intensities = precip_df['intensity_mm_hr'].values  # Keep positive, handle sign in flux expression
        
        time_converter = TimeConverter(pd.Timestamp(times[0]).to_pydatetime())
        _create_rain_events(scenario, times, intensities, time_converter)
        
        print(f"✓ Fetched Meteostat data: {len(scenario.events)} events from station {meteostat_station}")
        return scenario
    
    if csv_path:
        # Try loading with datetime column
        try:
            loader = CSVLoader(csv_path, time_col)
            has_datetime = isinstance(loader.df.index, pd.DatetimeIndex)
        except:
            # Not a datetime column, load as regular CSV
            loader = CSVLoader(csv_path)
            has_datetime = False
        
        if has_datetime:
            # Filter date range if specified
            if from_date or to_date:
                loader.filter_dates(from_date, to_date)
            
            # Setup time converter
            times = loader.get_datetimes()
            time_converter = TimeConverter(pd.Timestamp(times[0]).to_pydatetime())
            
            # Get rainfall data
            rain_values = loader.get_numeric(rain_col)
            
            # Convert to mm/hour based on rain_unit (if specified) or auto-compute
            if rain_unit and 'day' in rain_unit.lower():
                # User specified daily data - simple division by 24
                intensities = rain_values / 24.0
                print(f"✓ Using specified rain_unit: {rain_unit} (converted to mm/hr)")
            elif rain_unit and 'hour' in rain_unit.lower():
                # Already in hourly units
                intensities = rain_values
                print(f"✓ Using specified rain_unit: {rain_unit}")
            else:
                # Auto-compute from time differences (legacy behavior)
                time_diffs = pd.Series(times).diff().dt.total_seconds() / 3600
                time_diffs.iloc[0] = time_diffs.iloc[1] if len(time_diffs) > 1 else 1.0
                intensities = rain_values / time_diffs.values
                print(f"✓ Auto-computed intensities from datetime (time steps: {time_diffs.iloc[0]:.1f}h)")
        else:
            # Numeric time column - user must specify rain_unit
            if rain_unit is None:
                raise ValueError("Must specify 'rain_unit' (e.g., 'mm/hour' or 'mm/day') for numeric time columns")
            
            times = loader.get_numeric(time_col)
            time_converter = None
            
            # Get rainfall intensities and convert to mm/hour if needed
            intensities = loader.get_numeric(rain_col)
            if 'day' in rain_unit.lower():
                intensities = intensities / 24.0
            elif 'hour' not in rain_unit.lower():
                raise ValueError(f"rain_unit must contain 'hour' or 'day', got: {rain_unit}")
            
            print(f"✓ Using specified rain_unit: {rain_unit}")
        
        # Create events from intensity changes
        _create_rain_events(scenario, times, intensities, time_converter)
        
        print(f"✓ Created {len(scenario.events)} rain events with constant intensities")
    
    elif events:
        # Manual event specification
        zone_names = [z.name for z in scenario.zones]
        for i, e in enumerate(events):
            for zone_name in zone_names:
                scenario.add_event(f"event_{i}_{zone_name}", e['start'], e['end'], 
                                 e['rate'], zone_name)
    
    return scenario


def _create_rain_events(scenario: SourceScenario, times, intensities, time_converter=None):
    """
    Simple logic: go through each timestep, create event when rain > threshold.
    Each event spans exactly one timestep (e.g., one day for daily data).
    """
    
    # Minimum rain intensity threshold (mm/hr)
    min_intensity_threshold = 0.1
    
    for i, intensity in enumerate(intensities):
        # Create event for this timestep if above threshold
        if intensity >= min_intensity_threshold:
            
            # Calculate timestep boundaries
            if time_converter:
                # For datetime data
                start_time = times[i]
                if i + 1 < len(times):
                    end_time = times[i + 1]  # Next timestep
                else:
                    # Last timestep - estimate duration from previous interval
                    if i > 0:
                        delta = times[i] - times[i-1]
                        end_time = times[i] + delta
                    else:
                        # Single timestep - assume 1 day
                        end_time = start_time + pd.Timedelta(days=1)
                
                start_sec = time_converter.to_seconds(start_time)
                end_sec = time_converter.to_seconds(end_time)
                start_dt = start_time
                end_dt = end_time
            else:
                # For numeric data
                start_sec = times[i] * 86400
                if i + 1 < len(times):
                    end_sec = times[i + 1] * 86400
                else:
                    # Last timestep - estimate duration
                    if i > 0:
                        delta = times[i] - times[i-1]
                        end_sec = (times[i] + delta) * 86400
                    else:
                        # Single timestep - assume 1 day
                        end_sec = (times[i] + 1) * 86400
                start_dt = None
                end_dt = None
            
            # Add the event for this timestep
            scenario.add_event(
                f"rain_event_{len(scenario.events)}",
                start_sec / scenario._units[scenario.time_unit],
                end_sec / scenario._units[scenario.time_unit],
                intensity,
                list(scenario.zones),
                start_datetime=start_dt,
                end_datetime=end_dt
            )


def chloride_scenario(road_x_min: float, road_x_max: float, applications: list, 
                     time_unit: str = "hours", rate_unit: str = "kg/m2/hr"):
    """
    Create chloride application scenario
    
    Args:
        applications: list of dicts with 'time', 'duration', 'rate' keys
        rate_unit: unit of application rate (default: kg/m2/hr, no conversion needed)
    """
    # No unit conversion needed if already in SI-compatible units
    conversion = 1.0 if rate_unit == "kg/m2/hr" else 1.0
    scenario = SourceScenario(time_unit, rate_unit_conversion=conversion)
    scenario.add_zone("road", road_x_min, road_x_max)
    
    for i, app in enumerate(applications):
        scenario.add_event(f"app_{i}", app['time'], app['time'] + app['duration'], 
                          app['rate'], "road")
    
    return scenario