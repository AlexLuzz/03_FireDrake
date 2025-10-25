"""
Compact spatio-temporal source/sink manager for FEM simulations
(Handles rainfall, chloride, etc.)
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
from datetime import datetime
import pandas as pd
from firedrake import Constant, SpatialCoordinate, conditional, And

from tools.csv_loader import CSVLoader
from tools.time_converter import TimeConverter


# ============================================================
# Core Data Structures
# ============================================================

@dataclass
class Zone:
    """
    Spatial zone where sources/sinks are applied.
    
    Attributes:
        name: Zone identifier (e.g., "road", "left_half", "domain")
        x_min, x_max: Horizontal extent [m]
        y_min, y_max: Vertical extent [m] (optional for 1D problems)
                      ⚠️ IMPORTANT: If using y-boundaries in 2D:
                      - y_min typically corresponds to bottom boundary (marker 1)
                      - y_max typically corresponds to top boundary (marker 4)
                      - Check your mesh boundary markers to confirm!
                      - For vertical boundaries: left=2, right=3
        multiplier: Spatial intensity factor (default=1.0)
                   Example: multiplier=0.5 → half the base rate in this zone
    
    Examples:
        # 1D zone (rainfall across entire domain width)
        Zone("domain", x_min=0, x_max=100)
        
        # 2D zone (rainfall only on left half)
        Zone("left_half", x_min=0, x_max=50, y_min=0, y_max=5)
        
        # Zone with reduced intensity (e.g., road shoulder vs road)
        Zone("shoulder", x_min=10, x_max=15, multiplier=0.3)
    """
    name: str
    x_min: float
    x_max: float
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    multiplier: float = 1.0

    def contains(self, x, y=None):
        """Check if point (x, y) is inside zone."""
        in_x = self.x_min <= x <= self.x_max
        if self.y_min is not None and y is not None:
            return in_x and self.y_min <= y <= self.y_max
        return in_x

    def ufl_condition(self, coords):
        """Generate UFL conditional expression for this zone."""
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
        
        start_dt: Optional datetime for logging/debugging
                 Example: datetime(2024, 1, 15, 10, 0)
                 Useful for: "Event 'storm_1' occurred on Jan 15, 2024 at 10:00"
                 Not used in calculations, purely metadata.
        
        end_dt: Optional datetime for logging/debugging
               Example: datetime(2024, 1, 15, 16, 0)
               Useful for: "Event lasted from 10:00 to 16:00 (6 hours)"
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
    start: float      # seconds since simulation start
    end: float        # seconds since simulation start
    rate: float       # intensity in original units (before rate_conv)
    zones: List[Zone]
    profile: Optional[Callable[[float], float]] = None
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    def is_active(self, t: float) -> bool:
        """Check if event is active at time t (in seconds)"""
        return self.start <= t < self.end

    def get_rate(self, t: float) -> float:
        """
        Get rate at time t (in seconds), applying temporal profile if present.
        
        Args:
            t: Current time (seconds)
            
        Returns:
            Rate multiplied by profile value, or 0 if inactive
        """
        if not self.is_active(t):
            return 0.0
        if self.profile:
            # Normalize time within event to [0,1]
            τ = (t - self.start) / (self.end - self.start)
            return self.rate * self.profile(τ)
        return self.rate


# ============================================================
# SourceScenario: manages zones + events
# ============================================================

class SourceScenario:
    """
    Manages spatial-temporal source/sink distributions.
    
    IMPORTANT: Internally, all times are stored and used in SECONDS.
    The time_unit parameter only affects how add_event() interprets its inputs.
    """

    _units = {"s": 1, "min": 60, "h": 3600, "d": 86400, "day": 86400}

    def __init__(self, time_unit="h", rate_conv=1.0):
        """
        Initialize scenario manager.
        
        Args:
            time_unit: Unit for time inputs in add_event() (NOT internal storage)
                      Options: "s", "min", "h", "d"/"day"
            rate_conv: Conversion factor for rates (e.g., mm/h → m/s = 1/3600000)
        """
        self.zones: List[Zone] = []
        self.events: List[Event] = []
        self.time_factor = self._units.get(time_unit, 3600)
        self.rate_conv = rate_conv

    # ---------------- Zones ----------------
    def add_zone(self, name, x_min, x_max, y_min=None, y_max=None, multiplier=1.0):
        """Add a spatial zone to the scenario."""
        zone = Zone(name, x_min, x_max, y_min, y_max, multiplier)
        self.zones.append(zone)
        return zone

    def get_zone(self, name) -> Optional[Zone]:
        """Get zone by name."""
        return next((z for z in self.zones if z.name == name), None)

    # ---------------- Events ----------------
    def add_event(self, name, start, end, rate, zones, profile=None,
                  start_dt=None, end_dt=None):
        """
        Add an event to the scenario.
        
        Args:
            name: Event identifier
            start: Start time in time_unit (will be converted to seconds)
            end: End time in time_unit (will be converted to seconds)
            rate: Intensity in original units (before rate_conv)
            zones: Zone object(s) or zone name(s)
            profile: Optional temporal profile function τ ∈ [0,1] → multiplier
            start_dt: Optional datetime for metadata/logging
            end_dt: Optional datetime for metadata/logging
        
        Returns:
            Created Event object
        """
        if not isinstance(zones, list):
            zones = [zones]
        zone_objs = [self.get_zone(z) if isinstance(z, str) else z for z in zones]
        
        # Convert input times to seconds
        e = Event(
            name=name,
            start=start * self.time_factor,
            end=end * self.time_factor,
            rate=rate,
            zones=zone_objs,
            profile=profile,
            start_dt=start_dt,
            end_dt=end_dt
        )
        self.events.append(e)
        return e

    # ---------------- Evaluation ----------------
    def get_flux(self, x, y, t):
        """
        Numeric flux at (x, y, t).
        
        Args:
            x, y: Spatial coordinates
            t: Time in SECONDS
            
        Returns:
            Total flux at point (negative = into domain for rain)
        """
        total = 0.0
        for e in self.events:
            if not e.is_active(t):
                continue
            rate = e.get_rate(t) * self.rate_conv
            # ✅ Multiplier applied ONCE per zone
            total += sum(rate * z.multiplier for z in e.zones if z.contains(x, y))
        return total

    def get_flux_expression(self, mesh, t):
        """
        UFL flux expression at time t for use in weak formulation.
        
        Args:
            mesh: Firedrake mesh
            t: Time in SECONDS
            
        Returns:
            UFL expression for flux (negative = into domain for rain)
            
        Note: For overlapping zones, last zone takes precedence.
        For additive behavior, zones should not overlap.
        """
        coords = SpatialCoordinate(mesh)
        flux = Constant(0.0)
        
        for e in self.events:
            if not e.is_active(t):
                continue
            # ✅ Use e.rate (correct attribute name)
            # ✅ Apply rate_conv (correct conversion)
            rate = e.get_rate(t) * self.rate_conv
            
            # Build conditional for all zones in this event
            for z in e.zones:
                # ✅ Multiplier applied ONCE (no double application)
                # Apply negative sign: rain entering = negative flux
                flux = conditional(
                    z.ufl_condition(coords),
                    Constant(-rate * z.multiplier),
                    flux
                )
        
        return flux


# ============================================================
# Helper Scenarios
# ============================================================

def rainfall_scenario(domain_length=None, zones=None, csv_path=None,
                      time_col="Date", rain_col="Pluie tot. (mm)", events=None,
                      from_date=None, to_date=None,
                      time_unit="day", rain_unit="mm/day"):
    """
    Create rainfall scenario from events or CSV.
    
    Args:
        domain_length: Length of domain for default zone
        zones: List of zone dicts [{"name": ..., "x_min": ..., "x_max": ...}, ...]
        csv_path: Path to CSV with rainfall data
        time_col: Column name for time/date
        rain_col: Column name for rainfall intensity
        events: List of event dicts (alternative to CSV)
        from_date: Start date for filtering CSV data
        to_date: End date for filtering CSV data
        time_unit: Unit for event times ("day", "h", etc.)
        rain_unit: Unit for rainfall ("mm/day", "mm/h", etc.)
    
    Returns:
        SourceScenario configured for rainfall
    """
    # Conversion: mm/h → m/s = 1/3600000
    conv = 1 / 3600000.0
    sc = SourceScenario(time_unit, rate_conv=conv)

    # Zones
    if zones:
        for z in zones:
            sc.add_zone(**z)
    elif domain_length:
        sc.add_zone("domain", 0, domain_length)

    if csv_path:
        loader = CSVLoader(csv_path, time_col)
        df = loader.df
        
        if isinstance(df.index, pd.DatetimeIndex):
            # Filter by date range
            if from_date: 
                df = df[df.index >= from_date]
            if to_date: 
                df = df[df.index <= to_date]
            
            if len(df) == 0:
                print("⚠️  Warning: No data after date filtering")
                return sc
            
            times = df.index.to_pydatetime()
            time_converter = TimeConverter(times[0])
            
            # Convert datetimes to seconds since start
            numeric_times = [time_converter.to_seconds(t) for t in times]
            
            # Get intensities from filtered dataframe
            raw_vals = df[rain_col]
            if raw_vals.dtype == object or raw_vals.dtype == "string":
                cleaned = raw_vals.astype(str).str.replace(',', '.', regex=False)
                intensities = pd.to_numeric(cleaned, errors='coerce').values
            else:
                intensities = pd.to_numeric(raw_vals, errors='coerce').values
        else:
            # Numeric time column
            numeric_times = loader.get_numeric(time_col)
            intensities = loader.get_numeric(rain_col)
            time_converter = None

        # Convert rain intensity units
        if "day" in rain_unit.lower():
            intensities = intensities / 24.0  # mm/day → mm/h

        # Create events from time series
        _create_events_from_series(sc, numeric_times, intensities, time_converter)
        
    elif events:
        # Create events from explicit list
        for e in events:
            sc.add_event(
                e.get("name", "event"),
                e["start"], 
                e["end"], 
                e["rate"],
                [z.name for z in sc.zones],
                profile=e.get("profile"),
                start_dt=e.get("start_dt"),
                end_dt=e.get("end_dt")
            )

    return sc


def chloride_scenario(road_x_min, road_x_max, applications,
                      time_unit="h", rate_unit="kg/m2/h"):
    """
    Create chloride application scenario.
    
    Args:
        road_x_min, road_x_max: Road zone boundaries
        applications: List of dicts with keys: time, duration, rate
        time_unit: Unit for time values
        rate_unit: Unit for application rate
    
    Returns:
        SourceScenario configured for chloride applications
    """
    conv = 1.0 if rate_unit == "kg/m2/h" else 1.0
    sc = SourceScenario(time_unit, rate_conv=conv)
    sc.add_zone("road", road_x_min, road_x_max)
    
    for i, app in enumerate(applications):
        sc.add_event(
            f"chloride_{i}",
            app["time"], 
            app["time"] + app["duration"],
            app["rate"], 
            "road"
        )
    return sc


# ============================================================
# Internal Helpers
# ============================================================

def _create_events_from_series(sc: SourceScenario, times, rates, time_converter=None):
    """
    Convert time series into constant-rate events.
    
    Args:
        sc: SourceScenario to add events to
        times: Array of times (in seconds if time_converter, else in days)
        rates: Array of rates (mm/h)
        time_converter: Optional TimeConverter for datetime handling
    """
    if len(times) == 0 or len(rates) == 0:
        print("⚠️  Empty times or rates array")
        return
    
    if len(times) != len(rates):
        print(f"⚠️  Mismatch: {len(times)} times, {len(rates)} rates")
        return
    
    # State machine for event creation
    in_event = False
    start_idx = 0
    current_rate = 0.0
    tolerance = 1e-9  # Tolerance for rate comparison
    
    for i, rate in enumerate(rates):
        if rate > tolerance and not in_event:
            # Start new event
            in_event = True
            start_idx = i
            current_rate = rate
            
        elif in_event and (rate <= tolerance or abs(rate - current_rate) > tolerance):
            # End current event
            _add_event(sc, times[start_idx], times[i], current_rate, time_converter)
            in_event = False
            
            # Start new event if rate > 0
            if rate > tolerance:
                in_event = True
                start_idx = i
                current_rate = rate
    
    # Handle event extending to end
    if in_event:
        # Use last time point as end, or estimate duration
        if len(times) > 1:
            # Extrapolate end time using last interval
            dt = times[-1] - times[-2]
            end_time = times[-1] + dt
        else:
            # Single point: assume 1 time unit duration
            end_time = times[-1] + sc.time_factor
        
        _add_event(sc, times[start_idx], end_time, current_rate, time_converter)


def _add_event(sc, start, end, rate, time_converter=None):
    """
    Helper: add one event to scenario.
    
    Args:
        sc: SourceScenario
        start: Start time (seconds if time_converter, else days)
        end: End time (seconds if time_converter, else days)
        rate: Rate in mm/h
        time_converter: Optional TimeConverter
    """
    if time_converter:
        # Times are already in seconds
        start_s, end_s = start, end
        # Convert back to datetime for metadata
        start_dt = time_converter.to_datetime(start)
        end_dt = time_converter.to_datetime(end)
    else:
        # Times are in days, need conversion to seconds
        start_s, end_s = start * 86400, end * 86400
        start_dt, end_dt = None, None

    # add_event expects time in time_unit, will convert to seconds internally
    # So we need to divide by time_factor to reverse the conversion
    sc.add_event(
        f"event_{len(sc.events)}",
        start_s / sc.time_factor,  # Convert back to input units
        end_s / sc.time_factor,
        rate,
        [z.name for z in sc.zones],
        start_dt=start_dt,
        end_dt=end_dt
    )