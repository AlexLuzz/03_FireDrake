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
    name: str
    x_min: float
    x_max: float
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    multiplier: float = 1.0

    def contains(self, x, y=None):
        in_x = self.x_min <= x <= self.x_max
        if self.y_min is not None and y is not None:
            return in_x and self.y_min <= y <= self.y_max
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
    name: str
    start: float      # seconds
    end: float        # seconds
    rate: float
    zones: List[Zone]
    profile: Optional[Callable[[float], float]] = None
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    def is_active(self, t: float) -> bool:
        return self.start <= t < self.end

    def get_rate(self, t: float) -> float:
        if not self.is_active(t):
            return 0.0
        if self.profile:
            τ = (t - self.start) / (self.end - self.start)
            return self.rate * self.profile(τ)
        return self.rate


# ============================================================
# SourceScenario: manages zones + events
# ============================================================

class SourceScenario:
    """Manages spatial–temporal source/sink distributions."""

    _units = {"s": 1, "min": 60, "h": 3600, "d": 86400}

    def __init__(self, time_unit="h", rate_conv=1.0):
        self.zones: List[Zone] = []
        self.events: List[Event] = []
        self.time_factor = self._units.get(time_unit, 3600)
        self.rate_conv = rate_conv  # e.g., mm/h → m/s

    # ---------------- Zones ----------------
    def add_zone(self, name, x_min, x_max, y_min=None, y_max=None, multiplier=1.0):
        zone = Zone(name, x_min, x_max, y_min, y_max, multiplier)
        self.zones.append(zone)
        return zone

    def get_zone(self, name) -> Optional[Zone]:
        return next((z for z in self.zones if z.name == name), None)

    # ---------------- Events ----------------
    def add_event(self, name, start, end, rate, zones, profile=None,
                  start_dt=None, end_dt=None):
        if not isinstance(zones, list):
            zones = [zones]
        zone_objs = [self.get_zone(z) if isinstance(z, str) else z for z in zones]
        e = Event(name, start * self.time_factor, end * self.time_factor,
                  rate, zone_objs, profile, start_dt, end_dt)
        self.events.append(e)
        return e

    # ---------------- Evaluation ----------------
    def get_flux(self, x, y, t):
        """Numeric flux at (x, y, t)."""
        total = 0.0
        for e in self.events:
            if not e.is_active(t):
                continue
            rate = e.get_rate(t) * self.rate_conv
            total += sum(rate * z.multiplier for z in e.zones if z.contains(x, y))
        return total

    def get_flux_expression(self, mesh, t):
        """UFL flux expression at time t."""
        coords = SpatialCoordinate(mesh)
        flux = Constant(0.0)
        for e in self.events:
            if not e.is_active(t):
                continue
            rate = e.get_rate(t) * self.rate_conv
            for z in e.zones:
                flux = conditional(z.ufl_condition(coords),
                                   Constant(-rate * z.multiplier),
                                   flux)
        return flux


# ============================================================
# Helper Scenarios
# ============================================================

def rainfall_scenario(domain_length=None, zones=None, csv_path=None,
                      time_col="Date", rain_col="Pluie tot. (mm)", events=None,
                      from_date=None, to_date=None,
                      time_unit="day", rain_unit="mm/day"):
    """Create rainfall scenario from events or CSV."""
    conv = 1 / 3600000.0  # mm/h → m/s
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
            if from_date: df = df[df.index >= from_date]
            if to_date: df = df[df.index <= to_date]
            times = df.index.to_pydatetime()
            time_converter = TimeConverter(times[0])
            numeric_times = [time_converter.to_seconds(t) for t in times]
            
            # Get intensities from filtered dataframe (handle numeric or string with comma decimal)
            raw_vals = df[rain_col]
            # Ensure we operate on string replacement only when necessary
            if raw_vals.dtype == object or raw_vals.dtype == "string":
                cleaned = raw_vals.astype(str).str.replace(',', '.', regex=False)
                intensities = pd.to_numeric(cleaned, errors='coerce').values
            else:
                intensities = pd.to_numeric(raw_vals, errors='coerce').values
        else:
            numeric_times = loader.get_numeric(time_col)
            intensities = loader.get_numeric(rain_col)
            time_converter = None

        if "day" in rain_unit.lower():
            intensities = intensities / 24.0

        _create_events_from_series(sc, numeric_times, intensities, time_converter)
    elif events:
        for e in events:
            sc.add_event(e.get("name", "event"),
                         e["start"], e["end"], e["rate"],
                         [z.name for z in sc.zones])

    return sc


def chloride_scenario(road_x_min, road_x_max, applications,
                      time_unit="h", rate_unit="kg/m2/h"):
    """Create chloride application scenario."""
    conv = 1.0 if rate_unit == "kg/m2/h" else 1.0
    sc = SourceScenario(time_unit, rate_conv=conv)
    sc.add_zone("road", road_x_min, road_x_max)
    for i, app in enumerate(applications):
        sc.add_event(f"chloride_{i}",
                     app["time"], app["time"] + app["duration"],
                     app["rate"], "road")
    return sc


# ============================================================
# Internal Helpers
# ============================================================

def _create_events_from_series(sc: SourceScenario, times, rates, time_converter=None):
    """Convert time series into constant-rate events."""
    if len(times) == 0 or len(rates) == 0:
        print("⚠️  Empty times or rates array in _create_events_from_series")
        return
    
    if len(times) != len(rates):
        print(f"⚠️  Mismatch: times has {len(times)} elements, rates has {len(rates)} elements")
        return
    
    in_event = False
    start_i = 0
    current = 0
    
    for i, rate in enumerate(rates):
        if rate > 0 and not in_event:
            in_event, start_i, current = True, i, rate
        elif in_event and (rate <= 0 or abs(rate - current) > 1e-6):
            # End current event
            if i < len(times) and start_i < len(times):
                _add_event(sc, times[start_i], times[i], current, time_converter)
            in_event = False
            # Start new event if rate > 0
            if rate > 0:
                in_event, start_i, current = True, i, rate
    
    # Handle event that extends to the end
    if in_event and start_i < len(times):
        # For the last event, we need an end time. 
        # If we have multiple points, use the last time, otherwise estimate
        if len(times) > 1:
            end_time = times[-1]
        else:
            # Single time point - estimate a duration (e.g., 1 day)
            if time_converter:
                end_time = times[0] + 86400  # Add 1 day in seconds
            else:
                end_time = times[0] + 1.0  # Add 1 day in days
        
        _add_event(sc, times[start_i], end_time, current, time_converter)


def _add_event(sc, start, end, rate, time_converter=None):
    """Helper: add one event."""
    if time_converter:
        # start and end are already in seconds (converted from datetime)
        start_s, end_s = start, end
        # Convert back to datetime for metadata
        start_dt = time_converter.to_datetime(start)
        end_dt = time_converter.to_datetime(end)
    else:
        # start and end are in days, convert to seconds
        start_s, end_s = start * 86400, end * 86400
        start_dt, end_dt = None, None

    sc.add_event(f"event_{len(sc.events)}",
                 start_s / sc.time_factor,
                 end_s / sc.time_factor,
                 rate, [z.name for z in sc.zones],
                 start_dt=start_dt, end_dt=end_dt)
