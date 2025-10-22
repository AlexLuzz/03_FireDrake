"""
Source Terms for Flow and Transport
Works with your existing RainScenario from rain_config.py
"""
from abc import ABC, abstractmethod
from firedrake import Constant, SpatialCoordinate, conditional, And
import numpy as np

# ==========================================
# ABSTRACT BASE CLASS
# ==========================================

class SourceTerm(ABC):
    """Abstract base for all source/sink terms"""
    
    @abstractmethod
    def get_ufl_expression(self, t: float, mesh):
        """
        Get UFL expression for use in weak forms
        
        Parameters:
        -----------
        t : float
            Current time (seconds)
        mesh : firedrake.Mesh
            Mesh object
        
        Returns:
        --------
        UFL expression for integration (dx or ds)
        """
        pass
    
    def evaluate_at_point(self, t: float, x: float = None, y: float = None):
        """
        Evaluate source at specific point (for diagnostics)
        
        Parameters:
        -----------
        t : float (seconds)
        x, y : float (meters)
        
        Returns:
        --------
        float: source value
        """
        # Default implementation - override if needed
        return 0.0


# ==========================================
# HYDRAULIC FLOW SOURCES
# ==========================================

class HydraulicSource(SourceTerm):
    """
    Hydraulic source using your RainScenario from rain_config.py
    
    This wraps your existing RainScenario to work with the SourceTerm interface
    """
    
    def __init__(self, rain_scenario):
        """
        Parameters:
        -----------
        rain_scenario : RainScenario (from rain_config.py)
            Your existing rain scenario object
        """
        self.rain_scenario = rain_scenario
    
    def get_ufl_expression(self, t: float, mesh):
        """
        Build UFL expression for rainfall at time t
        Applied as Neumann BC on top boundary (ds(4))
        
        Returns:
        --------
        flux : UFL expression (m/s, negative = into domain)
        """
        t_hours = t / 3600.0
        coords = SpatialCoordinate(mesh)
        x = coords[0]
        
        # Build expression from all active events
        flux_expr = Constant(0.0)
        
        for event in self.rain_scenario.events:
            if event.is_active(t_hours):
                for zone in event.zones:
                    # Calculate flux for this zone (mm/hr → m/s)
                    zone_flux = event.intensity * zone.multiplier / 3600000.0
                    
                    # Add contribution if x is in zone
                    flux_expr = conditional(
                        And(x >= zone.x_min, x <= zone.x_max),
                        Constant(-zone_flux),  # Negative = into domain
                        flux_expr
                    )
        
        return flux_expr
    
    def evaluate_at_point(self, t: float, x: float = None, y: float = None):
        """Evaluate rain flux at specific location"""
        t_hours = t / 3600.0
        return self.rain_scenario.get_flux_at_x(x if x is not None else 0.0, t_hours)


# ==========================================
# CONTAMINANT TRANSPORT SOURCES
# ==========================================

class ContaminantSource(SourceTerm):
    """
    Generic contaminant source term
    Can represent: surface input, degradation, root uptake, etc.
    """
    
    def __init__(self, rate: float, 
                 x_min=None, x_max=None, 
                 y_min=None, y_max=None,
                 time_function=None):
        """
        Parameters:
        -----------
        rate : float
            Base source rate (mol/m³/s or mg/L/s)
            Positive = source, Negative = sink
        x_min, x_max : float (optional)
            Spatial extent in x (None = all x)
        y_min, y_max : float (optional)
            Spatial extent in y (None = all y)
        time_function : callable(t_seconds) -> float (optional)
            Time-varying multiplier (default = constant)
        """
        self.rate = rate
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.time_function = time_function or (lambda t: 1.0)
    
    def get_ufl_expression(self, t: float, mesh):
        """Build UFL expression for volume source term"""
        time_factor = self.time_function(t)
        
        # Simple case: uniform in space
        if self.x_min is None and self.y_min is None:
            return Constant(self.rate * time_factor)
        
        # Spatially variable case
        coords = SpatialCoordinate(mesh)
        x, y = coords[0], coords[1]
        
        # Build spatial condition
        conditions = []
        if self.x_min is not None:
            conditions.append(And(x >= self.x_min, x <= self.x_max))
        if self.y_min is not None:
            conditions.append(And(y >= self.y_min, y <= self.y_max))
        
        if len(conditions) == 1:
            condition = conditions[0]
        else:
            condition = And(*conditions)
        
        return conditional(
            condition,
            Constant(self.rate * time_factor),
            Constant(0.0)
        )
    
    def evaluate_at_point(self, t: float, x: float = None, y: float = None):
        """Evaluate at specific point"""
        time_factor = self.time_function(t)
        
        # Check spatial bounds
        if x is not None and self.x_min is not None:
            if not (self.x_min <= x <= self.x_max):
                return 0.0
        if y is not None and self.y_min is not None:
            if not (self.y_min <= y <= self.y_max):
                return 0.0
        
        return self.rate * time_factor


class ChlorideFromRain(SourceTerm):
    """
    Chloride input from rainfall
    Couples with HydraulicSource (rain flux × chloride concentration)
    
    Use this for background chloride in precipitation
    """
    
    def __init__(self, hydraulic_source: HydraulicSource, 
                 rain_chloride_concentration: float):
        """
        Parameters:
        -----------
        hydraulic_source : HydraulicSource
            The rain source (for timing and intensity)
        rain_chloride_concentration : float
            Cl⁻ concentration in rainwater (mg/L)
            Montreal typical: 0.1-0.5 mg/L
            Near coast or winter: up to 2-5 mg/L
        """
        self.hydraulic = hydraulic_source
        self.c_rain = rain_chloride_concentration
    
    def get_ufl_expression(self, t: float, mesh):
        """
        Chloride mass flux = rain flux × concentration
        Applied on top boundary (ds(4))
        """
        rain_flux = self.hydraulic.get_ufl_expression(t, mesh)
        # Make positive (source) and convert to mass flux
        # rain_flux is in m/s, multiply by concentration (mg/L = g/m³)
        # Result in mg/m²/s or g/m²/s depending on units
        return -rain_flux * Constant(self.c_rain)
    
    def evaluate_at_point(self, t: float, x: float = None, y: float = None):
        rain_flux = abs(self.hydraulic.evaluate_at_point(t, x, y))
        return rain_flux * self.c_rain


class RoadSaltSource(ContaminantSource):
    """
    De-icing salt application (episodic chloride input)
    Designed for Montreal winter road maintenance scenarios
    """
    
    def __init__(self, 
                 road_x_min: float, 
                 road_x_max: float,
                 application_rate_g_m2: float,
                 application_times_hours: list,
                 dissolution_time_hours: float = 2.0):
        """
        Parameters:
        -----------
        road_x_min, road_x_max : float
            Road surface location (m)
        application_rate_g_m2 : float
            Salt application rate (g NaCl/m² per event)
            Montreal typical: 50-300 g/m² per application
        application_times_hours : list of float
            Times of salt applications (hours from simulation start)
        dissolution_time_hours : float
            How long salt takes to dissolve and mobilize (hours)
        
        Example:
        --------
        # Road at x=10-12m, three salt applications
        salt = RoadSaltSource(
            road_x_min=10.0,
            road_x_max=12.0,
            application_rate_g_m2=200.0,  # 200 g/m²
            application_times_hours=[6, 18, 30],  # Morning, evening, next morning
            dissolution_time_hours=2.0
        )
        """
        self.road_x_min = road_x_min
        self.road_x_max = road_x_max
        self.app_rate = application_rate_g_m2  # g NaCl/m²
        self.app_times = application_times_hours
        self.dissolution_time = dissolution_time_hours * 3600  # Convert to seconds
        
        # NaCl → Na⁺ + Cl⁻
        # Molecular weights: NaCl=58.44, Cl=35.45, Na=22.99
        # 1 g NaCl → 0.607 g Cl⁻
        self.cl_fraction = 35.45 / 58.44  # 0.607
        
    def get_ufl_expression(self, t: float, mesh):
        """Build UFL expression for salt dissolution"""
        # Check if any application is currently active
        source_rate = 0.0
        for t_app_hr in self.app_times:
            t_app_sec = t_app_hr * 3600
            if t_app_sec <= t < t_app_sec + self.dissolution_time:
                # Convert g/m² to flux (mg/L/s or mol/m³/s)
                # g NaCl/m² → g Cl/m² → mg Cl/m² → averaged over dissolution time
                cl_mass = self.app_rate * self.cl_fraction * 1000  # mg Cl/m²
                source_rate = cl_mass / self.dissolution_time  # mg/m²/s
                break
        
        if source_rate == 0.0:
            return Constant(0.0)
        
        # Apply only to road surface
        coords = SpatialCoordinate(mesh)
        x = coords[0]
        return conditional(
            And(x >= self.road_x_min, x <= self.road_x_max),
            Constant(source_rate),
            Constant(0.0)
        )
    
    def evaluate_at_point(self, t: float, x: float = None, y: float = None):
        """Evaluate at point"""
        # Check active application
        for t_app_hr in self.app_times:
            t_app_sec = t_app_hr * 3600
            if t_app_sec <= t < t_app_sec + self.dissolution_time:
                if x is None or (self.road_x_min <= x <= self.road_x_max):
                    cl_mass = self.app_rate * self.cl_fraction * 1000
                    return cl_mass / self.dissolution_time
        return 0.0


class CompositeSource(SourceTerm):
    """
    Combine multiple sources (superposition principle)
    Example: Background chloride from rain + episodic road salt
    """
    
    def __init__(self, sources: list):
        """
        Parameters:
        -----------
        sources : list of SourceTerm
            Individual sources to combine
        """
        self.sources = sources
    
    def get_ufl_expression(self, t: float, mesh):
        """Sum all source contributions"""
        if not self.sources:
            return Constant(0.0)
        
        expr = self.sources[0].get_ufl_expression(t, mesh)
        for source in self.sources[1:]:
            expr = expr + source.get_ufl_expression(t, mesh)
        
        return expr
    
    def evaluate_at_point(self, t: float, x: float = None, y: float = None):
        """Sum all sources at point"""
        return sum(s.evaluate_at_point(t, x, y) for s in self.sources)

