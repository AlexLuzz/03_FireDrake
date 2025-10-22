"""
Boundary condition management for Richards equation
Single source of truth for water table configuration
"""
from firedrake import DirichletBC, Function
from datetime import datetime, timedelta

class BoundaryConditionManager:
    """
    Manages boundary conditions and water table configuration
    
    Water table can be:
    - Constant: fixed elevation
    - Linear trend: declining or rising over time
    """
    
    def __init__(self, V, initial_water_table=1.2, water_table_trend=None, time_converter=None):
        """
        Initialize boundary condition manager
        
        Args:
            V: Firedrake function space
            initial_water_table: float
                Initial water table elevation (m)
                Used if water_table_trend is None
            water_table_trend: dict or None
                For time-varying water table. Can use either seconds or datetime:
                
                Option 1 - Seconds (legacy):
                {
                    't_end': float (seconds),
                    'H0_end': float (m)
                }
                
                Option 2 - Datetime (recommended):
                {
                    'start_datetime': datetime (optional, defaults to simulation start),
                    'end_datetime': datetime,
                    'H0_start': float (m),
                    'H0_end': float (m)
                }
                
                Linear trend from H0_start → H0_end
            
            time_converter: TimeConverter object (required if using datetime mode)
                
        Examples:
            # Constant water table at 1.5m
            bc = BoundaryConditionManager(V, initial_water_table=1.5)
            
            # Declining: 1.5m → 1.0m over 1 year (seconds mode)
            bc = BoundaryConditionManager(
                V, 
                initial_water_table=1.5,
                water_table_trend={'t_end': 365*86400, 'H0_end': 1.0}
            )
            
            # Declining: 1.2m → 1.0m from Feb 22 to Dec 31, 2024 (datetime mode)
            bc = BoundaryConditionManager(
                V, 
                initial_water_table=1.2,
                water_table_trend={
                    'start_datetime': datetime(2024, 2, 22),
                    'end_datetime': datetime(2024, 12, 31),
                    'H0_start': 1.2,
                    'H0_end': 1.0
                },
                time_converter=config.time_converter
            )
        """
        self.V = V
        self.mesh = V.mesh()
        self.time_converter = time_converter
        
        # Water table configuration
        self.H0_initial = initial_water_table
        self.use_trend = water_table_trend is not None
        
        if self.use_trend:
            # Check if using datetime or seconds mode
            if 'end_datetime' in water_table_trend:
                # Datetime mode
                if time_converter is None:
                    raise ValueError("time_converter required when using end_datetime in water_table_trend")
                
                # Get trend parameters
                start_datetime = water_table_trend.get('start_datetime', time_converter.start)
                end_datetime = water_table_trend['end_datetime']
                H0_start = water_table_trend.get('H0_start', initial_water_table)
                H0_end = water_table_trend['H0_end']
                
                # Convert to seconds relative to simulation start
                self.t_start = time_converter.to_seconds(start_datetime)
                self.t_end = time_converter.to_seconds(end_datetime)
                self.H0_start = H0_start
                self.H0_end = H0_end
                
                # Linear slope: H0(t) = H0_start + slope × (t - t_start)
                self.slope = (self.H0_end - self.H0_start) / (self.t_end - self.t_start)
                
                print(f"Water table trend (datetime mode):")
                print(f"  From: {start_datetime.strftime('%Y-%m-%d')} at {H0_start:.3f} m")
                print(f"  To:   {end_datetime.strftime('%Y-%m-%d')} at {H0_end:.3f} m")
                print(f"  Slope: {self.slope * 86400 * 1000:.3f} mm/day")
            else:
                # Seconds mode (legacy)
                self.t_start = 0.0
                self.t_end = water_table_trend['t_end']
                self.H0_start = initial_water_table
                self.H0_end = water_table_trend['H0_end']
                
                # Linear slope: H0(t) = H0_initial + slope × t
                self.slope = (self.H0_end - self.H0_start) / self.t_end
                
                print(f"Water table trend (seconds mode):")
                print(f"  Initial: {self.H0_start:.3f} m")
                print(f"  Final:   {self.H0_end:.3f} m at t={self.t_end/86400:.1f} days")
            
        else:
            print(f"Water table: constant at {self.H0_initial:.3f} m")
        
        # Create hydrostatic profile
        self.hydrostatic_profile = Function(self.V)
        self.y_coords = self.mesh.coordinates.dat.data[:, 1]
        
        # Initialize with initial water table
        self._update_profile(self.H0_initial)
    
    def _update_profile(self, H0):
        """
        Update hydrostatic pressure profile
        
        EQUATION: Hp(y) = H0 - y
        
        Args:
            H0: Water table elevation (m)
        """
        for i, y in enumerate(self.y_coords):
            self.hydrostatic_profile.dat.data[i] = H0 - y
    
    def get_water_table_elevation(self, t):
        """
        Get water table elevation at time t
        
        LINEAR TREND: H0(t) = H0_initial + slope × t
        
        Args:
            t: Current time (seconds)
        
        Returns:
            Water table elevation (m)
        """
        if not self.use_trend:
            return self.H0_initial
        
        # Linear interpolation (clamp to bounds)
        if t <= self.t_start:
            return self.H0_start
        elif t >= self.t_end:
            return self.H0_end
        else:
            return self.H0_start + self.slope * (t - self.t_start)
    
    def get_dirichlet_bcs(self, t=0.0) -> list:
        """
        Get Dirichlet boundary conditions at time t
        Automatically updates if water table is time-varying
        
        Args:
            t: Current time (seconds)
        
        Returns:
            List of DirichletBC objects
        """
        # Update hydrostatic profile for current time
        H0_current = self.get_water_table_elevation(t)
        self._update_profile(H0_current)
        
        bcs = []
        
        # Lateral boundaries: hydrostatic pressure
        bc_left = DirichletBC(self.V, self.hydrostatic_profile, 1)
        bc_right = DirichletBC(self.V, self.hydrostatic_profile, 2)
        bcs.extend([bc_left, bc_right])
        
        # Bottom boundary: free drainage (no Dirichlet BC)
        # If you need Dirichlet at bottom, uncomment:
        # bc_bottom = DirichletBC(self.V, Constant(0), 3)
        # bcs.append(bc_bottom)
        
        return bcs
    