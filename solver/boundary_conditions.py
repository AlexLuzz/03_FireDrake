"""
Boundary condition management for Richards equation
Single source of truth for water table configuration
"""
from firedrake import DirichletBC, Function

class BoundaryConditionManager:
    """
    Manages boundary conditions and water table configuration
    
    Water table can be:
    - Constant: fixed elevation
    - Linear trend: declining or rising over time
    """
    
    def __init__(self, V, initial_water_table=1.2, water_table_trend=None):
        """
        Initialize boundary condition manager
        
        Args:
            V: Firedrake function space
            initial_water_table: float
                Initial water table elevation (m)
                Used if water_table_trend is None
            water_table_trend: dict or None
                For time-varying water table:
                {
                    't_end': float (seconds),
                    'H0_end': float (m)
                }
                Linear trend from initial_water_table → H0_end over [0, t_end]
                
        Examples:
            # Constant water table at 1.5m
            bc = BoundaryConditionManager(V, initial_water_table=1.5)
            
            # Declining: 1.5m → 1.0m over 1 year
            bc = BoundaryConditionManager(
                V, 
                initial_water_table=1.5,
                water_table_trend={'t_end': 365*86400, 'H0_end': 1.0}
            )
        """
        self.V = V
        self.mesh = V.mesh()
        
        # Water table configuration
        self.H0_initial = initial_water_table
        self.use_trend = water_table_trend is not None
        
        if self.use_trend:
            self.t_start = 0.0
            self.t_end = water_table_trend['t_end']
            self.H0_end = water_table_trend['H0_end']
            
            # Linear slope: H0(t) = H0_initial + slope × t
            self.slope = (self.H0_end - self.H0_initial) / self.t_end
            
            print(f"Water table trend:")
            print(f"  Initial: {self.H0_initial:.3f} m")
            print(f"  Final:   {self.H0_end:.3f} m ")
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
            return self.H0_initial
        elif t >= self.t_end:
            return self.H0_end
        else:
            return self.H0_initial + self.slope * t
    
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
    