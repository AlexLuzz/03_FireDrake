"""
Boundary conditions for Richards equation
"""
from firedrake import DirichletBC, Function

class BoundaryConditionManager:
    def __init__(self, V, left_wt=1.2, right_wt=None, 
                 left_trend=None, right_trend=None, time_converter=None):
        """
        Args:
            V: Function space
            left_wt: Left water table elevation (m)
            right_wt: Right water table elevation (m). If None, uses left_wt
            left_trend: (t_end, wt_end) or (datetime_end, wt_end) for left boundary
            right_trend: (t_end, wt_end) or (datetime_end, wt_end) for right boundary
            time_converter: Required if using datetime in trends
        
        Examples:
            # Constant symmetric
            bc = BoundaryConditionManager(V, left_wt=1.5)
            
            # Constant gradient
            bc = BoundaryConditionManager(V, left_wt=1.8, right_wt=1.2)
            
            # Symmetric declining
            bc = BoundaryConditionManager(V, left_wt=1.5, 
                                         left_trend=(365*86400, 1.0))
            
            # Asymmetric with different trends
            bc = BoundaryConditionManager(V, left_wt=1.8, right_wt=1.2,
                                         left_trend=(datetime(2024,12,31), 1.6),
                                         right_trend=(datetime(2024,12,31), 0.9),
                                         time_converter=config.time_converter)
        """
        self.V = V
        self.mesh = V.mesh()
        self.time_converter = time_converter
        self.y_coords = self.mesh.coordinates.dat.data[:, 1]
        
        # Set up left boundary
        self.left_wt_0 = left_wt
        if left_trend:
            t_end, wt_end = left_trend
            self.left_t_end = self._to_seconds(t_end)
            self.left_wt_end = wt_end
            self.left_slope = (wt_end - left_wt) / self.left_t_end
        else:
            self.left_t_end = None
        
        # Set up right boundary
        self.right_wt_0 = right_wt if right_wt is not None else left_wt
        if right_trend:
            t_end, wt_end = right_trend
            self.right_t_end = self._to_seconds(t_end)
            self.right_wt_end = wt_end
            self.right_slope = (wt_end - self.right_wt_0) / self.right_t_end
        else:
            self.right_t_end = None
        
        # Create boundary functions
        self.bc_left = Function(V, name="BC_left")
        self.bc_right = Function(V, name="BC_right")
        
        self._update(0.0)
    
    def _to_seconds(self, time):
        """Convert time to seconds (handles both datetime and float)"""
        if hasattr(time, 'year'):  # datetime object
            if self.time_converter is None:
                raise ValueError("time_converter required for datetime")
            return self.time_converter.to_seconds(time)
        return float(time)
    
    def _get_wt_left(self, t):
        """Get left water table at time t"""
        if self.left_t_end is None:
            return self.left_wt_0
        if t >= self.left_t_end:
            return self.left_wt_end
        return self.left_wt_0 + self.left_slope * t
    
    def _get_wt_right(self, t):
        """Get right water table at time t"""
        if self.right_t_end is None:
            return self.right_wt_0
        if t >= self.right_t_end:
            return self.right_wt_end
        return self.right_wt_0 + self.right_slope * t
    
    def _update(self, t):
        """Update hydrostatic profiles: h(y) = wt - y, keeping on adjoint tape"""
        from firedrake import SpatialCoordinate, interpolate
        
        left_wt = self._get_wt_left(t)
        right_wt = self._get_wt_right(t)
        
        # Use UFL expressions to keep on adjoint tape
        # left_wt and right_wt can be scalars or Functions (R-space)
        coords_ufl = SpatialCoordinate(self.mesh)
        
        # Create pressure profile expressions: p = wt - y
        left_pressure_expr = left_wt - coords_ufl[1]
        right_pressure_expr = right_wt - coords_ufl[1]
        
        # Interpolate to boundary Functions (stays on tape!)
        self.bc_left.interpolate(left_pressure_expr)
        self.bc_right.interpolate(right_pressure_expr)
    
    def get_dirichlet_bcs(self, t=0.0):
        """Get boundary conditions at time t"""
        self._update(t)
        return [
            DirichletBC(self.V, self.bc_left, 1),   # Left
            DirichletBC(self.V, self.bc_right, 2),  # Right
        ]
    
    def get_water_table(self, t):
        """Get (left_wt, right_wt) at time t"""
        return self._get_wt_left(t), self._get_wt_right(t)
    
    def get_gradient(self, t):
        """Get hydraulic gradient (left - right) at time t"""
        left, right = self.get_water_table(t)
        return left - right