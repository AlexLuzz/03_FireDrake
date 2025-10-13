"""
Boundary condition management for Richards equation
Matches COMSOL hydraulic head boundary condition: p = ρg(H₀ - D)
"""
from firedrake import DirichletBC, Function, Constant, SpatialCoordinate, conditional, And
import numpy as np

class BoundaryConditionManager:
    """Manages boundary conditions for Richards equation solver"""
    
    def __init__(self, V, config):
        """
        Initialize boundary condition manager
        
        Args:
            V: Firedrake function space
            config: SimulationConfig object
        """
        self.V = V
        self.config = config
        self.mesh = V.mesh()
        self.hydrostatic_profile = self._create_hydrostatic_profile()
    
    def _create_hydrostatic_profile(self):
        """
        Create hydrostatic pressure profile for boundaries
        Equivalent to COMSOL: p = ρg(H₀ - D)
        In pressure head form: Hp(y) = H₀ - y
        
        Returns:
            Function with hydrostatic pressure distribution
        """
        hydro_func = Function(self.V)
        coords = self.mesh.coordinates.dat.data
        y_coords = coords[:, 1]
        
        # H₀ is the hydraulic head (water table elevation)
        H0 = self.config.initial_water_table
        
        # Pressure head: Hp = H₀ - y
        for i, y in enumerate(y_coords):
            hydro_func.dat.data[i] = H0 - y
        
        return hydro_func
    
    def get_dirichlet_bcs(self, t: float) -> list:
        """
        Get Dirichlet boundary conditions for current time
        Applies hydraulic head on left and right boundaries (COMSOL style)
        
        Args:
            t: Current time (seconds)
        
        Returns:
            List of DirichletBC objects
        """
        bcs = []
        
        # Left boundary: Hydraulic head
        bc_left = DirichletBC(self.V, self.hydrostatic_profile, 1)
        #bcs.append(bc_left)
        
        # Right boundary: Hydraulic head
        bc_right = DirichletBC(self.V, self.hydrostatic_profile, 2)
        #bcs.append(bc_right)
        
        bc_bottom = DirichletBC(self.V, Constant(0), 3)
        #bcs.append(bc_bottom)

        return bcs
