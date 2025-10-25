"""
Domain geometry definition
Pure geometry - no physics, no materials
"""
import numpy as np
from typing import Callable, Optional

class Domain:
    """
    Spatial domain definition
    
    Handles:
    - Mesh generation
    - Coordinate systems
    - Named region definitions (via masks)
    
    Does NOT handle:
    - Material properties
    - Physics equations
    - Boundary conditions (those go in solver)
    """
    
    def __init__(self, nx: int, ny: int, Lx: float, Ly: float):
        """
        Create rectangular domain
        
        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x, y
        Lx, Ly : float
            Domain size [m]
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        
        # Create mesh
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Mesh grids
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Named regions (stored as boolean masks)
        self.regions = {}
    
    def add_region(self, name: str, mask_function: Callable):
        """
        Define a region using a mask function
        
        Parameters:
        -----------
        name : str
            Region name
        mask_function : callable
            Function(x, y) returning boolean array
        
        Example:
        --------
        domain.add_region("top_half", lambda x, y: y > Ly/2)
        domain.add_region("center_box", lambda x, y: (x > 2) & (x < 8) & (y > 1) & (y < 4))
        """
        mask = mask_function(self.X, self.Y)
        
        if mask.shape != (self.ny, self.nx):
            raise ValueError(f"Mask shape {mask.shape} doesn't match domain ({self.ny}, {self.nx})")
        
        self.regions[name] = mask
    
    def add_layer(self, name: str, y_bottom: float, y_top: float):
        """
        Convenience method for horizontal layers
        
        Example:
        --------
        domain.add_layer("layer1", 0.0, 1.5)
        domain.add_layer("layer2", 1.5, 3.0)
        """
        self.add_region(name, lambda x, y: (y >= y_bottom) & (y <= y_top))
    
    def get_region_mask(self, name: str) -> np.ndarray:
        """Get boolean mask for a region"""
        if name not in self.regions:
            raise KeyError(f"Region '{name}' not defined. Available: {list(self.regions.keys())}")
        return self.regions[name]
    
    def get_region_indices(self, name: str) -> tuple:
        """
        Get (i, j) indices for a region
        
        Returns:
        --------
        indices : tuple
            (i_array, j_array) where mask is True
        """
        mask = self.get_region_mask(name)
        return np.where(mask)
    
    def get_region_area(self, name: str) -> float:
        """Calculate area of a region [m²]"""
        mask = self.get_region_mask(name)
        n_cells = np.sum(mask)
        cell_area = self.dx * self.dy
        return n_cells * cell_area
    
    def __repr__(self):
        return f"Domain({self.nx}×{self.ny}, {self.Lx}m×{self.Ly}m, {len(self.regions)} regions)"