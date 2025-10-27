from firedrake import RectangleMesh
import numpy as np
from typing import Callable, List, Tuple

class Domain:
    def __init__(self, nx: int, ny: int, Lx: float, Ly: float):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        
        # Create Firedrake mesh directly
        self.mesh = RectangleMesh(nx, ny, Lx, Ly)
        
        # Get coordinates for region definitions
        self.coords = self.mesh.coordinates.dat.data_ro
        
        self.regions = {}
        self.materials = {}
        
        # Auto-create base layer
        self.add_region("base", lambda x, y: True)
    
    @classmethod
    def homogeneous(cls, nx: int, ny: int, Lx: float, Ly: float):
        """Alias for standard constructor"""
        return cls(nx, ny, Lx, Ly)
    
    @classmethod
    def layered(cls, layers: List[Tuple[str, float]], nx: int, ny: int, Lx: float, Ly: float):
        if not layers:
            raise ValueError("Must provide at least one layer")
        
        total_height = sum(thickness for _, thickness in layers)
        if abs(total_height - Ly) > 1e-6:
            raise ValueError(f"Layer heights sum to {total_height}, but Ly={Ly}")
        
        domain = cls(nx, ny, Lx, Ly)
        domain.regions.pop("base")
        
        y_bottom = 0.0
        for name, thickness in layers:
            y_top = y_bottom + thickness
            domain.add_layer(name, y_bottom, y_top)
            y_bottom = y_top
        return domain
    
    def add_region(self, name: str, mask_function: Callable):
        """
        Define region using mask function that operates on node coordinates
        mask_function: (x: float, y: float) -> bool
        """
        mask = np.array([mask_function(x, y) for x, y in self.coords], dtype=bool)
        self.regions[name] = mask
        return self
    
    def add_layer(self, name: str, y_bottom: float, y_top: float):
        self.add_region(name, lambda x, y: y_bottom <= y <= y_top)
        return self
    
    def add_rectangle(self, name: str, x_min: float, x_max: float, y_min: float, y_max: float):
        self.add_region(name, lambda x, y: x_min <= x <= x_max and y_min <= y <= y_max)
        return self
    
    def add_polygon(self, name: str, vertices: List[Tuple[float, float]]):
        """Point-in-polygon test"""
        vertices_array = np.array(vertices)
        
        def point_in_polygon(x: float, y: float) -> bool:
            n = len(vertices_array)
            inside = False
            p1x, p1y = vertices_array[0]
            
            for i in range(1, n + 1):
                p2x, p2y = vertices_array[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside
        
        self.add_region(name, point_in_polygon)
        return self
    
    def assign(self, region_name: str, material):
        """Assign material (HydraulicModel, Ks) to region"""
        if region_name not in self.regions:
            raise ValueError(f"Region '{region_name}' not in domain")
        self.materials[region_name] = material
        return self
    
    def validate(self):
        """Check all regions have materials"""
        unassigned = [r for r in self.regions if r not in self.materials]
        if unassigned:
            raise ValueError(f"Unassigned regions: {unassigned}")
    
    def __repr__(self):
        return f"Domain({self.nx}×{self.ny}, {self.Lx}m×{self.Ly}m, {len(self.regions)} regions, {len(self.materials)} assigned)"