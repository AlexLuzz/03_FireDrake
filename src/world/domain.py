from firedrake import RectangleMesh
import numpy as np
from typing import Callable, List, Tuple, Dict

class Domain:
    def __init__(self, nx: int, ny: int, Lx: float, Ly: float, use_UFL: bool = False):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.use_UFL = use_UFL
        
        self.mesh = RectangleMesh(nx, ny, Lx, Ly)
        self.coords = self.mesh.coordinates.dat.data_ro
        
        self.regions = {}          
        self.region_metadata = {}  
        self.materials = {}
        
        self.add_region("base", lambda x, y: True, 
                        {'type': 'rectangle', 'x_bounds': (0, Lx), 'y_bounds': (0, Ly)})
    
    @classmethod
    def layered(cls, layers: List[Tuple[str, float]], nx: int, ny: int, Lx: float, Ly: float):
        domain = cls(nx, ny, Lx, Ly)
        domain.regions.clear()
        domain.region_metadata.clear()
        
        y_bottom = 0.0
        for name, thickness in layers:
            y_top = y_bottom + thickness
            domain.add_layer(name, y_bottom, y_top)
            y_bottom = y_top
        return domain

    def add_region(self, name: str, mask_function: Callable, metadata: Dict = None):
        mask = np.array([mask_function(x, y) for x, y in self.coords], dtype=bool)
        self.regions[name] = mask
        if metadata:
            self.region_metadata[name] = metadata
        return self
    
    def add_layer(self, name: str, y_bottom: float, y_top: float):
        meta = {'type': 'layer', 'x_bounds': (0, self.Lx), 'y_bounds': (y_bottom, y_top)}
        return self.add_region(name, lambda x, y: y_bottom <= y <= y_top, metadata=meta)
    
    def add_rectangle(self, name: str, x_min: float, x_max: float, y_min: float, y_max: float):
        meta = {'type': 'rectangle', 'x_bounds': (x_min, x_max), 'y_bounds': (y_min, y_max)}
        return self.add_region(name, lambda x, y: x_min <= x <= x_max and y_min <= y <= y_max, metadata=meta)

    def add_polygon(self, name: str, vertices: List[Tuple[float, float]]):
        vertices_array = np.array(vertices)
        def point_in_polygon(x, y):
            n = len(vertices_array)
            inside = False
            p1x, p1y = vertices_array[0]
            for i in range(1, n + 1):
                p2x, p2y = vertices_array[i % n]
                if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
                p1x, p1y = p2x, p2y
            return inside
        
        meta = {'type': 'polygon', 'vertices': vertices_array}
        return self.add_region(name, point_in_polygon, metadata=meta)
    
    def assign(self, region_name: str, material):
        self.materials[region_name] = material
        return self

    def validate(self):
        unassigned = [r for r in self.regions if r not in self.materials]
        if unassigned:
            raise ValueError(f"Unassigned regions: {unassigned}")

    def __repr__(self):
        return f"Domain({self.nx}×{self.ny}, {self.Lx}m×{self.Ly}m, {len(self.regions)} regions)"
