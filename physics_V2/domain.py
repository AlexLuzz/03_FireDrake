import numpy as np
from typing import Callable, List, Tuple

class Domain:
    def __init__(self, nx: int, ny: int, Lx: float, Ly: float):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.regions = {}
    
    @classmethod
    def homogeneous(cls, nx: int, ny: int, Lx: float, Ly: float):
        """
        Single uniform domain
        Use with MaterialField to assign one material everywhere
        
        Example:
            domain = Domain.homogeneous(100, 50, 20.0, 5.0)
            field_map = MaterialField(domain)
            field_map.assign("all", till)
        """
        domain = cls(nx, ny, Lx, Ly)
        domain.add_region("all", lambda x, y: np.ones_like(x, dtype=bool))
        return domain
    
    @classmethod
    def layered(cls, layers: List[Tuple[str, float]], nx: int, ny: int, Lx: float, Ly: float):
        """
        Horizontal layers from bottom to top
        
        Args:
            layers: List of (name, thickness) tuples from bottom up
            nx, ny: Grid resolution
            Lx, Ly: Domain size
        
        Example:
            domain = Domain.layered([
                ("bottom", 1.0),
                ("middle", 2.0),
                ("top", 2.0)
            ], nx=100, ny=50, Lx=20.0, Ly=5.0)
        """
        if not layers:
            raise ValueError("Must provide at least one layer")
        
        total_height = sum(thickness for _, thickness in layers)
        if abs(total_height - Ly) > 1e-6:
            raise ValueError(f"Layer heights sum to {total_height}, but Ly={Ly}")
        
        domain = cls(nx, ny, Lx, Ly)
        
        y_bottom = 0.0
        for name, thickness in layers:
            y_top = y_bottom + thickness
            domain.add_layer(name, y_bottom, y_top)
            y_bottom = y_top
        
        return domain
    
    def add_region(self, name: str, mask_function: Callable):
        """
        Define region using mask function
        
        Args:
            name: Region name
            mask_function: Function(x, y) returning boolean array
        """
        mask = mask_function(self.X, self.Y)
        if mask.shape != (self.ny, self.nx):
            raise ValueError(f"Mask shape mismatch")
        self.regions[name] = mask
        return self
    
    def add_layer(self, name: str, y_bottom: float, y_top: float):
        """Horizontal layer between y_bottom and y_top"""
        self.add_region(name, lambda x, y: (y >= y_bottom) & (y <= y_top))
        return self
    
    def add_rectangle(self, name: str, x_min: float, x_max: float, 
                     y_min: float, y_max: float):
        """
        Add rectangular region
        
        Example:
            domain.add_rectangle("box", 5.0, 15.0, 1.0, 3.0)
        """
        def in_rectangle(x, y):
            return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        self.add_region(name, in_rectangle)
        return self
    
    def add_polygon(self, name: str, vertices: List[Tuple[float, float]]):
        """
        Add polygon region using ray casting algorithm
        
        Args:
            name: Region name
            vertices: List of (x, y) tuples defining polygon vertices in order
        
        Example:
            vertices = [(0, 0), (10, 0), (10, 3), (5, 3), (5, 5), (0, 5)]
            domain.add_polygon("L_shape", vertices)
        """
        vertices_array = np.array(vertices)
        
        def point_in_polygon_vec(x, y):
            """Vectorized ray casting for point-in-polygon test"""
            x_flat = x.ravel()
            y_flat = y.ravel()
            n_points = len(x_flat)
            inside = np.zeros(n_points, dtype=bool)
            
            n = len(vertices_array)
            p1x, p1y = vertices_array[0]
            
            for i in range(n_points):
                px, py = x_flat[i], y_flat[i]
                p1x, p1y = vertices_array[0]
                
                for j in range(1, n + 1):
                    p2x, p2y = vertices_array[j % n]
                    
                    if py > min(p1y, p2y):
                        if py <= max(p1y, p2y):
                            if px <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                    if p1x == p2x or px <= xinters:
                                        inside[i] = not inside[i]
                    p1x, p1y = p2x, p2y
            
            return inside.reshape(x.shape)
        
        self.add_region(name, point_in_polygon_vec)
        return self
    
    def get_region_mask(self, name: str):
        if name not in self.regions:
            raise KeyError(f"Unknown region: {name}")
        return self.regions[name]
    
    def get_region_indices(self, name: str):
        return np.where(self.get_region_mask(name))
    
    def get_region_area(self, name: str):
        n_cells = np.sum(self.get_region_mask(name))
        return n_cells * self.dx * self.dy
    
    def __repr__(self):
        return f"Domain({self.nx}×{self.ny}, {self.Lx}m×{self.Ly}m, {len(self.regions)} regions)"