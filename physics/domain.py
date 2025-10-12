"""
Domain management for spatially variable soil properties
Smart constructors for common multi-material scenarios
"""
from firedrake import Function
import numpy as np

class Domain:
    """
    Manages spatially variable soil properties
    Provides smart constructors for common geometries
    """
    
    def __init__(self, mesh, default_material):
        """
        Initialize domain with default material
        
        Args:
            mesh: Firedrake mesh
            default_material: SoilMaterial object to use as default
        """
        self.mesh = mesh
        self.default_material = default_material
        self.materials = {}  # Dictionary of material regions
        self.is_homogeneous = True
        self.material_cache = {}  # Cache for performance
    
    # ==========================================
    # SMART CONSTRUCTORS - Common Geometries
    # ==========================================
    
    @classmethod
    def homogeneous(cls, mesh, material):
        """
        Single uniform material everywhere
        
        Args:
            mesh: Firedrake mesh
            material: SoilMaterial object
        
        Example:
            domain = Domain.homogeneous(mesh, sand)
        """
        return cls(mesh, material)
    
    @classmethod
    def two_layer(cls, mesh, top_material, bottom_material, interface_y):
        """
        Two horizontal layers separated at given y coordinate
        
        Args:
            mesh: Firedrake mesh
            top_material: SoilMaterial for y > interface_y
            bottom_material: SoilMaterial for y < interface_y
            interface_y: Height of interface (m)
        
        Example:
            # Dirt on top, till below
            domain = Domain.two_layer(mesh, dirt, till, interface_y=2.5)
        """
        domain = cls(mesh, top_material)
        domain.add_material_region(
            bottom_material, 
            lambda x, y: y < interface_y,
            name="bottom_layer"
        )
        return domain
    
    @classmethod
    def horizontal_layers(cls, mesh, layers):
        """
        Multiple horizontal layers from bottom to top
        
        Args:
            mesh: Firedrake mesh
            layers: List of (material, thickness) tuples from bottom to top
        
        Example:
            # Three layers: gravel base, till, dirt on top
            domain = Domain.horizontal_layers(mesh, [
                (gravel, 1.0),  # 0-1m
                (till, 2.0),    # 1-3m
                (dirt, 2.0)     # 3-5m
            ])
        """
        if not layers:
            raise ValueError("Must provide at least one layer")
        
        # Top layer is default
        domain = cls(mesh, layers[-1][0])
        
        # Add layers from bottom up
        y_bottom = 0.0
        for i, (material, thickness) in enumerate(layers[:-1]):
            y_top = y_bottom + thickness
            domain.add_material_region(
                material,
                lambda x, y, yb=y_bottom, yt=y_top: yb <= y < yt,
                name=f"layer_{i}"
            )
            y_bottom = y_top
        
        return domain
    
    @classmethod
    def left_right_split(cls, mesh, left_material, right_material, interface_x):
        """
        Vertical interface splitting domain left/right
        
        Args:
            mesh: Firedrake mesh
            left_material: Material for x < interface_x
            right_material: Material for x > interface_x
            interface_x: Position of vertical interface (m)
        
        Example:
            # Till on left, dirt on right
            domain = Domain.left_right_split(mesh, till, dirt, interface_x=7.5)
        """
        domain = cls(mesh, right_material)
        domain.add_material_region(
            left_material,
            lambda x, y: x < interface_x,
            name="left_zone"
        )
        return domain
    
    @classmethod
    def from_material_array(cls, mesh, material_ids, materials):
        """
        Define materials from 2D array (useful for complex geometries)
        
        Args:
            mesh: Firedrake mesh
            material_ids: 2D numpy array (ny, nx) with integer material IDs
            materials: List of SoilMaterial objects, indexed by material_ids
        
        Example:
            # Create material map
            mat_map = np.zeros((30, 60), dtype=int)
            mat_map[:15, :] = 0  # Bottom: till
            mat_map[15:, :30] = 1  # Top-left: dirt
            mat_map[15:, 30:] = 2  # Top-right: sand
            
            domain = Domain.from_material_array(mesh, mat_map, [till, dirt, sand])
        """
        # Use first material as default
        domain = cls(mesh, materials[0])
        
        # Get mesh coordinates
        coords = mesh.coordinates.dat.data
        Lx = coords[:, 0].max()
        Ly = coords[:, 1].max()
        ny, nx = material_ids.shape
        
        # Create lookup function using nearest neighbor
        def get_material_id_at_point(x, y):
            ix = int(np.clip(x / Lx * nx, 0, nx - 1))
            iy = int(np.clip(y / Ly * ny, 0, ny - 1))
            return material_ids[iy, ix]
        
        # Add regions for each material (except default)
        for i, material in enumerate(materials[1:], start=1):
            domain.add_material_region(
                material,
                lambda x, y, mat_id=i: get_material_id_at_point(x, y) == mat_id,
                name=f"material_{i}"
            )
        
        return domain
    
    # ==========================================
    # POLYGON-BASED REGION DEFINITION
    # ==========================================
    
    def add_polygon(self, material, vertices, name=None):
        """
        Add material region defined by polygon vertices
        
        Args:
            material: SoilMaterial object
            vertices: List of (x, y) tuples defining polygon vertices in order
            name: Optional name for this region
        
        Example:
            # Define L-shaped region
            vertices = [
                (0, 0), (10, 0), (10, 3), (5, 3), (5, 5), (0, 5)
            ]
            domain.add_polygon(till, vertices, name="L_shaped_till")
            
            # Define notched rectangle (like your image)
            vertices = [
                (0, 0), (22, 0), (22, 5), (11, 5), (11, 4.5), 
                (10, 4.5), (10, 5), (0, 5)
            ]
            domain.add_polygon(soil, vertices)
        """
        vertices_array = np.array(vertices)
        
        def point_in_polygon(x, y):
            """Ray casting algorithm for point-in-polygon test"""
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
        
        self.add_material_region(material, point_in_polygon, name=name or "polygon")
        return self
    
    # ==========================================
    # LOW-LEVEL INTERFACE
    # ==========================================
    
    def add_material_region(self, material, region_func, name=None):
        """
        Add a material to a region defined by arbitrary function
        
        Args:
            material: SoilMaterial object
            region_func: Callable (x, y) -> bool
            name: Optional name for this region
        """
        if name is None:
            name = f"{material.name}_{len(self.materials)}"
        
        self.materials[name] = {
            'material': material,
            'region': region_func
        }
        self.is_homogeneous = False
        self.material_cache.clear()  # Clear cache when adding regions
        return self
    
    def get_material_at_point(self, x: float, y: float):
        """
        Get the soil material at given coordinates
        Uses caching for performance
        """
        if self.is_homogeneous:
            return self.default_material
        
        # Check cache
        cache_key = (round(x, 6), round(y, 6))
        if cache_key in self.material_cache:
            return self.material_cache[cache_key]
        
        # Check each region (last added has priority)
        material = self.default_material
        for name, data in reversed(list(self.materials.items())):
            if data['region'](x, y):
                material = data['material']
                break
        
        # Cache result
        self.material_cache[cache_key] = material
        return material
    
    def visualize_materials(self, V, save_path=None):
        """
        Create a visualization of the material distribution
        
        Args:
            V: Function space for visualization
            save_path: Optional path to save figure
        
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Create material ID field
        coords = self.mesh.coordinates.dat.data
        material_ids = np.zeros(len(coords))
        
        # Get unique materials and assign IDs
        unique_materials = [self.default_material]
        for data in self.materials.values():
            mat = data['material']
            if mat not in unique_materials:
                unique_materials.append(mat)
        
        material_to_id = {mat: i for i, mat in enumerate(unique_materials)}
        
        # Assign IDs at each point
        for i, (x, y) in enumerate(coords):
            mat = self.get_material_at_point(x, y)
            material_ids[i] = material_to_id[mat]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_materials)))
        cmap = ListedColormap(colors)
        
        scatter = ax.scatter(x, y, c=material_ids, cmap=cmap, s=20, 
                           vmin=-0.5, vmax=len(unique_materials)-0.5)
        
        # Add colorbar with material names
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(unique_materials)))
        cbar.set_ticklabels([mat.name for mat in unique_materials])
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Material Distribution')
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    # ==========================================
    # COMPUTATION METHODS (unchanged)
    # ==========================================
    
    def compute_coefficient_fields(self, pressure_field):
        """Compute Cm and kr fields from pressure field"""
        V = pressure_field.function_space()
        Cm_field = Function(V)
        kr_field = Function(V)
        
        coords = self.mesh.coordinates.dat.data
        p_vals = pressure_field.dat.data[:]
        
        if self.is_homogeneous:
            # Fast path for homogeneous domain
            Cm_vals, kr_vals, K_vals = self.default_material.compute_fields(p_vals)
            Cm_field.dat.data[:] = Cm_vals
            kr_field.dat.data[:] = kr_vals
        else:
            # Heterogeneous domain
            Cm_vals = np.zeros_like(p_vals)
            kr_vals = np.zeros_like(p_vals)
            
            for i, (x, y) in enumerate(coords):
                Hp = p_vals[i]
                material = self.get_material_at_point(x, y)
                Cm_vals[i] = material.moisture_capacity(Hp)
                kr_vals[i] = material.relative_permeability(Hp)
            
            Cm_field.dat.data[:] = Cm_vals
            kr_field.dat.data[:] = kr_vals
        
        return Cm_field, kr_field
    
    def compute_saturation_field(self, pressure_field):
        """Compute saturation field from pressure field"""
        V = pressure_field.function_space()
        S_field = Function(V)
        
        coords = self.mesh.coordinates.dat.data
        p_vals = pressure_field.dat.data[:]
        S_vals = np.zeros_like(p_vals)
        
        for i, (x, y) in enumerate(coords):
            Hp = p_vals[i]
            material = self.get_material_at_point(x, y)
            theta = material.water_content(Hp)
            S_vals[i] = max(material.theta_r/material.theta_s, 
                           min(theta / material.theta_s, 1.0))
        
        S_field.dat.data[:] = S_vals
        return S_field
    
    def print_summary(self):
        """Print summary of domain configuration"""
        print("Domain Configuration:")
        print(f"  Default material: {self.default_material.name}")
        if self.is_homogeneous:
            print("  Homogeneous domain")
        else:
            print(f"  Heterogeneous domain with {len(self.materials)} region(s):")
            for name, data in self.materials.items():
                print(f"    - {name}: {data['material'].name}")