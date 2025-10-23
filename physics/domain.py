"""
Domain management for spatially variable soil properties
Smart constructors for common multi-material scenarios
"""
import numpy as np
from firedrake import Function, RectangleMesh

class Domain:
    """
    Manages spatially variable soil properties
    Provides smart constructors for common geometries
    """

    def __init__(self, mesh, default_material, Lx=20.0, Ly=5.0, nx=80, ny=40, waxman_smits_model=None):
        """
        Initialize domain with default material
        
        Args:
            mesh: Firedrake mesh
            default_material: SoilMaterial object to use as default
            Lx, Ly: Domain dimensions (only needed if mesh was created externally)
            nx, ny: Mesh resolution (only needed if mesh was created externally)
            waxman_smits_model: Optional Waxman-Smits model for resistivity
        """
        self.mesh = mesh
        self.default_material = default_material
        self.materials = {}  # Dictionary of material regions
        self.is_homogeneous = True
        self.material_cache = {}  # Cache for performance
        
        # Store mesh parameters (will be set properly in smart constructors)
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        if waxman_smits_model is not None:
            self.ws_model = waxman_smits_model
    
    # ==========================================
    # SMART CONSTRUCTORS - Common Geometries
    # ==========================================
    
    @classmethod
    def homogeneous(cls, material, Lx=20.0, Ly=5.0, nx=80, ny=40):
        """
        Single uniform material everywhere with rectangular mesh
        
        Args:
            material: SoilMaterial object
            Lx: Domain length (m) - default 20.0
            Ly: Domain height (m) - default 5.0  
            nx: Number of elements in x - default 80
            ny: Number of elements in y - default 40
        
        Example:
            domain = Domain.homogeneous(sand, Lx=20.0, Ly=5.0, nx=80, ny=40)
        """
        mesh = RectangleMesh(nx, ny, Lx, Ly)
        instance = cls(mesh, material, Lx=Lx, Ly=Ly, nx=nx, ny=ny)
        return instance
    
    @classmethod
    def horizontal_layers(cls, layers, Lx=20.0, Ly=5.0, nx=80, ny=40):
        """
        Multiple horizontal layers from bottom to top with rectangular mesh
        
        Args:
            layers: List of (material, thickness) tuples from bottom to top
            Lx: Domain length (m) - default 20.0
            Ly: Domain height (m) - default 5.0  
            nx: Number of elements in x - default 80
            ny: Number of elements in y - default 40
        
        Example:
            # Three layers: gravel base, till, dirt on top
            domain = Domain.horizontal_layers([
                (gravel, 1.0),  # 0-1m
                (till, 2.0),    # 1-3m
                (dirt, 2.0)     # 3-5m
            ], Lx=20.0, Ly=5.0, nx=80, ny=40)
        """
        if not layers:
            raise ValueError("Must provide at least one layer")
        
        # Create mesh
        mesh = RectangleMesh(nx, ny, Lx, Ly)
        
        # Top layer is default
        domain = cls(mesh, layers[-1][0], Lx=Lx, Ly=Ly, nx=nx, ny=ny)
        
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
    
    # ==========================================
    # RECTANGLE HELPER (convenience method)
    # ==========================================
    
    def add_rectangle(self, material, x_min, x_max, y_min, y_max, name=None):
        """
        Add rectangular material region (convenience method)
        
        Args:
            material: SoilMaterial object
            x_min, x_max: x-coordinate bounds
            y_min, y_max: y-coordinate bounds
            name: Optional name for this region
        """
        def in_rectangle(x, y):
            return x_min <= x <= x_max and y_min <= y <= y_max
        
        self.add_material_region(material, in_rectangle, name=name or "rectangle")
        return self
    
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
    
    def compute_coefficient_fields(self, pressure_field):
        """
        Compute Cm and kr fields from pressure field
        
        Args:
            pressure_field: Firedrake Function containing pressure head values
        
        Returns:
            Tuple of (Cm_field, kr_field) as Firedrake Functions
        """
        V = pressure_field.function_space()
        Cm_field = Function(V)
        kr_field = Function(V)
        
        coords = self.mesh.coordinates.dat.data
        p_vals = pressure_field.dat.data[:]
        
        if self.is_homogeneous:
            # Fast path for homogeneous domain
            Cm_vals, kr_vals = self.default_material.hydraulic_model.compute_fields(p_vals)
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
    
    def compute_Ks_field(self, V):
        """
        Compute spatially varying Ks field
        
        Args:
            V: Function space
        
        Returns:
            Ks_field as Firedrake Function
        """
        Ks_field = Function(V)
        
        if self.is_homogeneous:
            # Uniform Ks
            Ks_field.dat.data[:] = self.default_material.Ks
        else:
            # Heterogeneous Ks
            coords = self.mesh.coordinates.dat.data
            Ks_vals = np.zeros(len(coords))
            
            for i, (x, y) in enumerate(coords):
                material = self.get_material_at_point(x, y)
                Ks_vals[i] = material.Ks
            
            Ks_field.dat.data[:] = Ks_vals
        
        return Ks_field

    def compute_saturation_field(self, pressure_field):
        """
        Compute saturation field from pressure field
        
        Args:
            pressure_field: Firedrake Function containing pressure head values
        
        Returns:
            S_field as Firedrake Function (saturation 0-1)
        """
        V = pressure_field.function_space()
        S_field = Function(V)
        
        coords = self.mesh.coordinates.dat.data
        p_vals = pressure_field.dat.data[:]
        S_vals = np.zeros_like(p_vals)
        
        for i, (x, y) in enumerate(coords):
            Hp = p_vals[i]
            material = self.get_material_at_point(x, y)
            theta = material.water_content(Hp)
            # Saturation = (θ - θr) / (θs - θr), clipped to [0, 1]
            S_vals[i] = max(0.0, min(1.0, 
                (theta - material.hydraulic_model.theta_r) / 
                (material.hydraulic_model.theta_s - material.hydraulic_model.theta_r)))
        
        S_field.dat.data[:] = S_vals
        return S_field

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
        import matplotlib.tri as tri
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        
        # Get mesh data
        coords = self.mesh.coordinates.dat.data
        cells = self.mesh.coordinates.cell_node_map().values
        
        # Get unique materials and assign IDs
        unique_materials = [self.default_material]
        for data in self.materials.values():
            mat = data['material']
            if mat not in unique_materials:
                unique_materials.append(mat)
        
        material_to_id = {mat: i for i, mat in enumerate(unique_materials)}
        
        # Assign material IDs per cell (not per vertex) to avoid artifacts
        material_ids = np.zeros(len(cells))
        for i, cell in enumerate(cells):
            # Get cell centroid
            cell_coords = coords[cell]
            centroid_x = np.mean(cell_coords[:, 0])
            centroid_y = np.mean(cell_coords[:, 1])
            
            # Assign material at centroid
            mat = self.get_material_at_point(centroid_x, centroid_y)
            material_ids[i] = material_to_id[mat]
        
        # Create triangulation
        x = coords[:, 0]
        y = coords[:, 1]
        triangulation = tri.Triangulation(x, y, cells)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create colormap with distinct colors
        colors = ['#8B4513', '#2E8B57']  # Brown for till, green for terreau
        cmap = ListedColormap(colors[:len(unique_materials)])
        
        # Use tripcolor for proper mesh visualization with transparency
        tpc = ax.tripcolor(triangulation, facecolors=material_ids, cmap=cmap, 
                          vmin=-0.5, vmax=len(unique_materials)-0.5,
                          edgecolors='none', alpha=0.7)
        
        # Create legend handles for materials
        legend_elements = [
            Patch(facecolor=colors[i], alpha=0.7, edgecolor='black', 
                  label=mat.name)
            for i, mat in enumerate(unique_materials)
        ]
        
        # Add water table line
        legend_elements.append(
            plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2,
                      label='Water Table (1.5m)')
        )
        
        ax.axhline(y=1.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        # Add green infrastructure outline
        if len(unique_materials) > 1:
            from matplotlib.patches import Rectangle
            rect = Rectangle((9, 4), 2, 1, linewidth=2, edgecolor='darkgreen', 
                           facecolor='none', linestyle='-')
            ax.add_patch(rect)
            legend_elements.append(
                plt.Line2D([0], [0], color='darkgreen', linewidth=2,
                          label='Green Infrastructure')
            )
        
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title('Material Distribution - Project Domain', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 5])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def predict_resistivity_field(self, pressure_field, chloride_field=None,
                                  background_sigma=0.02, temperature=25.0):
        """
        Predict bulk resistivity field from hydraulic state
        Compare with your ERT measurements for validation/calibration
        
        Parameters:
        -----------
        pressure_field : Function
            Pressure head from Richards solver
        chloride_field : Function (optional)
            Chloride concentration from transport solver (mg/L)
            If None, assumes clean water
        background_sigma : float
            Background water conductivity (S/m) - measure this!
        temperature : float
            Soil temperature (°C)
        
        Returns:
        --------
        resistivity_field : np.ndarray (Ω·m)
        """
        if self.ws_model is None:
            raise ValueError("No Waxman-Smits model set! Initialize Domain with ws_model.")
        
        coords = self.mesh.coordinates.dat.data
        n_nodes = len(coords)
        
        # Get fields
        p_vals = pressure_field.dat.data[:]
        if chloride_field is not None:
            cl_vals = chloride_field.dat.data[:]
        else:
            cl_vals = np.zeros(n_nodes)
        
        # Compute resistivity at each node
        resistivity = np.zeros(n_nodes)
        
        for i, (x, y) in enumerate(coords):
            # Get material properties
            material = self.get_material_at_point(x, y)
            porosity = material.porosity
            
            # Compute saturation from pressure
            Hp = p_vals[i]
            theta = material.water_content(Hp)
            saturation = theta / porosity  # S = θ/φ
            
            # Get fluid conductivity from chloride
            sigma_w = self.ws_model.fluid_conductivity_from_chloride(
                cl_vals[i], temperature, background_sigma
            )
            
            # Predict bulk resistivity
            resistivity[i] = self.ws_model.bulk_resistivity(
                porosity, saturation, sigma_w
            )
        
        return resistivity

if __name__ == "__main__":
    from firedrake import FunctionSpace
    import matplotlib.pyplot as plt
    from .materials import SoilMaterial

    till = SoilMaterial.from_curves(name="Till")

    terreau = SoilMaterial.from_curves(name="Terreau")
    
    # Create domain with Till as base material - mesh created internally
    domain = Domain.homogeneous(till, Lx=20.0, Ly=5.0, nx=100, ny=25)
    
    domain.add_rectangle(
        material=terreau,
        x_min=9.0, x_max=11.0,
        y_min=4.0, y_max=5.0,
        name="green_infrastructure"
    )
    
    V = FunctionSpace(domain.mesh, "CG", 1)
    fig = domain.visualize_materials(V, 
                                     #save_path='domain_configuration.png'
                                     )
        
    # Add annotations to explain the setup
    ax = fig.axes[0]
    ax.text(10, 0.5, 'SATURATED\nZONE', 
           ha='center', va='center', fontsize=11, 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(10, 3.0, 'UNSATURATED\nZONE', 
           ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.text(10, 4.5, 'GI', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           color='white',
           bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))
    
    plt.savefig('domain_configuration_annotated.png', dpi=150, bbox_inches='tight')
    