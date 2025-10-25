"""
Spatial mapping of materials to domain
Handles extraction of property fields for use in solvers
"""
import numpy as np
from typing import Dict, Optional, Callable
from materials import Material

# ==============================================
# MATERIAL FIELD MAPPING
# ==============================================

class MaterialField:
    """
    Maps materials to spatial domain regions
    
    This class:
    - Assigns materials to named regions
    - Extracts spatial property fields (Ks, porosity, etc.)
    - Computes state-dependent fields (K(p), θ(p), etc.)
    - Handles both hydraulic and transport properties
    """
    
    def __init__(self, domain):
        """
        Parameters:
        -----------
        domain : Domain
            The spatial domain with defined regions
        """
        self.domain = domain
        self.assignments: Dict[str, Material] = {}  # region_name → Material
        
        # Cache for frequently accessed static fields
        self._cache = {}
        self._cache_valid = False
    
    # =========================================
    # MATERIAL ASSIGNMENT
    # =========================================
    
    def assign(self, region_name: str, material: Material):
        """
        Assign a material to a region
        
        Parameters:
        -----------
        region_name : str
            Name of region (must exist in domain)
        material : Material
            Material to assign
        
        Example:
        --------
        field_map.assign("top_layer", terreau)
        field_map.assign("bottom_layer", till)
        """
        if region_name not in self.domain.regions:
            available = list(self.domain.regions.keys())
            raise ValueError(f"Region '{region_name}' not defined in domain. "
                           f"Available regions: {available}")
        
        self.assignments[region_name] = material
        self._cache_valid = False  # Invalidate cache
    
    def assign_multiple(self, assignments: Dict[str, Material]):
        """
        Assign multiple materials at once
        
        Parameters:
        -----------
        assignments : dict
            {region_name: Material}
        
        Example:
        --------
        field_map.assign_multiple({
            "layer1": sand,
            "layer2": clay,
            "layer3": till
        })
        """
        for region_name, material in assignments.items():
            self.assign(region_name, material)
    
    def get_material_at(self, i: int, j: int) -> Material:
        """
        Get material at grid point (i, j)
        
        Parameters:
        -----------
        i, j : int
            Grid indices
        
        Returns:
        --------
        material : Material
        """
        for region_name, material in self.assignments.items():
            mask = self.domain.regions[region_name]
            if mask[i, j]:
                return material
        
        raise ValueError(f"No material assigned at grid point ({i}, {j})")
    
    def get_region_for_material(self, material_name: str) -> Optional[str]:
        """Find which region a material is assigned to"""
        for region_name, material in self.assignments.items():
            if material.name == material_name:
                return region_name
        return None
    
    # =========================================
    # STATIC PROPERTY FIELDS (independent of state)
    # =========================================
    
    def _build_static_field(self, property_getter: Callable) -> np.ndarray:
        """
        Generic method to build static property field
        
        Parameters:
        -----------
        property_getter : callable
            Function that takes Material and returns property value
        
        Returns:
        --------
        field : np.ndarray
            Spatial field of the property
        """
        field = np.zeros((self.domain.ny, self.domain.nx))
        
        for region_name, material in self.assignments.items():
            mask = self.domain.regions[region_name]
            field[mask] = property_getter(material)
        
        return field
    
    def get_Ks_field(self, use_cache: bool = True) -> np.ndarray:
        """
        Get saturated hydraulic conductivity field
        
        Returns:
        --------
        Ks_field : np.ndarray [m/s]
        """
        if use_cache and self._cache_valid and 'Ks' in self._cache:
            return self._cache['Ks']
        
        field = self._build_static_field(lambda mat: mat.Ks)
        
        if use_cache:
            self._cache['Ks'] = field
            self._cache_valid = True
        
        return field
    
    def get_porosity_field(self, use_cache: bool = True) -> np.ndarray:
        """
        Get porosity field
        
        Returns:
        --------
        porosity_field : np.ndarray [-]
        """
        if use_cache and self._cache_valid and 'porosity' in self._cache:
            return self._cache['porosity']
        
        field = self._build_static_field(lambda mat: mat.porosity)
        
        if use_cache:
            self._cache['porosity'] = field
            self._cache_valid = True
        
        return field
    
    def get_theta_r_field(self) -> np.ndarray:
        """Get residual water content field [m³/m³]"""
        return self._build_static_field(lambda mat: mat.theta_r)
    
    def get_theta_s_field(self) -> np.ndarray:
        """Get saturated water content field [m³/m³]"""
        return self._build_static_field(lambda mat: mat.theta_s)
    
    def get_bulk_density_field(self) -> np.ndarray:
        """Get bulk density field [kg/m³]"""
        return self._build_static_field(lambda mat: mat.bulk_density or 1600.0)
    
    # =========================================
    # STATE-DEPENDENT FIELDS (functions of pressure, saturation, etc.)
    # =========================================
    
    def get_theta_field(self, pressure_field: np.ndarray) -> np.ndarray:
        """
        Get water content field as function of pressure
        θ = θ(p)
        
        Parameters:
        -----------
        pressure_field : np.ndarray
            Pressure head field [m]
        
        Returns:
        --------
        theta_field : np.ndarray [m³/m³]
        """
        theta_field = np.zeros_like(pressure_field)
        
        for region_name, material in self.assignments.items():
            mask = self.domain.regions[region_name]
            # Vectorized computation per region
            theta_field[mask] = np.vectorize(material.theta)(pressure_field[mask])
        
        return theta_field
    
    def get_kr_field(self, pressure_field: np.ndarray) -> np.ndarray:
        """
        Get relative permeability field
        kr = kr(p)
        
        Parameters:
        -----------
        pressure_field : np.ndarray
            Pressure head field [m]
        
        Returns:
        --------
        kr_field : np.ndarray [-]
        """
        kr_field = np.zeros_like(pressure_field)
        
        for region_name, material in self.assignments.items():
            mask = self.domain.regions[region_name]
            kr_field[mask] = np.vectorize(material.kr)(pressure_field[mask])
        
        return kr_field
    
    def get_K_field(self, pressure_field: np.ndarray) -> np.ndarray:
        """
        Get absolute hydraulic conductivity field
        K = kr(p) × Ks
        
        Parameters:
        -----------
        pressure_field : np.ndarray
            Pressure head field [m]
        
        Returns:
        --------
        K_field : np.ndarray [m/s]
        """
        K_field = np.zeros_like(pressure_field)
        
        for region_name, material in self.assignments.items():
            mask = self.domain.regions[region_name]
            K_field[mask] = np.vectorize(material.K)(pressure_field[mask])
        
        return K_field
    
    def get_Cm_field(self, pressure_field: np.ndarray) -> np.ndarray:
        """
        Get moisture capacity field
        Cm = ∂θ/∂p
        
        Parameters:
        -----------
        pressure_field : np.ndarray
            Pressure head field [m]
        
        Returns:
        --------
        Cm_field : np.ndarray [1/m]
        """
        Cm_field = np.zeros_like(pressure_field)
        
        for region_name, material in self.assignments.items():
            mask = self.domain.regions[region_name]
            Cm_field[mask] = np.vectorize(material.dtheta_dp)(pressure_field[mask])
        
        return Cm_field
    
    def get_Se_field(self, pressure_field: np.ndarray) -> np.ndarray:
        """
        Get effective saturation field
        Se = (θ - θr) / (θs - θr)
        
        Parameters:
        -----------
        pressure_field : np.ndarray
            Pressure head field [m]
        
        Returns:
        --------
        Se_field : np.ndarray [-]
        """
        Se_field = np.zeros_like(pressure_field)
        
        for region_name, material in self.assignments.items():
            mask = self.domain.regions[region_name]
            Se_field[mask] = np.vectorize(material.Se)(pressure_field[mask])
        
        return Se_field
    
    def compute_all_hydraulic_fields(self, pressure_field: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute all hydraulic fields at once (optimized)
        
        Parameters:
        -----------
        pressure_field : np.ndarray
            Pressure head field [m]
        
        Returns:
        --------
        fields : dict
            {'theta': ..., 'Cm': ..., 'kr': ..., 'K': ..., 'Se': ...}
        """
        # Initialize fields
        theta_field = np.zeros_like(pressure_field)
        Cm_field = np.zeros_like(pressure_field)
        kr_field = np.zeros_like(pressure_field)
        Se_field = np.zeros_like(pressure_field)
        
        # Compute per region (allows model-specific optimization)
        for region_name, material in self.assignments.items():
            mask = self.domain.regions[region_name]
            
            # Use material's optimized compute method if available
            theta_region, Cm_region, kr_region = material.compute_hydraulic_fields(
                pressure_field[mask]
            )
            
            theta_field[mask] = theta_region
            Cm_field[mask] = Cm_region
            kr_field[mask] = kr_region
        
        # Compute derived fields
        Ks_field = self.get_Ks_field()
        K_field = kr_field * Ks_field
        
        theta_r = self.get_theta_r_field()
        theta_s = self.get_theta_s_field()
        Se_field = (theta_field - theta_r) / (theta_s - theta_r)
        
        return {
            'theta': theta_field,
            'Cm': Cm_field,
            'kr': kr_field,
            'K': K_field,
            'Se': Se_field
        }
    
    # =========================================
    # TRANSPORT FIELDS
    # =========================================
    
    def get_D_eff_field(self, saturation_field: np.ndarray) -> np.ndarray:
        """
        Get effective diffusion coefficient field
        
        Parameters:
        -----------
        saturation_field : np.ndarray
            Water saturation field [-]
        
        Returns:
        --------
        D_eff_field : np.ndarray [m²/s]
        """
        D_eff_field = np.zeros_like(saturation_field)
        
        for region_name, material in self.assignments.items():
            if not material.has_transport:
                continue
            
            mask = self.domain.regions[region_name]
            D_eff_field[mask] = np.vectorize(material.D_eff)(saturation_field[mask])
        
        return D_eff_field
    
    def get_retardation_field(self, saturation_field: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get retardation factor field
        
        Parameters:
        -----------
        saturation_field : np.ndarray
            Water saturation field [-]
        **kwargs : optional
            pH, ionic_strength, etc.
        
        Returns:
        --------
        R_field : np.ndarray [-]
        """
        R_field = np.ones_like(saturation_field)
        
        for region_name, material in self.assignments.items():
            if not material.has_transport:
                continue
            
            mask = self.domain.regions[region_name]
            R_field[mask] = np.vectorize(
                lambda s: material.retardation(s, **kwargs)
            )(saturation_field[mask])
        
        return R_field
    
    def check_transport_availability(self) -> bool:
        """Check if all materials have transport models"""
        for material in self.assignments.values():
            if not material.has_transport:
                return False
        return True
    
    # =========================================
    # UTILITY METHODS
    # =========================================
    
    def validate(self):
        """Check that all domain regions have materials assigned"""
        unassigned = []
        for region_name in self.domain.regions.keys():
            if region_name not in self.assignments:
                unassigned.append(region_name)
        
        if unassigned:
            raise ValueError(f"Regions without material assignment: {unassigned}")
    
    def summary(self) -> str:
        """Print summary of material assignments"""
        lines = [
            "Material Field Summary",
            "=" * 60,
            f"Domain: {self.domain.nx} × {self.domain.ny} grid",
            f"Total regions: {len(self.domain.regions)}",
            f"Assigned regions: {len(self.assignments)}",
            "",
            "Assignments:"
        ]
        
        for region_name, material in self.assignments.items():
            mask = self.domain.regions[region_name]
            n_cells = np.sum(mask)
            pct = 100 * n_cells / (self.domain.nx * self.domain.ny)
            
            lines.append(f"  {region_name:20s} → {material.name:20s} "
                        f"({n_cells:6d} cells, {pct:5.1f}%)")
        
        return "\n".join(lines)
    
    def get_material_list(self) -> list:
        """Get list of unique materials"""
        return list(set(self.assignments.values()))
    
    def clear_cache(self):
        """Clear cached static fields"""
        self._cache = {}
        self._cache_valid = False
    
    def __repr__(self):
        return f"MaterialField({len(self.assignments)} regions assigned)"