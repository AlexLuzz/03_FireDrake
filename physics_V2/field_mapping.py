from firedrake import Function
import numpy as np

class MaterialField:
    """
    Computes spatial fields on Firedrake mesh
    All inputs/outputs are Firedrake Functions
    """
    
    def __init__(self, domain, function_space):
        self.domain = domain
        self.V = function_space
    
    def _compute_field(self, state_function, property_func):
        """
        Generic field computation on Firedrake mesh
        
        Args:
            state_function: Firedrake Function (e.g., pressure) or None
            property_func: lambda mat, state_value: property_value
        
        Returns:
            Firedrake Function
        """
        field = Function(self.V)
        field_data = field.dat.data
        
        # Get state data if provided
        state_data = state_function.dat.data_ro if state_function is not None else None
        
        # Loop over regions
        for region_name, material in self.domain.materials.items():
            mask = self.domain.regions[region_name]
            
            if state_data is None:
                # Static property
                field_data[mask] = property_func(material, None)
            else:
                # State-dependent property (vectorized)
                field_data[mask] = np.vectorize(lambda s: property_func(material, s))(state_data[mask])
        
        return field
    
    # === STATIC FIELDS ===
    def get_Ks_field(self):
        """K_s: Saturated hydraulic conductivity [m/s]"""
        return self._compute_field(None, lambda mat, _: mat[1])
    
    def get_porosity_field(self):
        """φ: Porosity [-]"""
        return self._compute_field(None, lambda mat, _: mat[0].theta_s)
    
    def get_theta_r_field(self):
        """θ_r: Residual water content [m³/m³]"""
        return self._compute_field(None, lambda mat, _: mat[0].theta_r)
    
    def get_theta_s_field(self):
        """θ_s: Saturated water content [m³/m³]"""
        return self._compute_field(None, lambda mat, _: mat[0].theta_s)
    
    # === STATE-DEPENDENT FIELDS (take Firedrake Function) ===
    def get_theta_field(self, pressure_function):
        """θ(p): Water content [m³/m³]"""
        return self._compute_field(pressure_function, lambda mat, p: mat[0].theta(p))
    
    def get_kr_field(self, pressure_function):
        """k_r(p): Relative permeability [-]"""
        return self._compute_field(pressure_function, lambda mat, p: mat[0].kr(p))
    
    def get_Cm_field(self, pressure_function):
        """C_m(p): Moisture capacity [1/m]"""
        return self._compute_field(pressure_function, lambda mat, p: mat[0].dtheta_dp(p))
    
    def get_K_field(self, pressure_function):
        """K(p): Hydraulic conductivity [m/s]"""
        kr = self.get_kr_field(pressure_function)
        Ks = self.get_Ks_field()
        K = Function(self.V)
        K.dat.data[:] = kr.dat.data_ro * Ks.dat.data_ro
        return K
    
    def compute_saturation_field(self, pressure_function):
        """S_e(p): Effective saturation [-]"""
        theta = self.get_theta_field(pressure_function)
        theta_r = self.get_theta_r_field()
        theta_s = self.get_theta_s_field()
        
        Se = Function(self.V)
        Se.dat.data[:] = ((theta.dat.data_ro - theta_r.dat.data_ro) / 
                          (theta_s.dat.data_ro - theta_r.dat.data_ro))
        return Se
    
    def get_saturation_field(self, pressure_function):
        """S(p): Saturation [-]"""
        theta = self.get_theta_field(pressure_function)
        phi = self.get_porosity_field()
        
        S = Function(self.V)
        S.dat.data[:] = theta.dat.data_ro / phi.dat.data_ro
        return S
    
    def get_material_at_point(self, x: float, y: float):
        """Get material (model, Ks) at continuous point"""
        # Find closest node
        coords = self.domain.coords
        distances = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)
        closest_idx = np.argmin(distances)
        
        # Check which region this node belongs to
        for region_name, material in self.domain.materials.items():
            if self.domain.regions[region_name][closest_idx]:
                return material
        
        raise ValueError(f"No material at ({x}, {y})")