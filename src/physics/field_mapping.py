from firedrake import Function
import numpy as np
from typing import Optional
from .geophysical_models import archie_resistivity, ArchieParams, fluid_resistivity_from_concentration

class MaterialField:
    """
    Make the link between materials and spatial fields.
    Computes hydraulic, transport, and geophysical fields on Firedrake mesh
    All inputs/outputs are Firedrake Functions
    """
    
    def __init__(self, domain, function_space):
        self.domain = domain
        self.V = function_space
        
        # Geophysical model settings (optional)
        self.archie_params = None
        self.fluid_resistivity = 25.0  # Default clean water [Ω·m]
    
    def _compute_field(self, state_function, property_func):
        """
        Generic field computation on Firedrake mesh, CORE method handling ALL field computations.
        It works by:
        1. Creating an empty field (Firedrake Function)
        2. Looping over each region in the domain
        3. For each region:
           - Get the mask (boolean array of which mesh nodes are in this region)
           - Get the material assigned to this region
           - Extract the property using property_func(material, state)
           - Fill in the field values for all nodes in this region
        
        Parameters:
        -----------
        state_function : Function or None
            - None for static properties (Ks, porosity, etc.)
            - Pressure field for state-dependent properties (theta(p), kr(p))
        
        property_func : lambda function
            Function that extracts property from material:
            - For static: lambda mat, _: mat.Ks
            - For dynamic: lambda mat, p: mat.hydraulic.theta(p)
        
        Returns:
        --------
        field : Function
            Firedrake Function with property values at all mesh nodes
        
        Example:
        --------
        # Static property (Ks doesn't depend on pressure)
        Ks_field = self._compute_field(None, lambda mat, _: mat.Ks)
        
        # Dynamic property (theta depends on pressure)
        theta_field = self._compute_field(pressure, lambda mat, p: mat.hydraulic.theta(p))
        
        How it works for a 2-region domain:
        - Region "base" with till material → fills nodes where mask_base=True with till properties
        - Region "GI" with terreau material → fills nodes where mask_GI=True with terreau properties
        """
        field = Function(self.V)
        field_data = field.dat.data
        
        state_data = state_function.dat.data_ro if state_function is not None else None
        
        for region_name, material in self.domain.materials.items():
            mask = self.domain.regions[region_name]
            
            if state_data is None:
                field_data[mask] = property_func(material, None)
            else:
                field_data[mask] = np.vectorize(lambda s: property_func(material, s))(state_data[mask])
        
        return field
    
    # =========================================
    # HYDRAULIC FIELDS
    # =========================================
    
    def get_Ks_field(self):
        """K_s: Saturated hydraulic conductivity [m/s]"""
        return self._compute_field(None, lambda mat, _: mat.Ks)

    def get_porosity_field(self):
        """φ: Porosity [-]"""
        return self._compute_field(None, lambda mat, _: mat.hydraulic.theta_s)
    
    def get_theta_r_field(self):
        """θ_r: Residual water content [m³/m³]"""
        return self._compute_field(None, lambda mat, _: mat.hydraulic.theta_r)
    
    def get_theta_s_field(self):
        """θ_s: Saturated water content [m³/m³]"""
        return self._compute_field(None, lambda mat, _: mat.hydraulic.theta_s)
    
    def get_theta_field(self, pressure_function):
        """θ(p): Water content [m³/m³]"""
        return self._compute_field(pressure_function, lambda mat, p: mat.hydraulic.theta(p))

    def get_kr_field(self, pressure_function):
        """k_r(p): Relative permeability [-]"""
        return self._compute_field(pressure_function, lambda mat, p: mat.hydraulic.kr(p))

    def get_Cm_field(self, pressure_function):
        """C_m(p): Moisture capacity [1/m]"""
        return self._compute_field(pressure_function, lambda mat, p: mat.hydraulic.dtheta_dp(p))

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
        """S(p): Saturation = θ / φ [-]"""
        theta = self.get_theta_field(pressure_function)
        phi = self.get_porosity_field()
        
        S = Function(self.V)
        S.dat.data[:] = theta.dat.data_ro / phi.dat.data_ro
        return S
    
    # =========================================
    # TRANSPORT FIELDS
    # =========================================
    
    def has_transport(self) -> bool:
        """Check if any materials have transport models"""
        for material in self.domain.materials.values():
            if material.transport is not None:
                return True
        return False
    
    def get_D_eff_field(self, pressure_function):
        """D_eff: Effective diffusion coefficient [m²/s]"""
        if not self.has_transport():
            raise ValueError("No transport models assigned")
        
        phi = self.get_porosity_field()
        S = self.get_saturation_field(pressure_function)
        
        D_eff = Function(self.V)
        D_eff_data = D_eff.dat.data
        phi_data = phi.dat.data_ro
        S_data = S.dat.data_ro
        
        for region_name, material in self.domain.materials.items():
            if material.transport is None:
                continue
            
            mask = self.domain.regions[region_name]
            transport_model = material.transport
            
            for i in np.where(mask)[0]:
                D_eff_data[i] = transport_model.effective_diffusion(phi_data[i], S_data[i])
        
        return D_eff
    
    def get_retardation_field(self, pressure_function):
        """R: Retardation factor [-]"""
        if not self.has_transport():
            raise ValueError("No transport models assigned")
        
        phi = self.get_porosity_field()
        S = self.get_saturation_field(pressure_function)
        
        R = Function(self.V)
        R_data = R.dat.data
        R_data[:] = 1.0  # Default
        phi_data = phi.dat.data_ro
        S_data = S.dat.data_ro
        
        for region_name, material in self.domain.materials.items():
            if material.transport is None:
                continue
            
            mask = self.domain.regions[region_name]
            transport_model = material.transport
            
            for i in np.where(mask)[0]:
                R_data[i] = transport_model.retardation_factor(phi_data[i], S_data[i])
        
        return R
    
    def get_dispersion_field(self, pressure_function, velocity_x, velocity_y):
        """
        D_L, D_T: Dispersion coefficients [m²/s]
        
        Parameters:
        -----------
        pressure_function : Function
        velocity_x, velocity_y : Function
            Velocity components
        
        Returns:
        --------
        (D_L, D_T) : tuple of Functions
        """
        if not self.has_transport():
            raise ValueError("No transport models assigned")
        
        phi = self.get_porosity_field()
        S = self.get_saturation_field(pressure_function)
        
        D_L = Function(self.V)
        D_T = Function(self.V)
        
        vx_data = velocity_x.dat.data_ro
        vy_data = velocity_y.dat.data_ro
        phi_data = phi.dat.data_ro
        S_data = S.dat.data_ro
        
        for region_name, material in self.domain.materials.items():
            if material.transport is None:
                continue
            
            mask = self.domain.regions[region_name]
            transport_model = material.transport
            
            for i in np.where(mask)[0]:
                v = np.array([vx_data[i], vy_data[i]])
                DL, DT = transport_model.dispersion_coefficients(v, phi_data[i], S_data[i])
                D_L.dat.data[i] = DL
                D_T.dat.data[i] = DT
        
        return D_L, D_T
    
    # =========================================
    # GEOPHYSICAL FIELDS (Archie's Law)
    # =========================================
    
    def set_archie_model(self, fluid_resistivity: float = 25.0, **params):
        """
        Set Archie parameters for resistivity predictions
        
        Parameters:
        -----------
        fluid_resistivity : float
            Clean water resistivity [Ω·m]
        **params : a, m, n for ArchieParams
        
        Example:
        --------
        field_map.set_archie_model(fluid_resistivity=25.0, a=1.0, m=2.0, n=2.0)
        field_map.set_archie_model(25.0, **ArchieParams.till().__dict__)
        """
        self.fluid_resistivity = fluid_resistivity
        self.archie_params = ArchieParams(**params) if params else ArchieParams.sand()
    
    def get_resistivity_field(self, pressure_function, 
                             concentration_function: Optional[Function] = None,
                             temperature: float = 25.0):
        """
        ρ: Bulk electrical resistivity [Ω·m]
        
        Parameters:
        -----------
        pressure_function : Function
        concentration_function : Function, optional
            Salt concentration [mg/L]
        temperature : float [°C]
        
        Returns:
        --------
        resistivity : Function [Ω·m]
        """
        if self.archie_params is None:
            raise ValueError("No Archie model set. Use set_archie_model() first.")
        
        phi = self.get_porosity_field()
        S = self.get_saturation_field(pressure_function)
        
        rho = Function(self.V)
        rho_data = rho.dat.data
        phi_data = phi.dat.data_ro
        S_data = S.dat.data_ro
        
        # Get fluid resistivity (spatially varying if concentration provided)
        if concentration_function is not None:
            conc_data = concentration_function.dat.data_ro
            fluid_rho = np.vectorize(
                lambda c: fluid_resistivity_from_concentration(c, temperature)
            )(conc_data)
        else:
            fluid_rho = np.full(len(phi_data), self.fluid_resistivity)
        
        # Compute bulk resistivity
        for i in range(len(rho_data)):
            rho_data[i] = archie_resistivity(phi_data[i], S_data[i], 
                                            fluid_rho[i], self.archie_params)
        
        return rho
    
    def get_conductivity_field(self, pressure_function, **kwargs):
        """σ: Bulk electrical conductivity [S/m]"""
        rho = self.get_resistivity_field(pressure_function, **kwargs)
        sigma = Function(self.V)
        rho_data = rho.dat.data_ro
        sigma.dat.data[:] = np.where(rho_data > 0, 1.0 / rho_data, 0.0)
        return sigma
    
    # =========================================
    # UTILITY
    # =========================================
    
    def get_material_at_point(self, x: float, y: float):
        """Get Material object at point"""
        coords = self.domain.coords
        distances = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)
        closest_idx = np.argmin(distances)
        
        for region_name, material in self.domain.materials.items():
            if self.domain.regions[region_name][closest_idx]:
                return material
        
        raise ValueError(f"No material at ({x}, {y})")