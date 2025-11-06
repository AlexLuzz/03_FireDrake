from firedrake import Function, Constant
import numpy as np
from typing import Optional
from ..physics.geophysical_models import archie_resistivity, ArchieParams, fluid_resistivity_from_concentration

class MaterialField:
    """
    Make the link between materials and spatial fields.
    Computes hydraulic, transport, and geophysical fields on Firedrake mesh
    All inputs/outputs are Firedrake Functions
    """
    
    def __init__(self, domain, function_space):

        self.domain = domain
        domain.validate()
        print(domain)

        self.V = function_space

        self.domain.prepare_for_symbolic_mode(function_space)
        
    def _compute_field(self, state_functions, property_func):
        """
        Generic spatial mapping: material properties → mesh fields
        
        Handles both numeric (fast) and symbolic (tape-preserving) modes
        """
        field = Function(self.V)
        
        # Normalize state_functions to a list
        if state_functions is None:
            state_list = []
        elif isinstance(state_functions, (list, tuple)):
            state_list = list(state_functions)
        else:
            state_list = [state_functions]
        
        # Check if we need symbolic mode
        # Condition 1: State variables are UFL expressions
        states_are_symbolic = any(hasattr(sf, "ufl_element") for sf in state_list)
        
        # Condition 2: Property values are UFL (test with first material)
        test_material = list(self.domain.materials.values())[0]
        test_val = (property_func(test_material, *state_list) if state_list 
                   else property_func(test_material))
        property_is_symbolic = (hasattr(test_val, "ufl_element") or 
                               isinstance(test_val, Constant))
        
        symbolic_mode = states_are_symbolic or property_is_symbolic
        
        if symbolic_mode:
            # ========================================
            # SYMBOLIC PATH: Preserve UFL structure
            # ========================================
            
            # Build piecewise expression using region indicators
            field_expr = Constant(0.0)  # Default value
            
            for region_name, material in self.domain.materials.items():
                # Get pre-computed indicator Function
                indicator = self.domain.region_indicators[region_name]
                
                # Compute property value for this material
                if state_list:
                    val = property_func(material, *state_list)
                else:
                    val = property_func(material)
                
                # Build: field_expr = indicator * val + (1 - indicator) * field_expr
                # This accumulates each region's contribution
                field_expr = indicator * val + (1.0 - indicator) * field_expr
            
            # Assign the complete symbolic expression ONCE
            field.interpolate(field_expr)
        
        else:
            # ========================================
            # NUMERIC PATH: Fast numpy operations
            # ========================================
            field_data = field.dat.data
            
            if state_list:
                # Extract numpy arrays from Functions
                state_data_list = [
                    sf.dat.data_ro if isinstance(sf, Function) else sf 
                    for sf in state_list
                ]
            
            for region_name, material in self.domain.materials.items():
                mask = self.domain.regions[region_name]
                
                if not state_list:
                    # Static property
                    field_data[mask] = property_func(material)
                else:
                    # Dynamic property (depends on state)
                    field_data[mask] = np.vectorize(
                        lambda *states: float(property_func(material, *states))
                    )(*[sd[mask] for sd in state_data_list])
        
        return field

    # =========================================
    # GENERIC FIELDS
    # =========================================
    def get_rho_b_field(self):
        """ρ_b: Bulk density [kg/m³]"""
        return self._compute_field(None, lambda mat: mat.soil.rho_b)
    
    def get_saturation_field(self, pressure_function):
        """S(p): Saturation = θ / φ [-]"""
        return self._compute_field(pressure_function, lambda mat, 
                                   p: mat.hydraulic._theta(p)/mat.soil.porosity)
    
    # Not used
    def get_porosity_field(self):
        """φ: Porosity [-]"""
        return self._compute_field(None, lambda mat: mat.soil.porosity)
    
    # =========================================
    # HYDRAULIC FIELDS
    # =========================================
    #----------- Static fields -----------
    # Not used
    def get_theta_s_field(self):
        """θ_s: Saturated water content [m³/m³]"""
        return self._compute_field(None, lambda mat: mat.hydraulic.theta_s)

    #----------- Dynamic fields ----------- 
    # Not used
    def get_theta_field(self, pressure_function):
        """θ(p): Water content [m³/m³]"""
        return self._compute_field(pressure_function, lambda mat, 
                                   p: mat.hydraulic._theta(p))

    # Used for Richards equation
    def get_Cm_field(self, pressure_function):
        """C_m(p): Moisture capacity [1/m]"""
        return self._compute_field(pressure_function, lambda mat, 
                                   p: mat.hydraulic._Cm(p))

    # Used for Richards equation
    def get_K_field(self, pressure_function):
        """K(p): Hydraulic conductivity [m/s]"""
        return self._compute_field(pressure_function, lambda mat, 
                                   p: mat.hydraulic._k(p, mat.soil.Ks))

    # Used for Richards results visualization and to compute water level
    def get_Se_field(self, pressure_function):
        """S_e(p): Effective saturation [-]"""
        return self._compute_field(pressure_function, lambda mat, 
                                   p: mat.hydraulic._Se(p))
    
    # =========================================
    # TRANSPORT FIELDS
    # =========================================
    
    def has_transport(self) -> bool:
        """Check if any materials have transport models"""
        for material in self.domain.materials.values():
            if material.transport is not None:
                return True
        return False
    
    # # Dynamic fields
    def get_R_field(self, pressure_function):
        """R: Retardation factor [-]
        R = 1 + (ρ_b * K_d) / theta
        """
        theta = self.get_theta_field(pressure_function)
        return self._compute_field(theta, lambda mat, 
                                   th: mat.transport._R(th))
    
    def get_alpha_T_field(self):
        """α_T: Longitudinal dispersivity [m]"""
        return self._compute_field(None, lambda mat: mat.transport.props.alpha_T)

    def get_alpha_L_field(self):
        """α_L: Longitudinal dispersivity [m]"""
        return self._compute_field(None, lambda mat: mat.transport.props.alpha_L)

    def get_D0_field(self, pressure_function):
        """D_0: Molecular coefficient [m²/s]"""

        phi = self.get_porosity_field()
        theta = self.get_theta_field(pressure_function)
        return self._compute_field((theta, phi), lambda mat, 
                                   th, phi: mat.transport._D0(th, phi))

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
