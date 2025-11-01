from dataclasses import dataclass, replace
from typing import Union
from ..physics.hydraulic_models import VanGenuchtenModel, CurveBasedHydraulicModel, VanGenuchtenParams
from ..physics.transport_models import AnalyticalTransportModel
from ..library import SoilParameters, ContaminantProperties, HydraulicCurves

@dataclass
class Material:
    """Material container with hydraulic and transport models"""
    soil: SoilParameters
    hydraulic: Union[VanGenuchtenModel, CurveBasedHydraulicModel]
    transport: AnalyticalTransportModel
    
    def __repr__(self):
        return f"Material(soil={self.soil.name})"
    
    # ==============================================
    # PRIVATE FACTORY HELPERS
    # ==============================================
    
    @staticmethod
    def _create_transport(soil: SoilParameters, contaminant='chloride', **contaminant_kwargs):
        """
        Create transport model for a contaminant
        
        Parameters:
        -----------
        contaminant : str or ContaminantProperties
            Either a contaminant name ('chloride', 'sodium', 'calcium') 
            or a custom ContaminantProperties object
        **contaminant_kwargs : dict
            Override specific contaminant properties (Kd, diffusion_coeff, etc.)
        """
        # Case 1: Custom ContaminantProperties object passed directly
        if isinstance(contaminant, ContaminantProperties):
            props = contaminant
            if contaminant_kwargs:
                props = replace(props, **contaminant_kwargs)
        
        # Case 2: String name - use predefined
        else:
            contaminant_map = {
                'chloride': ContaminantProperties.chloride,
                'chloride test': ContaminantProperties.chloride_test,
                'sodium': ContaminantProperties.sodium,
                'calcium': ContaminantProperties.calcium,
            }
            
            contaminant_name = contaminant.lower()
            if contaminant_name not in contaminant_map:
                raise ValueError(f"Unknown contaminant: {contaminant_name}. "
                            f"Choose from {list(contaminant_map.keys())}")
            
            props = contaminant_map[contaminant_name]()
            if contaminant_kwargs:
                props = replace(props, **contaminant_kwargs)
        
        return AnalyticalTransportModel(props, bulk_density=soil.bulk_density)
    
    @classmethod
    def _from_vg(cls, soil_factory, vg_defaults: dict, contaminant='chloride', **kwargs):
        """Generic Van Genuchten material factory"""
        # Split kwargs: VG, soil, and contaminant parameters
        vg_keys = VanGenuchtenParams.__dataclass_fields__.keys()
        contaminant_keys = ContaminantProperties.__dataclass_fields__.keys()
        
        vg_kwargs = {k: v for k, v in kwargs.items() if k in vg_keys}
        contaminant_kwargs = {k: v for k, v in kwargs.items() if k in contaminant_keys}
        soil_kwargs = {k: v for k, v in kwargs.items() 
                    if k not in vg_keys and k not in contaminant_keys}
        
        # Create soil and VG models with overrides
        soil_params = soil_factory()
        if soil_kwargs:
            soil_params = replace(soil_params, **soil_kwargs)
        
        vg_params = replace(VanGenuchtenParams(**vg_defaults), **vg_kwargs)
        hydro_model = VanGenuchtenModel(vg_params)
        transport_model = cls._create_transport(soil_params, contaminant, **contaminant_kwargs)
        
        return cls(soil_params, hydro_model, transport_model)
    
    @classmethod
    def _from_curves(cls, soil_factory, theta_curve, kr_curve, contaminant='chloride', **kwargs):
        """Generic curve-based material factory"""
        contaminant_keys = ContaminantProperties.__dataclass_fields__.keys()
        contaminant_kwargs = {k: v for k, v in kwargs.items() if k in contaminant_keys}
        soil_kwargs = {k: v for k, v in kwargs.items() if k not in contaminant_keys}
        
        soil_params = soil_factory()
        if soil_kwargs:
            soil_params = replace(soil_params, **soil_kwargs)
        
        hydro_model = CurveBasedHydraulicModel(theta_curve=theta_curve, kr_curve=kr_curve)
        transport_model = cls._create_transport(soil_params, contaminant, **contaminant_kwargs)
        
        return cls(soil_params, hydro_model, transport_model)
    
    # ==============================================
    # PUBLIC MATERIAL FACTORIES
    # ==============================================
    
    @classmethod
    def till(cls, contaminant='chloride', **kwargs):
        """
        Till material with Van Genuchten model
        
        Parameters:
        -----------
        contaminant : str
            Contaminant name ('chloride', 'sodium', 'calcium')
        **kwargs : dict
            Override any soil or VG parameter (Ks, alpha, n, theta_s, etc.)
            
        Examples:
        ---------
        >>> mat = Material.till()
        >>> mat = Material.till(Ks=2e-5, alpha=4.0)
        >>> mat = Material.till(contaminant='sodium', porosity=0.45)
        """
        return cls._from_vg(
            SoilParameters.till,
            {'theta_r': 0.08, 'theta_s': 0.41, 'alpha': 3.6, 'n': 1.56},
            contaminant, **kwargs
        )
    
    @classmethod
    def terreau(cls, contaminant='chloride', **kwargs):
        """Terreau material with Van Genuchten model"""
        return cls._from_vg(
            SoilParameters.terreau,
            {'theta_r': 0.08, 'theta_s': 0.41, 'alpha': 3.6, 'n': 1.56},
            contaminant, **kwargs
        )
    
    @classmethod
    def sand(cls, contaminant='chloride', **kwargs):
        """Sand material with Van Genuchten model"""
        return cls._from_vg(
            SoilParameters.sand,
            {'theta_r': 0.045, 'theta_s': 0.35, 'alpha': 14.5, 'n': 2.68},
            contaminant, **kwargs
        )
    
    @classmethod
    def clay(cls, contaminant='chloride', **kwargs):
        """Clay material with Van Genuchten model"""
        return cls._from_vg(
            SoilParameters.clay,
            {'theta_r': 0.068, 'theta_s': 0.38, 'alpha': 0.8, 'n': 1.09},
            contaminant, **kwargs
        )
    
    @classmethod
    def till_curve_RAF(cls, contaminant='chloride', **kwargs):
        """
        Till material with curve-based hydraulics
        
        Parameters:
        -----------
        contaminant : str
            Contaminant name
        **kwargs : dict
            Soil parameter overrides (Ks, porosity, bulk_density, etc.)
        """
        return cls._from_curves(
            SoilParameters.till,
            HydraulicCurves.till_theta(),
            HydraulicCurves.till_kr(),
            contaminant, **kwargs
        )
    
    @classmethod
    def terreau_curve_RAF(cls, contaminant='chloride', **kwargs):
        """Terreau material with curve-based hydraulics"""
        return cls._from_curves(
            SoilParameters.terreau,
            HydraulicCurves.terreau_theta(),
            HydraulicCurves.terreau_kr(),
            contaminant, **kwargs
        )
    

"""
# 1. Use predefined contaminant
mat1 = Material.till(contaminant='chloride')

# 2. Override specific contaminant properties
mat2 = Material.till(contaminant='chloride', Kd=0.5, diffusion_coeff=1e-9)

# 3. Pass completely custom ContaminantProperties
custom_props = ContaminantProperties(
    name='my_pollutant',
    Kd=2.0,
    diffusion_coeff=5e-10,
    # ... other properties
)
mat3 = Material.sand(contaminant=custom_props)

# 4. Combine: custom object + overrides (for optimization!)
mat4 = Material.clay(contaminant=custom_props, Kd=optimized_kd)

# 5. Override everything at once for optimization
mat5 = Material.till(
    alpha=opt_params[0],      # VG parameter
    Ks=opt_params[1],          # Soil parameter
    Kd=opt_params[2]           # Contaminant parameter
)
"""