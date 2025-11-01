from dataclasses import dataclass
from typing import Optional, Union
from ..physics.hydraulic_models import VanGenuchtenModel, CurveBasedHydraulicModel, VanGenuchtenParams
from ..physics.transport_models import AnalyticalTransportModel

from ..library import SoilParameters, ContaminantProperties, HydraulicCurves

@dataclass
class Material:
    """Material container - cleaner than tuples!"""
    soil: SoilParameters
    hydraulic: Union[VanGenuchtenModel, CurveBasedHydraulicModel]
    transport: Optional[AnalyticalTransportModel] = None
    
    def __repr__(self):
        return f"Material(soil={self.soil.name})"

def create_transport_model(soil: SoilParameters, contaminant_name: str = 'chloride'):
    """
    Create transport model for a contaminant
    -----------
    contaminant_name : str
        'chloride', 'sodium', or 'calcium'
    """
    contaminant_name = contaminant_name.lower()
    
    if contaminant_name == 'chloride':
        props = ContaminantProperties.chloride()
    elif contaminant_name == 'chloride test':
        props = ContaminantProperties.chloride_test()
    elif contaminant_name == 'sodium':
        props = ContaminantProperties.sodium()
    elif contaminant_name == 'calcium':
        props = ContaminantProperties.calcium()
    else:
        raise ValueError(f"Unknown contaminant: {contaminant_name}")

    return AnalyticalTransportModel(props, bulk_density=soil.bulk_density)


# ==============================================
# MATERIAL FACTORIES
# ==============================================

# Detailed examples of material creation functions
def till_curve_RAF(transport=None):
    """
    Till material with curve-based hydraulics
    
    Parameters:
    -----------
    transport : optional, str of the contaminant name
        "chloride", "chloride test", "sodium", or "calcium"
        If provided, create transport model for this contaminant
    """
    soil_params = SoilParameters.till()
    hydro_model = CurveBasedHydraulicModel(theta_curve=HydraulicCurves.till_theta(),
                                           kr_curve=HydraulicCurves.till_kr())
    transport_model = create_transport_model(soil_params, transport) if transport else None
    return Material(soil_params, hydro_model, transport_model)

def terreau_curve_RAF(transport=None):
    soil_params = SoilParameters.terreau()
    hydro_model = CurveBasedHydraulicModel(theta_curve=HydraulicCurves.terreau_theta(),
                                           kr_curve=HydraulicCurves.terreau_kr())
    transport_model = create_transport_model(soil_params, transport) if transport else None
    return Material(soil_params, hydro_model, transport_model)


def till(transport=None):
    """Till material with Van Genuchten model"""
    soil_params = SoilParameters.till()
    vg = VanGenuchtenParams(theta_r=0.08, theta_s=0.41, alpha=3.6, n=1.56)
    hydro_model = VanGenuchtenModel(vg)
    transport_model = create_transport_model(soil_params, transport) if transport else None
    return Material(soil_params, hydro_model, transport_model)


def terreau(transport=None):
    """Terreau material with Van Genuchten model"""
    soil_params = SoilParameters.terreau()
    vg = VanGenuchtenParams(theta_r=0.08, theta_s=0.41, alpha=3.6, n=1.56)
    hydro_model = VanGenuchtenModel(vg)
    transport_model = create_transport_model(soil_params, transport) if transport else None
    return Material(soil_params, hydro_model, transport_model)

def sand(transport=None):
    """Sand material with Van Genuchten model"""
    soil_params = SoilParameters.sand()
    vg = VanGenuchtenParams(theta_r=0.045, theta_s=0.35, alpha=14.5, n=2.68)
    hydro_model = VanGenuchtenModel(vg)
    transport_model = create_transport_model(soil_params, transport) if transport else None
    return Material(soil_params, hydro_model, transport_model)

def clay(transport=None):
    """Clay material with Van Genuchten model"""
    soil_params = SoilParameters.clay()
    vg = VanGenuchtenParams(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09)
    hydro_model = VanGenuchtenModel(vg)
    transport_model = create_transport_model(soil_params, transport) if transport else None
    return Material(soil_params, hydro_model, transport_model)
