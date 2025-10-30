from dataclasses import dataclass
from typing import Optional
from ..physics.hydraulic_models import HydraulicModel, CurveBasedHydraulicModel, VanGenuchtenModel, VanGenuchtenParams
from ..physics.transport_models import AnalyticalTransportModel
from ..library import SoilParameters, ContaminantProperties

@dataclass
class Material:
    """Material container - cleaner than tuples!"""
    Ks: float # Saturated hydraulic conductivity [m/s]
    porosity: float  # Porosity [-]
    hydraulic: HydraulicModel
    transport: Optional[AnalyticalTransportModel] = None
    
    def __repr__(self):
        transport_str = f", transport={self.transport.props.name}" if self.transport else ""
        return f"Material(Ks={self.Ks:.2e}{transport_str})"


# ==============================================
# TRANSPORT MODEL FACTORY
# ==============================================

def create_transport_model(contaminant_name: str = 'chloride',
                           alpha_L: float = 0.01,
                           bulk_density: float = 1600.0):
    """
    Create transport model for a contaminant
    
    Parameters:
    -----------
    contaminant_name : str
        'chloride', 'sodium', or 'calcium'
    alpha_L : float
        Longitudinal dispersivity [m]
    bulk_density : float
        Bulk density [kg/m³]
    """
    contaminant_name = contaminant_name.lower()
    
    if contaminant_name == 'chloride':
        props = ContaminantProperties.chloride(alpha_L)
    elif contaminant_name == 'sodium':
        props = ContaminantProperties.sodium(alpha_L)
    elif contaminant_name == 'calcium':
        props = ContaminantProperties.calcium(alpha_L)
    else:
        raise ValueError(f"Unknown contaminant: {contaminant_name}")
    
    return AnalyticalTransportModel(props, bulk_density)


# ==============================================
# MATERIAL FACTORIES
# ==============================================

def till_curve_RAF(Ks=9e-6, porosity=0.4, transport=None):
    """
    Till material with curve-based hydraulics
    
    Parameters:
    -----------
    Ks : float
        Saturated hydraulic conductivity [m/s]
    porosity : float
        Porosity [-]
    transport : bool or str, optional
        If True or 'chloride': use chloride transport
        If string: 'chloride', 'sodium', or 'calcium'
        If False/None: no transport
    """
    hydro_model = CurveBasedHydraulicModel.from_library("till")
    
    transport_model = None
    if transport:
        soil_params = SoilParameters.till()
        if transport is True:
            transport_model = create_transport_model('chloride', bulk_density=soil_params.bulk_density)
        elif isinstance(transport, str):
            transport_model = create_transport_model(transport, bulk_density=soil_params.bulk_density)
    
    return Material(hydraulic=hydro_model, Ks=Ks, porosity=porosity, transport=transport_model)


def terreau_curve_RAF(Ks=4e-5, porosity=0.3, transport=None):
    """
    Terreau material with curve-based hydraulics
    
    Parameters:
    -----------
    Ks : float
        Saturated hydraulic conductivity [m/s]
    porosity : float
        Porosity [-]
    transport : bool or str, optional
        If True or 'chloride': use chloride transport
        If string: 'chloride', 'sodium', or 'calcium'
        If False/None: no transport
    """
    hydro_model = CurveBasedHydraulicModel.from_library("terreau")
    
    transport_model = None
    if transport:
        soil_params = SoilParameters.terreau()
        if transport is True:
            transport_model = create_transport_model('chloride', bulk_density=soil_params.bulk_density)
        elif isinstance(transport, str):
            transport_model = create_transport_model(transport, bulk_density=soil_params.bulk_density)
    
    return Material(hydraulic=hydro_model, Ks=Ks, porosity=porosity, transport=transport_model)


def till(Ks=9e-6, porosity=0.4, transport=None):
    """Till material with Van Genuchten model"""
    vg = VanGenuchtenParams(theta_r=0.08, theta_s=0.41, alpha=3.6, n=1.56)
    model = VanGenuchtenModel(vg)
    
    transport_model = None
    if transport:
        soil_params = SoilParameters.till()
        if transport is True:
            transport_model = create_transport_model('chloride', bulk_density=soil_params.bulk_density)
        elif isinstance(transport, str):
            transport_model = create_transport_model(transport, bulk_density=soil_params.bulk_density)
    
    return Material(hydraulic=model, Ks=Ks, porosity=porosity, transport=transport_model)


def terreau(Ks=4e-5, porosity=0.3, transport=None):
    """Terreau material with Van Genuchten model"""
    vg = VanGenuchtenParams(theta_r=0.08, theta_s=0.41, alpha=3.6, n=1.56)
    model = VanGenuchtenModel(vg)
    
    transport_model = None
    if transport:
        soil_params = SoilParameters.terreau()
        if transport is True:
            transport_model = create_transport_model('chloride', bulk_density=soil_params.bulk_density)
        elif isinstance(transport, str):
            transport_model = create_transport_model(transport, bulk_density=soil_params.bulk_density)
    
    return Material(hydraulic=model, Ks=Ks, porosity=porosity, transport=transport_model)


def sand(Ks=1e-4, porosity=0.35, transport=None):
    """Sand material with Van Genuchten model"""
    vg = VanGenuchtenParams(theta_r=0.045, theta_s=0.35, alpha=14.5, n=2.68)
    model = VanGenuchtenModel(vg)
    
    transport_model = None
    if transport:
        soil_params = SoilParameters.sand()
        if transport is True:
            transport_model = create_transport_model('chloride', bulk_density=soil_params.bulk_density)
        elif isinstance(transport, str):
            transport_model = create_transport_model(transport, bulk_density=soil_params.bulk_density)
    
    return Material(hydraulic=model, Ks=Ks, porosity=porosity, transport=transport_model)


def clay(Ks=1e-9, porosity=0.45, transport=None):
    """Clay material with Van Genuchten model"""
    vg = VanGenuchtenParams(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09)
    model = VanGenuchtenModel(vg)
    
    transport_model = None
    if transport:
        soil_params = SoilParameters.clay()
        if transport is True:
            transport_model = create_transport_model('chloride', bulk_density=soil_params.bulk_density)
        elif isinstance(transport, str):
            transport_model = create_transport_model(transport, bulk_density=soil_params.bulk_density)
    
    return Material(hydraulic=model, Ks=Ks, porosity=porosity, transport=transport_model)