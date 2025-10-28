from dataclasses import dataclass
from typing import Optional
from ..physics.hydraulic_models import HydraulicModel, CurveBasedHydraulicModel, VanGenuchtenModel, VanGenuchtenParams
from ..physics.transport_models import AnalyticalTransportModel, chloride_transport

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


# Update factories to return Material objects
def till_curve_RAF(Ks=9e-6, porosity=0.4, transport=False):
    """Till material"""
    hydro_model = CurveBasedHydraulicModel.from_library("till")
    if transport:
        transport_model = chloride_transport()
    return Material(hydraulic=hydro_model, Ks=Ks, porosity=porosity, transport=transport_model)

def terreau_curve_RAF(Ks=4e-5, porosity=0.3, transport=False):
    """Terreau material"""
    hydro_model = CurveBasedHydraulicModel.from_library("terreau")
    if transport:
        transport_model = chloride_transport()
    return Material(hydraulic=hydro_model, Ks=Ks, porosity=porosity, transport=transport_model)

def till(Ks=9e-6):
    """Till material"""
    vg = VanGenuchtenParams(theta_r=0.08, theta_s=0.41, alpha=3.6, n=1.56)
    model = VanGenuchtenModel(vg)
    return Material(hydraulic=model, Ks=Ks)

def terreau(Ks=4e-5):
    """Terreau material"""
    vg = VanGenuchtenParams(theta_r=0.08, theta_s=0.41, alpha=3.6, n=1.56)
    model = VanGenuchtenModel(vg)
    return Material(hydraulic=model, Ks=Ks)

def sand(Ks=1e-4):
    """Sand material"""
    vg = VanGenuchtenParams(theta_r=0.045, theta_s=0.35, alpha=14.5, n=2.68)
    model = VanGenuchtenModel(vg)
    return Material(hydraulic=model, Ks=Ks)

def clay(Ks=1e-9):
    """Clay material"""
    vg = VanGenuchtenParams(theta_r=0.068, theta_s=0.38, alpha=0.8, n=1.09)
    model = VanGenuchtenModel(vg)
    return Material(hydraulic=model, Ks=Ks)