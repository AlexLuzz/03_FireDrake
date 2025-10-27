from dataclasses import dataclass
from typing import Optional
from .hydraulic_models import HydraulicModel, CurveBasedHydraulicModel, VanGenuchtenModel, VanGenuchtenParams
from .transport_models import TransportModel

@dataclass
class Material:
    """Material container - cleaner than tuples!"""
    hydraulic: HydraulicModel
    Ks: float # Saturated hydraulic conductivity [m/s]
    transport: Optional[TransportModel] = None
    
    def __repr__(self):
        transport_str = f", transport={self.transport.props.name}" if self.transport else ""
        return f"Material(Ks={self.Ks:.2e}{transport_str})"


# Update factories to return Material objects
def till(Ks=9e-6):
    """Till material"""
    model = CurveBasedHydraulicModel.from_library("till")
    return Material(hydraulic=model, Ks=Ks)

def terreau(Ks=4e-5):
    """Terreau material"""
    model = CurveBasedHydraulicModel.from_library("terreau")
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