"""
Soil parameter bundles
Complete soil definitions with hydraulic and transport properties
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SoilParameters:
    """
    Complete bundle of soil parameters for hydraulic and transport modeling
    """
    name: str
    
    # Hydraulic properties
    Ks: float                    # Saturated hydraulic conductivity [m/s]
    porosity: float                 # Porosity [-]
    theta_r: float = 0.01           # Residual water content [-]

    # Transport properties (optional)
    bulk_density: Optional[float] = 1600.0  # Bulk density [kg/m³]
    rho_b : Optional[float] = None          # Bulk density [kg/m³]
    
    # Physical properties
    
    def __post_init__(self):
        self.rho_b = self.bulk_density if self.bulk_density is not None else 1600.0

    @classmethod
    def till(cls):
        """Till soil (glacial deposit)"""
        return cls(
            name="Till",
            Ks=9e-6,          # Very low permeability
            porosity=0.4,
            theta_r=0.01,
            bulk_density=2200.0   # Dense material
        )
    
    @classmethod
    def terreau(cls):
        """Terreau (green infrastructure soil)"""
        return cls(
            name="Terreau",
            Ks=4e-5,          # Medium permeability
            porosity=0.3,
            theta_r=0.02,
            bulk_density=1200.0   # Light material
        )
    
    @classmethod
    def sand(cls):
        """Sandy soil"""
        return cls(
            name="Sand",
            Ks=1e-4,          # High permeability
            porosity=0.35,
            theta_r=0.02,
            bulk_density=1600.0
        )
    
    @classmethod
    def clay(cls):
        """Clay soil"""
        return cls(
            name="Clay",
            Ks=1e-8,          # Very low permeability
            porosity=0.45,
            theta_r=0.10,
            bulk_density=1400.0
        )
    
    @classmethod
    def loam(cls):
        """Loam soil"""
        return cls(
            name="Loam",
            Ks=1e-5,
            porosity=0.40,
            theta_r=0.08,
            bulk_density=1500.0
        )
