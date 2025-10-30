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
    K_sat: float                    # Saturated hydraulic conductivity [m/s]
    porosity: float                 # Porosity [-]
    theta_r: float = 0.01          # Residual water content [-]
    
    # Transport properties (optional)
    bulk_density: Optional[float] = 1600.0  # Bulk density [kg/m³]
    tortuosity_model: str = 'millington_quirk'
    
    # Physical properties
    thermal_conductivity: Optional[float] = None  # [W/m/K]
    specific_heat: Optional[float] = None         # [J/kg/K]
    
    @classmethod
    def till(cls):
        """Till soil (glacial deposit)"""
        return cls(
            name="Till",
            K_sat=1e-6,          # Very low permeability
            porosity=0.14,
            theta_r=0.011,
            bulk_density=2200.0   # Dense material
        )
    
    @classmethod
    def terreau(cls):
        """Terreau (potting soil/compost)"""
        return cls(
            name="Terreau",
            K_sat=1e-5,          # Medium permeability
            porosity=0.43,
            theta_r=0.017,
            bulk_density=1200.0   # Light material
        )
    
    @classmethod
    def sand(cls):
        """Sandy soil"""
        return cls(
            name="Sand",
            K_sat=1e-4,          # High permeability
            porosity=0.35,
            theta_r=0.05,
            bulk_density=1600.0
        )
    
    @classmethod
    def clay(cls):
        """Clay soil"""
        return cls(
            name="Clay",
            K_sat=1e-8,          # Very low permeability
            porosity=0.45,
            theta_r=0.10,
            bulk_density=1400.0
        )
    
    @classmethod
    def loam(cls):
        """Loam soil"""
        return cls(
            name="Loam",
            K_sat=1e-5,
            porosity=0.40,
            theta_r=0.08,
            bulk_density=1500.0
        )
