"""
Geophysical models - Archie's law only (simple start)
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ArchieParams:
    """Archie's law parameters"""
    a: float = 1.0      # Tortuosity factor
    m: float = 2.0      # Cementation exponent
    n: float = 2.0      # Saturation exponent
    
    @classmethod
    def sand(cls):
        return cls(a=1.0, m=2.0, n=2.0)
    
    @classmethod
    def consolidated(cls):
        return cls(a=0.62, m=2.15, n=2.0)
    
    @classmethod
    def till(cls):
        return cls(a=1.0, m=2.2, n=2.0)


def archie_resistivity(porosity: float, 
                       saturation: float, 
                       fluid_resistivity: float,
                       params: Optional[ArchieParams] = None) -> float:
    """
    Archie's law: ρ_bulk = a × φ^(-m) × S^(-n) × ρ_fluid
    
    Parameters:
    -----------
    porosity : float [-]
    saturation : float [-]
    fluid_resistivity : float [Ω·m]
    params : ArchieParams (default: sand)
    
    Returns:
    --------
    bulk_resistivity : float [Ω·m]
    """
    if params is None:
        params = ArchieParams.sand()
    
    if porosity < 1e-6 or saturation < 1e-6:
        return 1e6
    
    formation_factor = params.a * porosity**(-params.m)
    rho_bulk = formation_factor * saturation**(-params.n) * fluid_resistivity
    
    return rho_bulk


def fluid_resistivity_from_concentration(concentration: float,
                                        temperature: float = 25.0) -> float:
    """
    Estimate fluid resistivity from NaCl concentration
    
    Parameters:
    -----------
    concentration : float [mg/L]
    temperature : float [°C]
    
    Returns:
    --------
    resistivity : float [Ω·m]
    """
    # σ ≈ 0.011e-3 × C [S/m] for NaCl
    sigma_25 = 0.011e-3 * concentration
    
    # Temperature correction (2% per °C)
    sigma_T = sigma_25 * (1 + 0.02 * (temperature - 25.0))
    
    return 1.0 / sigma_T if sigma_T > 0 else 1e6