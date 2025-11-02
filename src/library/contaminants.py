"""
Contaminant properties library
Physical and chemical properties of various contaminants
"""
from dataclasses import dataclass


@dataclass
class ContaminantProperties:
    """Physical/chemical properties of contaminants"""
    name: str
    Dd: float               # Molecular diffusion [m²/s]
    Kd: float = 0.0         # Sorption coefficient [L/kg]
    alpha_L: float = 1      # Longitudinal dispersivity [m]
    alpha_T: float = None   # Transverse dispersivity [m]
    lambda_: float = 0.0    # Degradation reaction rate coefficient [-]
    
    def __post_init__(self):
        if self.alpha_T is None:
            self.alpha_T = self.alpha_L / 10.0
    
    @classmethod
    def chloride(cls, alpha_L: float = 1):
        """Cl⁻ - conservative tracer"""
        return cls(name="Chloride", Dd=2.03e-9, Kd=0.0, alpha_L=alpha_L)
    
    @classmethod
    def chloride_test(cls, alpha_L: float = 1):
        """Cl⁻ - conservative tracer"""
        return cls(name="Chloride test", 
                   Dd=2.03e-9,      # Higher diffusion for testing (faster spreading)
                   Kd=0.0,          # Conservative, no sorption
                   lambda_=0.0,     # Non-reactive, no degradation
                   alpha_L=alpha_L)
    
    @classmethod
    def sodium(cls, alpha_L: float = 0.01):
        """Na⁺"""
        return cls(name="Sodium", Dd=1.33e-9, Kd=0.5, alpha_L=alpha_L)
    
    @classmethod
    def calcium(cls, alpha_L: float = 0.01):
        """Ca²⁺"""
        return cls(name="Calcium", Dd=0.79e-9, Kd=2.0, alpha_L=alpha_L)
