"""
Transport models for solute transport
Minimal ABC pattern like hydraulic_models.py
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from firedrake import function
# ==============================================
# ABSTRACT BASE CLASS
# ==============================================

class TransportModel(ABC):
    """
    Abstract base class for transport models
    All models must implement these methods
    """

    @abstractmethod
    def _R(self, theta, Kd) -> float:
        """R: Retardation factor [-]"""
        pass
    
    @abstractmethod
    def _tortuosity(self, porosity: float, saturation: float) -> float:
        """τ: Tortuosity factor [-]"""
        pass

    @abstractmethod
    def _D0(self, porosity: float, saturation: float) -> float:
        """D0: Molecular diffusion coefficient (including tortuosity) [m²/s]"""
        pass


# ==============================================
# CONTAMINANT PROPERTIES
# ==============================================

@dataclass
class ContaminantProperties:
    """Physical/chemical properties of contaminants"""
    name: str
    Dd: float               # Molecular diffusion [m²/s]
    Kd: float = 0.0         # Sorption coefficient [L/kg]
    alpha_L: float = 1   # Longitudinal dispersivity [m]
    alpha_T: float = None   # Transverse dispersivity [m]
    lambda_: float = 0.0    # Degradation reaction rate coefficient [-]
    
    def __post_init__(self):
        if self.alpha_T is None:
            self.alpha_T = self.alpha_L / 10.0
    
    @classmethod
    def chloride(cls, alpha_L: float = 0.01):
        """Cl⁻ - conservative tracer"""
        return cls(name="Chloride", Dd=2.03e-7, Kd=0.0, alpha_L=alpha_L)
    
    @classmethod
    def sodium(cls, alpha_L: float = 0.01):
        """Na⁺"""
        return cls(name="Sodium", Dd=1.33e-9, Kd=0.5, alpha_L=alpha_L)
    
    @classmethod
    def calcium(cls, alpha_L: float = 0.01):
        """Ca²⁺"""
        return cls(name="Calcium", Dd=0.79e-9, Kd=2.0, alpha_L=alpha_L)


# ==============================================
# ANALYTICAL TRANSPORT MODEL
# ==============================================

class AnalyticalTransportModel(TransportModel):
    """
    Analytical transport model with Millington-Quirk tortuosity
    """
    
    def __init__(self, 
                 properties: ContaminantProperties,
                 bulk_density: float = 1600.0,
                 tortuosity_model: str = 'millington_quirk'):
        """
        Parameters:
        -----------
        properties : ContaminantProperties
        bulk_density : float [kg/m³]
        tortuosity_model : str
            'millington_quirk', 'bruggeman', or 'simple'
        """
        self.props = properties
        self.rho_b = bulk_density
        self.tortuosity_model = tortuosity_model

    def _R(self, theta: float) -> float:
        """R = 1 + (ρb/θ) × Kd [-]"""
        if theta < 1e-6:
            return 1.0
        
        R = 1.0 + (self.rho_b / theta) * (self.props.Kd * 1e-3)  # L/kg to m³/kg
        return max(R, 1.0)

    def _tortuosity(self, theta: float, porosity: float) -> float:
        """τ: Tortuosity factor [-]"""
        if self.tortuosity_model == 'millington_quirk':
            return theta**(10.0/3.0) / (porosity**2) if porosity > 0 else 0.0
        elif self.tortuosity_model == 'bruggeman':
            return theta**1.5
        else:  # simple
            return theta / porosity if porosity > 0 else 0.0

    def _D0(self, theta: float, porosity: float) -> float:
        """D_0 = Dd × τ(θ) [m²/s]"""
        tau = self._tortuosity(theta, porosity)
        return self.props.Dd * tau


# ==============================================
# MATERIAL FACTORIES
# ==============================================

def chloride_transport(alpha_L: float = 0.01, bulk_density: float = 1600.0):
    """Conservative tracer (Cl⁻)"""
    props = ContaminantProperties.chloride(alpha_L)
    return AnalyticalTransportModel(props, bulk_density)

def sodium_transport(alpha_L: float = 0.01, bulk_density: float = 1600.0):
    """Sodium (Na⁺) with sorption"""
    props = ContaminantProperties.sodium(alpha_L)
    return AnalyticalTransportModel(props, bulk_density)

def calcium_transport(alpha_L: float = 0.01, bulk_density: float = 1600.0):
    """Calcium (Ca²⁺) with sorption"""
    props = ContaminantProperties.calcium(alpha_L)
    return AnalyticalTransportModel(props, bulk_density)