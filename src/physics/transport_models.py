"""
Transport models for solute transport
Minimal ABC pattern like hydraulic_models.py
"""
from abc import ABC, abstractmethod
from src.library.contaminants import ContaminantProperties

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
# ANALYTICAL TRANSPORT MODEL
# ==============================================

class AnalyticalTransportModel(TransportModel):
    """
    Analytical transport model with Millington-Quirk tortuosity
    """
    
    def __init__(self, 
                 properties: ContaminantProperties,
                 bulk_density: float,
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
