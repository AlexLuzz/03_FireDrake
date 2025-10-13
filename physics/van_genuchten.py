"""
Van Genuchten model for soil hydraulic properties
Defines pressure-saturation relationships
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

# ==========================================
# ABSTRACT BASE CLASS
# ==========================================

class HydraulicModel(ABC):
    """
    Abstract base class for hydraulic models
    Defines the interface that all hydraulic models must implement
    """
    
    @property
    @abstractmethod
    def theta_r(self) -> float:
        """Residual water content"""
        pass
    
    @property
    @abstractmethod
    def theta_s(self) -> float:
        """Saturated water content"""
        pass
    
    @abstractmethod
    def water_content(self, Hp: float) -> float:
        """Calculate volumetric water content at pressure head Hp"""
        pass
    
    @abstractmethod
    def moisture_capacity(self, Hp: float) -> float:
        """Calculate specific moisture capacity Cm = ∂θ/∂Hp"""
        pass
    
    @abstractmethod
    def relative_permeability(self, Hp: float) -> float:
        """Calculate relative permeability kr"""
        pass
    
    def compute_fields(self, pressure_array: np.ndarray) -> tuple:
        """
        Vectorized computation of Cm and kr fields
        Default implementation (can be overridden for performance)
        """
        Cm = np.array([self.moisture_capacity(p) for p in pressure_array])
        kr = np.array([self.relative_permeability(p) for p in pressure_array])
        return Cm, kr


# ==========================================
# VAN GENUCHTEN ANALYTICAL MODEL
# ==========================================

@dataclass
class VanGenuchtenParams:
    """Van Genuchten model parameters"""
    theta_r: float = 0.05   # Residual water content
    theta_s: float = 0.40   # Saturated water content
    alpha: float = 2.0      # van Genuchten alpha (1/m)
    n: float = 2.0          # van Genuchten n parameter
    l_param: float = 0.5    # Pore connectivity parameter
    
    @property
    def m(self):
        """van Genuchten m parameter (dependent on n)"""
        return 1.0 - 1.0 / self.n


class VanGenuchtenModel(HydraulicModel):
    """
    Van Genuchten analytical model for hydraulic properties
    Uses closed-form equations
    """
    
    def __init__(self, params: VanGenuchtenParams, epsilon: float = 0.01, 
                 kr_min: float = 1e-8, Ss: float = 1e-4):
        """
        Initialize van Genuchten model
        
        Args:
            params: Van Genuchten parameters
            epsilon: Smoothing parameter near water table (m)
            kr_min: Minimum relative permeability
            Ss: Specific storage coefficient (1/m)
        """
        self.params = params
        self.epsilon = epsilon
        self.kr_min = kr_min
        self.Ss = Ss
    
    @property
    def theta_r(self) -> float:
        return self.params.theta_r
    
    @property
    def theta_s(self) -> float:
        return self.params.theta_s
    
    def effective_saturation(self, Hp: float) -> float:
        """Calculate effective saturation Se from pressure head"""
        if Hp >= self.epsilon:
            return 1.0
        elif Hp <= -self.epsilon:
            return 1.0 / (1.0 + abs(self.params.alpha * Hp)**self.params.n)**self.params.m
        else:
            # Smooth transition zone
            Se_neg = 1.0 / (1.0 + abs(self.params.alpha * (-self.epsilon))**self.params.n)**self.params.m
            weight = (Hp + self.epsilon) / (2.0 * self.epsilon)
            return Se_neg + (1.0 - Se_neg) * weight
    
    def water_content(self, Hp: float) -> float:
        """Calculate volumetric water content θ from pressure head"""
        Se = self.effective_saturation(Hp)
        return self.params.theta_r + Se * (self.params.theta_s - self.params.theta_r)
    
    def moisture_capacity(self, Hp: float) -> float:
        """Calculate specific moisture capacity Cm = ∂θ/∂Hp"""
        if Hp >= self.epsilon:
            return self.Ss
        elif Hp <= -self.epsilon:
            Se = self.effective_saturation(Hp)
            Cm_val = ((self.params.alpha * self.params.m) / (1.0 - self.params.m) * 
                     (self.params.theta_s - self.params.theta_r) * 
                     Se**(1.0/self.params.m) * 
                     (1.0 - Se**(1.0/self.params.m))**self.params.m)
            return max(self.Ss, Cm_val)
        else:
            # Smooth transition zone
            Cm_sat = self.Ss
            Se_unsat = self.effective_saturation(-self.epsilon)
            Cm_unsat = ((self.params.alpha * self.params.m) / (1.0 - self.params.m) * 
                       (self.params.theta_s - self.params.theta_r) * 
                       Se_unsat**(1.0/self.params.m) * 
                       (1.0 - Se_unsat**(1.0/self.params.m))**self.params.m)
            weight = (Hp + self.epsilon) / (2.0 * self.epsilon)
            return Cm_unsat + (Cm_sat - Cm_unsat) * weight
    
    def relative_permeability(self, Hp: float) -> float:
        """Calculate relative permeability kr using Mualem model"""
        if Hp >= self.epsilon:
            return 1.0
        elif Hp <= -self.epsilon:
            Se = self.effective_saturation(Hp)
            kr_val = Se**self.params.l_param * (1.0 - (1.0 - Se**(1.0/self.params.m))**self.params.m)**2
            return max(self.kr_min, min(kr_val, 1.0))
        else:
            # Smooth transition zone
            kr_neg = self.relative_permeability(-self.epsilon)
            weight = (Hp + self.epsilon) / (2.0 * self.epsilon)
            return kr_neg + (1.0 - kr_neg) * weight


# ==========================================
# CURVE-BASED EMPIRICAL MODEL
# ==========================================

class CurveBasedModel(HydraulicModel):
    """
    Hydraulic model based on empirical curve data
    Uses linear interpolation between measured points
    """
    
    def __init__(self, pressure_heads, theta_values, kr_values, theta_r, theta_s, 
                 epsilon=0.01, Ss=1e-4, kr_min=1e-8):
        from scipy.interpolate import interp1d
        
        self._theta_r = theta_r
        self._theta_s = theta_s
        self.epsilon = epsilon
        self.Ss = Ss
        self.kr_min = kr_min
        
        # Use simple linear interpolation
        self.theta_interp = interp1d(pressure_heads, theta_values, 
                                     kind='linear', fill_value='extrapolate', 
                                     bounds_error=False)
        self.kr_interp = interp1d(pressure_heads, kr_values, 
                                  kind='linear', fill_value='extrapolate', 
                                  bounds_error=False)
        
        # Store range for reference
        self.p_min = pressure_heads[0]
        self.p_max = pressure_heads[-1]
    
    @property
    def theta_r(self) -> float:
        return self._theta_r
    
    @property
    def theta_s(self) -> float:
        return self._theta_s
    
    def water_content(self, Hp: float) -> float:
        """Get water content at pressure head Hp via interpolation"""
        theta = float(self.theta_interp(Hp))
        return np.clip(theta, self._theta_r, self._theta_s)
    
    def moisture_capacity(self, Hp: float) -> float:
        """Calculate moisture capacity via numerical derivative"""
        if Hp >= 0:
            return self.Ss
        else:
            # Numerical derivative
            dh = 1e-4
            theta1 = self.water_content(Hp - dh)
            theta2 = self.water_content(Hp + dh)
            Cm = abs((theta2 - theta1) / (2 * dh))
            return max(self.Ss, Cm)
    
    def relative_permeability(self, Hp: float) -> float:
        """Get relative permeability at pressure head Hp via interpolation"""
        if Hp >= 0:
            return 1.0
        else:
            kr = float(self.kr_interp(Hp))
            return np.clip(kr, self.kr_min, 1.0)