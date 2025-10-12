"""
Van Genuchten model for soil hydraulic properties
Defines pressure-saturation relationships
"""
from dataclasses import dataclass
import numpy as np

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
    
    @classmethod
    def sandy_loam(cls):
        """Factory for sandy loam soil"""
        return cls(theta_r=0.065, theta_s=0.41, alpha=2.8, n=1.56, l_param=0.5)
    
    @classmethod
    def clay(cls):
        """Factory for clay soil"""
        return cls(theta_r=0.098, theta_s=0.459, alpha=0.5, n=1.25, l_param=0.5)
    
    @classmethod
    def sand(cls):
        """Factory for sand soil"""
        return cls(theta_r=0.045, theta_s=0.43, alpha=3.5, n=2.68, l_param=0.5)


class VanGenuchtenModel:
    """
    Van Genuchten model for hydraulic properties
    Implements the relationships between pressure head and hydraulic properties
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
    
    def effective_saturation(self, Hp: float) -> float:
        """
        Calculate effective saturation Se from pressure head
        
        Args:
            Hp: Pressure head (m)
        
        Returns:
            Effective saturation (0 to 1)
        """
        if Hp >= self.epsilon:
            # Saturated zone
            return 1.0
        elif Hp <= -self.epsilon:
            # Unsaturated zone - van Genuchten formula
            return 1.0 / (1.0 + abs(self.params.alpha * Hp)**self.params.n)**self.params.m
        else:
            # Smooth transition zone
            Se_neg = 1.0 / (1.0 + abs(self.params.alpha * (-self.epsilon))**self.params.n)**self.params.m
            weight = (Hp + self.epsilon) / (2.0 * self.epsilon)
            return Se_neg + (1.0 - Se_neg) * weight
    
    def water_content(self, Hp: float) -> float:
        """
        Calculate volumetric water content θ from pressure head
        
        Args:
            Hp: Pressure head (m)
        
        Returns:
            Water content (m³/m³)
        """
        Se = self.effective_saturation(Hp)
        return self.params.theta_r + Se * (self.params.theta_s - self.params.theta_r)
    
    def moisture_capacity(self, Hp: float) -> float:
        """
        Calculate specific moisture capacity Cm = ∂θ/∂Hp
        
        Args:
            Hp: Pressure head (m)
        
        Returns:
            Moisture capacity (1/m)
        """
        if Hp >= self.epsilon:
            # Saturated zone: use specific storage
            return self.Ss
        elif Hp <= -self.epsilon:
            # Unsaturated zone: van Genuchten formula
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
        """
        Calculate relative permeability kr using Mualem model
        
        Args:
            Hp: Pressure head (m)
        
        Returns:
            Relative permeability (0 to 1)
        """
        if Hp >= self.epsilon:
            # Saturated zone
            return 1.0
        elif Hp <= -self.epsilon:
            # Unsaturated zone - Mualem model
            Se = self.effective_saturation(Hp)
            kr_val = Se**self.params.l_param * (1.0 - (1.0 - Se**(1.0/self.params.m))**self.params.m)**2
            return max(self.kr_min, min(kr_val, 1.0))
        else:
            # Smooth transition zone
            kr_neg = self.relative_permeability(-self.epsilon)
            weight = (Hp + self.epsilon) / (2.0 * self.epsilon)
            return kr_neg + (1.0 - kr_neg) * weight
    
    def compute_fields(self, pressure_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized computation of Cm and kr fields
        
        Args:
            pressure_array: Array of pressure head values
        
        Returns:
            Tuple of (Cm_array, kr_array)
        """
        Cm = np.array([self.moisture_capacity(p) for p in pressure_array])
        kr = np.array([self.relative_permeability(p) for p in pressure_array])
        return Cm, kr