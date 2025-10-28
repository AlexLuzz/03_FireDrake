"""
Hydraulic models for soil water flow

PURPOSE:
========
These classes define the constitutive relationships between pressure head and:
- θ(p): Water content [m³/m³]
- kr(p): Relative permeability [-]
- Cm(p) = ∂θ/∂p: Moisture capacity [1/m]

Two model types are available:
1. **VanGenuchtenModel**: Analytical equations (Van Genuchten-Mualem)
   - Best for: Standard soils with known parameters
   - Advantages: Smooth, differentiable, widely validated
   - Disadvantages: Limited to Van Genuchten form, sometimes inappropriate curvatures
   
2. **CurveBasedHydraulicModel**: Empirical data interpolation
   - Best for: Custom soils with measured retention/permeability curves
   - Advantages: Can represent any measured behavior, allows specific curves


Both implement the same HydraulicModel interface, so they're interchangeable
in the simulation framework.

USAGE:
======
# Analytical model
vg_params = VanGenuchtenParams(theta_r=0.045, theta_s=0.35, alpha=14.5, n=2.68)
sand_model = VanGenuchtenModel(vg_params)

# Curve-based model from library
till_model = CurveBasedHydraulicModel.from_library("till")

# Curve-based model from custom data
custom_model = CurveBasedHydraulicModel.from_data(
    pressure_heads=[-10, -5, -1, 0],
    theta_values=[0.05, 0.15, 0.30, 0.40],
    kr_values=[0.001, 0.01, 0.1, 1.0]
)
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from .curve_tools import CurveData, CurveInterpolator, HydraulicCurves

# ==============================================
# ABSTRACT BASE CLASS
# ==============================================

class HydraulicModel(ABC):
    """
    Abstract base class for hydraulic models
    All models must implement these methods
    """
    
    @property
    @abstractmethod
    def theta_r(self) -> float:
        """Residual water content [m³/m³]"""
        pass
    
    @property
    @abstractmethod
    def theta_s(self) -> float:
        """Saturated water content [m³/m³]"""
        pass
    
    @abstractmethod
    def _theta(self, pressure) -> float:
        """Water content at pressure head [m³/m³]"""
        pass
    
    @abstractmethod
    def _k(self, pressure) -> float:
        """Relative permeability at pressure head [-]"""
        pass
    
    @abstractmethod
    def _Se(self, pressure) -> float:
        """Moisture capacity: ∂θ/∂p [1/m]"""
        pass

# ==============================================
# VAN GENUCHTEN ANALYTICAL MODEL
# ==============================================

@dataclass
class VanGenuchtenParams:
    """Van Genuchten model parameters"""
    theta_r: float      # Residual water content [m³/m³]
    theta_s: float      # Saturated water content [m³/m³]
    alpha: float        # van Genuchten alpha [1/m]
    n: float            # van Genuchten n parameter [-]
    l_param: float = 0.5  # Pore connectivity parameter [-]
    m: float = None     # Computed from n if None
    
    def __post_init__(self):
        """Compute m from n if not provided"""
        if self.m is None:
            self.m = 1.0 - 1.0 / self.n


class VanGenuchtenModel(HydraulicModel):
    """
    Van Genuchten analytical model
    With smooth transition near saturation (epsilon smoothing)
    """
    
    def __init__(self, 
                 params: VanGenuchtenParams, 
                 epsilon: float = 0.051,
                 kr_min: float = 1e-8,
                 Ss: float = 1e-4):
        """
        Parameters:
        -----------
        params : VanGenuchtenParams
            Van Genuchten parameters
        epsilon : float
            Smoothing parameter near water table [m]
            Creates smooth transition between saturated/unsaturated
        kr_min : float
            Minimum relative permeability (prevents division by zero)
        Ss : float
            Specific storage coefficient [1/m]
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
    
    def _Se(self, pressure: float) -> float:
        """
        S_e(p): Effective saturation [-]
        S_e = 1 / [1 + (α|p|)^n]^m
        """
        eps = self.epsilon
        alpha = self.params.alpha
        n = self.params.n
        m = self.params.m

        if pressure >= eps:
            return 1.0
        elif pressure <= -eps:
            Se = 1.0 / (1.0 + (alpha * abs(pressure))**n)**m
            return max(0.0, min(1.0, Se))  # clamp between 0 and 1
        else:
            Se_unsat = 1.0 / (1.0 + (alpha * eps)**n)**m # lower end (unsaturated)
            # Linear interpolation between unsaturated and saturated ends
            weight = (pressure + eps) / (2.0 * eps)
            Se = Se_unsat + (1.0 - Se_unsat) * weight
            return max(0.0, min(1.0, Se))
    
    def _theta(self, pressure) -> float:
        """
        θ(p): Water content via Van Genuchten equation
        θ(p) = θ_r + S_e(p) * (θ_s - θ_r)
        """
        theta_r = self.params.theta_r
        theta_s = self.params.theta_s
        Se = self._Se(pressure)
        return theta_r + Se * (theta_s - theta_r)

    def _kr(self, pressure) -> float:
        """
        Kr(p): Relative permeability [-] via Mualem model
        Kr(p) = S_e(p)^l * [1 - (1 - S_e(p)^(1/m))^m]^2
        """
        eps = self.epsilon
        l = self.params.l_param
        m = self.params.m
        if pressure >= eps:
            return 1.0
        elif pressure <= -eps:
            Se = self._Se(pressure)
            term = max(0.0, 1.0 - Se**(1.0/m))  # avoid tiny negatives
            kr_val = (Se**l) * (1.0 - term**m)**2
            return max(self.kr_min, min(kr_val, 1.0))
        else:
            # Compute unsaturated endpoint value
            Se_unsat = self._Se(-eps)
            term = max(0.0, 1.0 - Se_unsat**(1.0/m))
            kr_unsat = (Se_unsat**l) * (1.0 - term**m)**2
            kr_unsat = max(self.kr_min, min(kr_unsat, 1.0))

            # Linear interpolation between unsaturated and saturated states
            weight = (pressure + eps) / (2.0 * eps)
            kr_val = kr_unsat + (1.0 - kr_unsat) * weight
            return max(self.kr_min, min(kr_val, 1.0))
        
    def _k(self, pressure, Ks) -> float:
        """
        K(p): Hydraulic conductivity [m/s]
        K(p) = k_r(p) * K_s
        """
        return self._kr(pressure) * Ks

    def _Cm(self, pressure) -> float:
        """
        Cm(p): Moisture capacity [1/m]
        Cm(p) = alpha * m / (1 - m) * (θ_s - θ_r) * S_e^(1/m) * (1 - S_e^(1/m))^m
        """
        alpha = self.params.alpha
        m = self.params.m
        theta_s = self.params.theta_s
        theta_r = self.params.theta_r
        eps = self.epsilon
        if pressure >= eps:
            return self.Ss
        elif pressure <= -eps:
            Se = self._Se(pressure)
            term = max(0.0, 1.0 - Se**(1.0 / m))
            Cm_val = ((alpha * m) / (1.0 - m) *
                    (theta_s - theta_r) *
                    Se**(1.0 / m) *
                    term**m)
            return max(self.Ss, Cm_val)
        else:
            # Unsaturated endpoint at -eps
            Se_unsat = self._Se(-eps)
            term = max(0.0, 1.0 - Se_unsat**(1.0 / m))
            Cm_unsat = ((alpha * m) / (1.0 - m) *
                        (theta_s - theta_r) *
                        Se_unsat**(1.0 / m) *
                        term**m)
            Cm_unsat = max(self.Ss, Cm_unsat)

            # Linear interpolation between unsat and sat
            weight = (pressure + eps) / (2.0 * eps)
            Cm_val = Cm_unsat + (self.Ss - Cm_unsat) * weight
            return max(self.Ss, Cm_val)

    
# ==============================================
# CURVE-BASED EMPIRICAL MODEL
# ==============================================

class CurveBasedHydraulicModel(HydraulicModel):
    """
    Hydraulic model using empirical curve data
    Uses generic CurveInterpolator for flexibility
    """
    
    def __init__(self, 
                 theta_curve: CurveData,
                 kr_curve: CurveData,
                 theta_r: float = None,
                 theta_s: float = None,
                 epsilon: float = 0.005,
                 kr_min: float = 1e-8,
                 Ss: float = 1e-4,
                 ):
        """
        Parameters:
        -----------
        theta_curve : CurveData
            Pressure → water content curve
        kr_curve : CurveData
            Pressure → relative permeability curve
        theta_r, theta_s : float, optional
            If None, inferred from curve bounds
        epsilon : float
            Pressure threshold for saturation [m]
        Ss : float
            Specific storage coefficient [1/m]
        kr_min : float
            Minimum relative permeability
        """
        # Infer bounds if not provided
        self._theta_r = theta_r if theta_r is not None else theta_curve.y_min
        self._theta_s = theta_s if theta_s is not None else theta_curve.y_max
        
        self.epsilon = epsilon
        self.Ss = Ss
        self.kr_min = kr_min
        
        # Create interpolators with appropriate settings
        self._theta_interp = CurveInterpolator(
            theta_curve,
            method='linear',
            extrapolate_mode='clip',
            bounds=(self._theta_r, self._theta_s),
            fill_value_above=self._theta_s
        )
        
        self._kr_interp = CurveInterpolator(
            kr_curve,
            method='linear',
            extrapolate_mode='clip',
            bounds=(self.kr_min, 1.0),
            fill_value_above=1.0
        )
        
        self.theta_curve = theta_curve
        self.kr_curve = kr_curve
    
    @property
    def theta_r(self) -> float:
        return self._theta_r
    
    @property
    def theta_s(self) -> float:
        return self._theta_s

    def _theta(self, pressure) -> float:
        """Water content via interpolation"""
        if pressure >= self.epsilon:
            return self._theta_s
        return self._theta_interp(pressure)
    
    def _kr(self, pressure) -> float:
        """Relative permeability via interpolation"""
        if pressure >= self.epsilon:
            return 1.0
        return self._kr_interp(pressure)
    
    def _k(self, pressure, Ks) -> float:
        """Hydraulic conductivity"""
        return self._kr(pressure) * Ks

    def _Cm(self, pressure) -> float:
        """Moisture capacity"""
        Cm = self._theta_interp.derivative(pressure)
        return max(self.Ss, abs(Cm))

    def _Se(self, pressure) -> float:
        """Effective saturation via numerical derivative"""
        theta = self._theta(pressure)
        Se = (theta - self._theta_r) / (self._theta_s - self._theta_r)
        return max(0.0, min(1.0, Se))
    
    @classmethod
    def from_library(cls, soil_type: str, smooth_window: int = 1, **kwargs):
        """
        Create model from predefined library curves
        
        Parameters:
        -----------
        soil_type : str
            'till' or 'terreau'
        smooth_window : int
            Window size for smoothing (1 = no smoothing)
        **kwargs : additional parameters for __init__
        """
        soil_type = soil_type.lower()
        
        if soil_type == "till":
            theta_curve = HydraulicCurves.till_theta()
            kr_curve = HydraulicCurves.till_kr()
        elif soil_type == "terreau":
            theta_curve = HydraulicCurves.terreau_theta()
            kr_curve = HydraulicCurves.terreau_kr()
        else:
            raise ValueError(f"Unknown soil type: {soil_type}. Available: 'till', 'terreau'")
        
        if smooth_window > 1:
            theta_curve = theta_curve.smooth(smooth_window)
            kr_curve = kr_curve.smooth(smooth_window)
        
        return cls(theta_curve, kr_curve, **kwargs)
    
    @classmethod
    def from_data(cls, 
                  pressure_heads: np.ndarray,
                  theta_values: np.ndarray,
                  kr_values: np.ndarray,
                  smooth_window: int = 1,
                  **kwargs):
        """
        Create model from custom data arrays
        
        Example:
        --------
        model = CurveBasedHydraulicModel.from_data(
            pressure_heads=[-10, -5, -1, 0],
            theta_values=[0.05, 0.15, 0.30, 0.40],
            kr_values=[0.001, 0.01, 0.1, 1.0]
        )
        """
        theta_curve = CurveData(pressure_heads, theta_values, 
                               x_name="pressure", y_name="theta",
                               units_x="m", units_y="m³/m³")
        kr_curve = CurveData(pressure_heads, kr_values,
                            x_name="pressure", y_name="kr",
                            units_x="m", units_y="-")
        
        if smooth_window > 1:
            theta_curve = theta_curve.smooth(smooth_window)
            kr_curve = kr_curve.smooth(smooth_window)
        
        return cls(theta_curve, kr_curve, **kwargs)