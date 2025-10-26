"""
Material definitions - Central hub linking all property models
Acts as interface between domain and constitutive models
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

from .hydraulic_models import (
    HydraulicModel, 
    VanGenuchtenModel, 
    VanGenuchtenParams,
    CurveBasedHydraulicModel
)
from .transport_models import (
    TransportModel,
    CurveBasedTransportModel,
    TransportProperties
)
from .curve_tools import HydraulicCurves

# ==============================================
# MATERIAL CLASS
# ==============================================

@dataclass
class Material:
    """
    Complete material definition
    
    This class acts as a "hub" that links:
    - Hydraulic properties and models (Van Genuchten or curves)
    - Transport properties and models (optional)
    - Other properties (thermal, geophysical) for future extension
    
    It delegates all calculations to the appropriate models.
    NO equations here - just a clean interface.
    """
    
    name: str
    
    # === HYDRAULIC PROPERTIES ===
    Ks: float                              # Saturated hydraulic conductivity [m/s]
    porosity: float                        # Total porosity [-]
    hydraulic_model: HydraulicModel        # VG or Curve-based
    
    # === TRANSPORT PROPERTIES (optional) ===
    transport_model: Optional[TransportModel] = None
    
    # === OTHER PROPERTIES (for future extensions) ===
    thermal_conductivity: Optional[float] = None      # [W/m/K]
    electrical_conductivity: Optional[float] = None   # [S/m]
    bulk_density: Optional[float] = 1600.0            # [kg/m³]
    
    # === METADATA ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate material definition"""
        if self.Ks <= 0:
            raise ValueError(f"Ks must be positive, got {self.Ks}")
        if not (0 < self.porosity <= 1):
            raise ValueError(f"Porosity must be in (0,1], got {self.porosity}")
    
    # =========================================
    # HYDRAULIC DELEGATION
    # =========================================
    
    def theta(self, pressure) -> float:
        """
        Water content at pressure head
        
        Parameters:
        -----------
        pressure : float or np.ndarray
            Pressure head [m]
        
        Returns:
        --------
        theta : float or np.ndarray
            Volumetric water content [m³/m³]
        """
        return self.hydraulic_model.theta(pressure)
    
    def kr(self, pressure) -> float:
        """
        Relative permeability at pressure head
        
        Parameters:
        -----------
        pressure : float or np.ndarray
            Pressure head [m]
        
        Returns:
        --------
        kr : float or np.ndarray
            Relative permeability [-]
        """
        return self.hydraulic_model.kr(pressure)
    
    def K(self, pressure) -> float:
        """
        Absolute hydraulic conductivity
        K = kr(p) × Ks
        
        Parameters:
        -----------
        pressure : float or np.ndarray
            Pressure head [m]
        
        Returns:
        --------
        K : float or np.ndarray
            Hydraulic conductivity [m/s]
        """
        return self.kr(pressure) * self.Ks
    
    def dtheta_dp(self, pressure) -> float:
        """
        Moisture capacity (specific moisture capacity)
        Cm = ∂θ/∂p
        
        Parameters:
        -----------
        pressure : float or np.ndarray
            Pressure head [m]
        
        Returns:
        --------
        Cm : float or np.ndarray
            Moisture capacity [1/m]
        """
        return self.hydraulic_model.dtheta_dp(pressure)
    
    def Se(self, pressure) -> float:
        """
        Effective saturation
        Se = (θ - θr) / (θs - θr)
        
        Parameters:
        -----------
        pressure : float or np.ndarray
            Pressure head [m]
        
        Returns:
        --------
        Se : float or np.ndarray
            Effective saturation [-]
        """
        return self.hydraulic_model.Se(pressure)
    
    @property
    def theta_r(self) -> float:
        """Residual water content [m³/m³]"""
        return self.hydraulic_model.theta_r
    
    @property
    def theta_s(self) -> float:
        """Saturated water content [m³/m³]"""
        return self.hydraulic_model.theta_s
    
    def compute_hydraulic_fields(self, pressure_field: np.ndarray) -> tuple:
        """
        Compute all hydraulic fields at once (optimized)
        
        Parameters:
        -----------
        pressure_field : np.ndarray
            Pressure head field [m]
        
        Returns:
        --------
        theta, Cm, kr : np.ndarray
            Water content, moisture capacity, relative permeability
        """
        return self.hydraulic_model.compute_fields(pressure_field)
    
    # =========================================
    # TRANSPORT DELEGATION
    # =========================================
    
    def D_eff(self, saturation: float) -> float:
        """
        Effective diffusion coefficient
        
        Parameters:
        -----------
        saturation : float
            Water saturation [-]
        
        Returns:
        --------
        D_eff : float
            Effective diffusion [m²/s]
        """
        if self.transport_model is None:
            raise ValueError(f"No transport model assigned to material '{self.name}'")
        return self.transport_model.effective_diffusion(self.porosity, saturation)
    
    def retardation(self, saturation: float, **kwargs) -> float:
        """
        Retardation factor (sorption effect)
        
        Parameters:
        -----------
        saturation : float
            Water saturation [-]
        **kwargs : optional
            pH, ionic_strength, etc. for advanced models
        
        Returns:
        --------
        R : float
            Retardation factor (≥ 1)
        """
        if self.transport_model is None:
            raise ValueError(f"No transport model assigned to material '{self.name}'")
        return self.transport_model.retardation_factor(self.porosity, saturation, **kwargs)
    
    def dispersion(self, velocity: np.ndarray, saturation: float) -> tuple:
        """
        Dispersion coefficients
        
        Parameters:
        -----------
        velocity : np.ndarray
            Darcy velocity vector [m/s]
        saturation : float
            Water saturation [-]
        
        Returns:
        --------
        (D_L, D_T) : tuple
            Longitudinal and transverse dispersion [m²/s]
        """
        if self.transport_model is None:
            raise ValueError(f"No transport model assigned to material '{self.name}'")
        return self.transport_model.dispersion_coefficients(velocity, self.porosity, saturation)
    
    @property
    def has_transport(self) -> bool:
        """Check if transport model is available"""
        return self.transport_model is not None
    
    # =========================================
    # FACTORY METHODS
    # =========================================
    
    @classmethod
    def from_curves(cls,
                    name: str,
                    curve_name: str,
                    Ks: float,
                    transport_props: Optional[TransportProperties] = None,
                    smooth_window: int = 1,
                    **kwargs):
        """
        Create material from predefined hydraulic curves
        
        Parameters:
        -----------
        name : str
            Material name (for identification)
        curve_name : str
            'till' or 'terreau' (predefined curves)
        Ks : float
            Saturated hydraulic conductivity [m/s]
        transport_props : TransportProperties, optional
            Transport properties (if None, no transport model)
        smooth_window : int
            Window size for curve smoothing (1 = no smoothing)
        **kwargs : additional Material parameters
        
        Example:
        --------
        till = Material.from_curves(
            name="Till Layer",
            curve_name="till",
            Ks=9e-6,
            transport_props=TransportProperties.from_contaminant('chloride')
        )
        """
        # Create hydraulic model from library
        hydraulic_model = CurveBasedHydraulicModel.from_library(
            curve_name, 
            smooth_window=smooth_window
        )
        
        # Infer porosity from saturated water content
        porosity = hydraulic_model.theta_s
        
        # Create transport model if requested
        transport_model = None
        if transport_props is not None:
            bulk_density = kwargs.pop('bulk_density', 1600.0)
            transport_model = TransportModel(transport_props, bulk_density=bulk_density)
        
        return cls(
            name=name,
            Ks=Ks,
            porosity=porosity,
            hydraulic_model=hydraulic_model,
            transport_model=transport_model,
            **kwargs
        )
    
    @classmethod
    def from_van_genuchten(cls,
                          name: str,
                          Ks: float,
                          vg_params: VanGenuchtenParams,
                          porosity: Optional[float] = None,
                          transport_props: Optional[TransportProperties] = None,
                          **kwargs):
        """
        Create material from Van Genuchten parameters
        
        Parameters:
        -----------
        name : str
            Material name
        Ks : float
            Saturated hydraulic conductivity [m/s]
        vg_params : VanGenuchtenParams
            Van Genuchten parameters
        porosity : float, optional
            Total porosity (default: vg_params.theta_s)
        transport_props : TransportProperties, optional
            Transport properties
        **kwargs : additional Material parameters
        
        Example:
        --------
        sand = Material.from_van_genuchten(
            name="Clean Sand",
            Ks=1e-4,
            vg_params=VanGenuchtenParams(
                theta_r=0.045,
                theta_s=0.35,
                alpha=14.5,
                n=2.68
            )
        )
        """
        # Create hydraulic model
        epsilon = kwargs.pop('epsilon', 0.01)
        hydraulic_model = VanGenuchtenModel(vg_params, epsilon=epsilon)
        
        # Infer porosity if not provided
        if porosity is None:
            porosity = vg_params.theta_s
        
        # Create transport model if requested
        transport_model = None
        if transport_props is not None:
            bulk_density = kwargs.pop('bulk_density', 1600.0)
            transport_model = TransportModel(transport_props, bulk_density=bulk_density)
        
        return cls(
            name=name,
            Ks=Ks,
            porosity=porosity,
            hydraulic_model=hydraulic_model,
            transport_model=transport_model,
            **kwargs
        )
    
    @classmethod
    def from_custom_curves(cls,
                          name: str,
                          Ks: float,
                          pressure_heads: np.ndarray,
                          theta_values: np.ndarray,
                          kr_values: np.ndarray,
                          porosity: Optional[float] = None,
                          transport_props: Optional[TransportProperties] = None,
                          smooth_window: int = 1,
                          **kwargs):
        """
        Create material from custom measured curves
        
        Parameters:
        -----------
        name : str
            Material name
        Ks : float
            Saturated hydraulic conductivity [m/s]
        pressure_heads : array-like
            Measured pressure heads [m]
        theta_values : array-like
            Measured water contents [m³/m³]
        kr_values : array-like
            Measured relative permeabilities [-]
        porosity : float, optional
            Total porosity (default: max(theta_values))
        transport_props : TransportProperties, optional
            Transport properties
        smooth_window : int
            Window size for smoothing
        **kwargs : additional Material parameters
        
        Example:
        --------
        custom_soil = Material.from_custom_curves(
            name="Lab Sample A",
            Ks=5e-6,
            pressure_heads=[-10, -5, -1, -0.1, 0],
            theta_values=[0.05, 0.10, 0.20, 0.35, 0.40],
            kr_values=[0.001, 0.01, 0.1, 0.8, 1.0]
        )
        """
        # Create hydraulic model
        hydraulic_model = CurveBasedHydraulicModel.from_data(
            pressure_heads=pressure_heads,
            theta_values=theta_values,
            kr_values=kr_values,
            smooth_window=smooth_window
        )
        
        # Infer porosity
        if porosity is None:
            porosity = hydraulic_model.theta_s
        
        # Create transport model if requested
        transport_model = None
        if transport_props is not None:
            bulk_density = kwargs.pop('bulk_density', 1600.0)
            transport_model = TransportModel(transport_props, bulk_density=bulk_density)
        
        return cls(
            name=name,
            Ks=Ks,
            porosity=porosity,
            hydraulic_model=hydraulic_model,
            transport_model=transport_model,
            **kwargs
        )
    
    # =========================================
    # STANDARD MATERIALS LIBRARY
    # =========================================
    
    @classmethod
    def library_till(cls, transport_contaminant: Optional[str] = None, **kwargs):
        """
        Standard Till material from library
        
        Parameters:
        -----------
        transport_contaminant : str, optional
            'chloride', 'sodium', 'calcium', 'nitrate'
        **kwargs : override default properties
        
        Example:
        --------
        till = Material.library_till(transport_contaminant='chloride')
        """
        transport_props = None
        if transport_contaminant is not None:
            alpha_L = kwargs.pop('alpha_L', 0.01)
            transport_props = TransportProperties.from_contaminant(
                transport_contaminant, alpha_L=alpha_L
            )
        
        return cls.from_curves(
            name="Till",
            curve_name="till",
            Ks=kwargs.pop('Ks', 9e-6),
            transport_props=transport_props,
            **kwargs
        )
    
    @classmethod
    def library_terreau(cls, transport_contaminant: Optional[str] = None, **kwargs):
        """
        Standard Terreau (engineered soil) from library
        
        Parameters:
        -----------
        transport_contaminant : str, optional
            'chloride', 'sodium', 'calcium', 'nitrate'
        **kwargs : override default properties
        
        Example:
        --------
        terreau = Material.library_terreau(transport_contaminant='chloride')
        """
        transport_props = None
        if transport_contaminant is not None:
            alpha_L = kwargs.pop('alpha_L', 0.02)
            transport_props = TransportProperties.from_contaminant(
                transport_contaminant, alpha_L=alpha_L
            )
        
        return cls.from_curves(
            name="Terreau",
            curve_name="terreau",
            Ks=kwargs.pop('Ks', 4e-5),
            transport_props=transport_props,
            **kwargs
        )
    
    @classmethod
    def library_sand(cls, transport_contaminant: Optional[str] = None, **kwargs):
        """
        Standard sand (Van Genuchten parameters)
        
        Example:
        --------
        sand = Material.library_sand()
        """
        vg_params = VanGenuchtenParams(
            theta_r=kwargs.pop('theta_r', 0.045),
            theta_s=kwargs.pop('theta_s', 0.35),
            alpha=kwargs.pop('alpha', 14.5),
            n=kwargs.pop('n', 2.68)
        )
        
        transport_props = None
        if transport_contaminant is not None:
            alpha_L = kwargs.pop('alpha_L', 0.05)
            transport_props = TransportProperties.from_contaminant(
                transport_contaminant, alpha_L=alpha_L
            )
        
        return cls.from_van_genuchten(
            name="Sand",
            Ks=kwargs.pop('Ks', 1e-4),
            vg_params=vg_params,
            transport_props=transport_props,
            **kwargs
        )
    
    @classmethod
    def library_clay(cls, transport_contaminant: Optional[str] = None, **kwargs):
        """
        Standard clay (Van Genuchten parameters)
        
        Example:
        --------
        clay = Material.library_clay()
        """
        vg_params = VanGenuchtenParams(
            theta_r=kwargs.pop('theta_r', 0.068),
            theta_s=kwargs.pop('theta_s', 0.38),
            alpha=kwargs.pop('alpha', 0.8),
            n=kwargs.pop('n', 1.09)
        )
        
        transport_props = None
        if transport_contaminant is not None:
            alpha_L = kwargs.pop('alpha_L', 0.005)
            transport_props = TransportProperties.from_contaminant(
                transport_contaminant, alpha_L=alpha_L
            )
        
        return cls.from_van_genuchten(
            name="Clay",
            Ks=kwargs.pop('Ks', 1e-9),
            vg_params=vg_params,
            transport_props=transport_props,
            **kwargs
        )
    
    # =========================================
    # UTILITY METHODS
    # =========================================
    
    def __repr__(self):
        """Clean string representation"""
        transport_str = f", transport={self.transport_model.props.name}" if self.has_transport else ""
        return f"Material('{self.name}', Ks={self.Ks:.2e}, φ={self.porosity:.3f}{transport_str})"
    
    def summary(self) -> str:
        """Detailed material summary"""
        lines = [
            f"Material: {self.name}",
            f"{'='*60}",
            f"HYDRAULIC PROPERTIES:",
            f"  Saturated conductivity (Ks): {self.Ks:.2e} m/s",
            f"  Porosity (φ): {self.porosity:.3f}",
            f"  Residual water content (θr): {self.theta_r:.3f}",
            f"  Saturated water content (θs): {self.theta_s:.3f}",
            f"  Model type: {type(self.hydraulic_model).__name__}",
        ]
        
        if self.has_transport:
            lines.extend([
                f"\nTRANSPORT PROPERTIES:",
                f"  Contaminant: {self.transport_model.props.name}",
                f"  Molecular diffusion (D0): {self.transport_model.props.D0:.2e} m²/s",
                f"  Longitudinal dispersivity (αL): {self.transport_model.props.alpha_L:.4f} m",
                f"  Transverse dispersivity (αT): {self.transport_model.props.alpha_T:.4f} m",
                f"  Sorption coefficient (Kd): {self.transport_model.props.Kd:.3f} L/kg",
                f"  Decay rate: {self.transport_model.props.decay_rate:.2e} 1/s",
            ])
        
        if self.bulk_density:
            lines.append(f"\nBulk density: {self.bulk_density:.0f} kg/m³")
        
        if self.metadata:
            lines.append(f"\nMetadata: {self.metadata}")
        
        return "\n".join(lines)