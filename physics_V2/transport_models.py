"""
Transport models for solute transport
Includes analytical and curve-based approaches
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional
from .curve_tools import CurveData, CurveInterpolator, TransportCurves

# ==============================================
# CONTAMINANT DATABASE
# ==============================================

@dataclass
class ContaminantData:
    """Physical/chemical properties of contaminants"""
    name: str
    molecular_weight: float  # g/mol
    diffusion_coeff: float   # m²/s in free water at 25°C
    sorption_kd: float = 0.0  # L/kg (linear sorption coefficient)
    decay_rate: float = 0.0   # 1/s (first-order decay)
    charge: int = 0           # For ionic species
    
    @classmethod
    def chloride(cls):
        """Chloride (Cl⁻) - conservative tracer"""
        return cls(
            name="Chloride",
            molecular_weight=35.45,
            diffusion_coeff=2.03e-9,
            sorption_kd=0.0,
            decay_rate=0.0,
            charge=-1
        )
    
    @classmethod
    def sodium(cls):
        """Sodium (Na⁺)"""
        return cls(
            name="Sodium",
            molecular_weight=22.99,
            diffusion_coeff=1.33e-9,
            sorption_kd=0.5,
            decay_rate=0.0,
            charge=+1
        )
    
    @classmethod
    def calcium(cls):
        """Calcium (Ca²⁺)"""
        return cls(
            name="Calcium",
            molecular_weight=40.08,
            diffusion_coeff=0.79e-9,
            sorption_kd=2.0,
            decay_rate=0.0,
            charge=+2
        )
    
    @classmethod
    def nitrate(cls):
        """Nitrate (NO₃⁻)"""
        return cls(
            name="Nitrate",
            molecular_weight=62.00,
            diffusion_coeff=1.90e-9,
            sorption_kd=0.0,
            decay_rate=1.0e-7,
            charge=-1
        )


# ==============================================
# DISPERSIVITY MODELS
# ==============================================

class DispersivityModel:
    """
    Estimate dispersivity from spatial scale
    Based on field observations and empirical relations
    """
    
    @staticmethod
    def gelhar_relation(length_scale: float) -> tuple:
        """
        Gelhar (1992) empirical relation
        αL ≈ 0.1 × L^0.83
        
        Parameters:
        -----------
        length_scale : float
            Characteristic length scale [m]
        
        Returns:
        --------
        (alpha_L, alpha_T) : tuple of floats [m]
        """
        alpha_L = 0.1 * (length_scale ** 0.83)
        alpha_T = alpha_L / 10.0  # Typical ratio
        return alpha_L, alpha_T
    
    @staticmethod
    def scale_dependent(distance: float, grain_size: float = 0.001) -> tuple:
        """
        Scale-dependent dispersivity
        
        Near-field (< 1m): αL ≈ grain size
        Far-field (> 1m): αL ≈ 0.1 × distance
        
        Parameters:
        -----------
        distance : float
            Distance from source [m]
        grain_size : float
            Median grain diameter [m]
        
        Returns:
        --------
        (alpha_L, alpha_T) : tuple [m]
        """
        if distance < 1.0:
            alpha_L = max(grain_size, 0.001)
        else:
            alpha_L = 0.1 * distance
        
        alpha_T = alpha_L / 10.0
        return alpha_L, alpha_T
    
    @staticmethod
    def xu_eckstein(length_scale: float, reliability: str = 'high') -> tuple:
        """
        Xu & Eckstein (1995) with reliability classes
        
        Parameters:
        -----------
        length_scale : float
            Observation scale [m]
        reliability : str
            'high': Well-controlled studies
            'medium': Field studies
            'low': Estimates
        
        Returns:
        --------
        (alpha_L, alpha_T) : tuple [m]
        """
        if reliability == 'high':
            alpha_L = 0.1 * (length_scale ** 0.9)
        elif reliability == 'medium':
            alpha_L = 0.2 * (length_scale ** 0.83)
        else:  # low
            alpha_L = 0.3 * (length_scale ** 0.8)
        
        alpha_T = alpha_L / 10.0
        return alpha_L, alpha_T


# ==============================================
# TRANSPORT PROPERTIES
# ==============================================

@dataclass
class TransportProperties:
    """Transport property data container"""
    name: str
    D0: float                # Molecular diffusion [m²/s]
    alpha_L: float           # Longitudinal dispersivity [m]
    alpha_T: float           # Transverse dispersivity [m]
    Kd: float = 0.0          # Sorption coefficient [L/kg]
    decay_rate: float = 0.0  # Decay rate [1/s]
    charge: int = 0          # Ionic charge
    
    @classmethod
    def from_contaminant(cls, 
                        contaminant_name: str,
                        alpha_L: float = 0.01,
                        alpha_T: float = None):
        """
        Create from contaminant database
        
        Parameters:
        -----------
        contaminant_name : str
            'chloride', 'sodium', 'calcium', 'nitrate'
        alpha_L : float
            Longitudinal dispersivity [m]
        alpha_T : float, optional
            Transverse dispersivity [m] (default: alpha_L/10)
        
        Example:
        --------
        cl_props = TransportProperties.from_contaminant('chloride', alpha_L=0.05)
        """
        contaminant_map = {
            'chloride': ContaminantData.chloride,
            'sodium': ContaminantData.sodium,
            'calcium': ContaminantData.calcium,
            'nitrate': ContaminantData.nitrate
        }
        
        name_lower = contaminant_name.lower()
        if name_lower not in contaminant_map:
            raise ValueError(f"Unknown contaminant: {contaminant_name}. "
                           f"Available: {list(contaminant_map.keys())}")
        
        cont = contaminant_map[name_lower]()
        
        if alpha_T is None:
            alpha_T = alpha_L / 10.0
        
        return cls(
            name=cont.name,
            D0=cont.diffusion_coeff,
            alpha_L=alpha_L,
            alpha_T=alpha_T,
            Kd=cont.sorption_kd,
            decay_rate=cont.decay_rate,
            charge=cont.charge
        )


# ==============================================
# TRANSPORT MODELS
# ==============================================

class TransportModel:
    """
    Analytical transport model
    Computes effective properties from soil/water state
    """
    
    def __init__(self, 
                 transport_props: TransportProperties,
                 bulk_density: float = 1600.0,
                 tortuosity_model: str = 'millington_quirk'):
        """
        Parameters:
        -----------
        transport_props : TransportProperties
            Transport property data
        bulk_density : float
            Dry bulk density [kg/m³]
        tortuosity_model : str
            'millington_quirk', 'bruggeman', or 'simple'
        """
        self.props = transport_props
        self.rho_b = bulk_density
        self.tortuosity_model = tortuosity_model
    
    def tortuosity(self, porosity: float, saturation: float) -> float:
        """
        Calculate tortuosity factor τ
        
        Parameters:
        -----------
        porosity : float
            Total porosity [-]
        saturation : float
            Water saturation [-]
        
        Returns:
        --------
        tau : float
            Tortuosity factor [-]
        """
        theta = porosity * saturation
        
        if self.tortuosity_model == 'millington_quirk':
            # τ = θ^(10/3) / φ^2
            return theta**(10.0/3.0) / (porosity**2)
        elif self.tortuosity_model == 'bruggeman':
            # τ = θ^1.5
            return theta**1.5
        else:  # simple
            # τ = θ / φ
            return theta / porosity if porosity > 0 else 0.0
    
    def effective_diffusion(self, porosity: float, saturation: float) -> float:
        """
        Effective diffusion coefficient
        D_eff = D₀ × τ(θ)
        
        Returns:
        --------
        D_eff : float [m²/s]
        """
        tau = self.tortuosity(porosity, saturation)
        return self.props.D0 * tau
    
    def retardation_factor(self, porosity: float, saturation: float, **kwargs) -> float:
        """
        Retardation factor (accounts for sorption)
        R = 1 + (ρb/θ) × Kd
        
        kwargs can include pH, ionic_strength for advanced models
        
        Returns:
        --------
        R : float (≥ 1)
        """
        theta = porosity * saturation
        
        if theta < 1e-6:
            return 1.0
        
        Kd = self.props.Kd  # Can be overridden in derived classes
        R = 1.0 + (self.rho_b / theta) * (Kd * 1e-3)  # Convert L/kg to m³/kg
        
        return max(R, 1.0)
    
    def decay_coefficient(self) -> float:
        """First-order decay coefficient [1/s]"""
        return self.props.decay_rate
    
    def dispersion_coefficients(self, velocity: np.ndarray, 
                                porosity: float, saturation: float) -> tuple:
        """
        Dispersion tensor coefficients
        D_L = α_L |v| + D_eff
        D_T = α_T |v| + D_eff
        
        Parameters:
        -----------
        velocity : np.ndarray
            Darcy velocity vector [m/s]
        porosity : float
        saturation : float
        
        Returns:
        --------
        (D_L, D_T) : tuple [m²/s]
        """
        v_mag = np.linalg.norm(velocity)
        D_eff = self.effective_diffusion(porosity, saturation)
        
        D_L = self.props.alpha_L * v_mag + D_eff
        D_T = self.props.alpha_T * v_mag + D_eff
        
        return D_L, D_T


class CurveBasedTransportModel(TransportModel):
    """
    Transport model using empirical curves
    Can override analytical formulas with measured data
    """
    
    def __init__(self,
                 transport_props: TransportProperties,
                 D_eff_curve: Optional[CurveData] = None,
                 Kd_curve: Optional[CurveData] = None,
                 **kwargs):
        """
        Parameters:
        -----------
        transport_props : TransportProperties
        D_eff_curve : CurveData, optional
            Saturation → D_eff curve (overrides analytical)
        Kd_curve : CurveData, optional
            pH → Kd curve (for pH-dependent sorption)
        **kwargs : passed to TransportModel.__init__
        """
        super().__init__(transport_props, **kwargs)
        
        self.use_D_curve = D_eff_curve is not None
        self.use_Kd_curve = Kd_curve is not None
        
        if self.use_D_curve:
            self._D_interp = CurveInterpolator(D_eff_curve, bounds=(0, None))
        
        if self.use_Kd_curve:
            self._Kd_interp = CurveInterpolator(Kd_curve, bounds=(0, None))
    
    def effective_diffusion(self, porosity: float, saturation: float) -> float:
        """Override with curve if available"""
        if self.use_D_curve:
            return self._D_interp(saturation)
        else:
            return super().effective_diffusion(porosity, saturation)
    
    def retardation_factor(self, porosity: float, saturation: float, **kwargs) -> float:
        """Override Kd with curve if available (e.g., pH-dependent)"""
        theta = porosity * saturation
        
        if theta < 1e-6:
            return 1.0
        
        if self.use_Kd_curve:
            pH = kwargs.get('pH', 7.0)
            Kd = self._Kd_interp(pH)
        else:
            Kd = self.props.Kd
        
        R = 1.0 + (self.rho_b / theta) * (Kd * 1e-3)
        return max(R, 1.0)


# ==============================================
# MULTI-SPECIES TRANSPORT
# ==============================================

class MultiSpeciesTransport:
    """
    Manage transport of multiple contaminants simultaneously
    Useful for de-icing salt (Cl⁻, Na⁺, Ca²⁺)
    """
    
    def __init__(self, species: list):
        """
        Parameters:
        -----------
        species : list of TransportModel
            All species to simulate
        """
        self.species = species
        self.n_species = len(species)
    
    def get_species(self, name: str) -> Optional[TransportModel]:
        """Get specific species by name"""
        for sp in self.species:
            if sp.props.name.lower() == name.lower():
                return sp
        return None
    
    def total_ionic_strength(self, concentrations: dict) -> float:
        """
        Calculate ionic strength
        I = 0.5 × Σ(cᵢ × zᵢ²)
        
        Parameters:
        -----------
        concentrations : dict
            {species_name: concentration [mol/L]}
        
        Returns:
        --------
        I : float [mol/L]
        """
        I = 0.0
        for sp in self.species:
            if sp.props.name in concentrations:
                c = concentrations[sp.props.name]
                z = sp.props.charge
                I += c * z**2
        
        return 0.5 * I
    
    def check_electroneutrality(self, concentrations: dict, tolerance: float = 0.01) -> bool:
        """
        Check charge balance: Σ(zᵢ × cᵢ) ≈ 0
        
        Returns:
        --------
        balanced : bool
        """
        charge_balance = 0.0
        for sp in self.species:
            if sp.props.name in concentrations:
                c = concentrations[sp.props.name]
                z = sp.props.charge
                charge_balance += z * c
        
        if abs(charge_balance) > tolerance:
            print(f"WARNING: Charge imbalance = {charge_balance:.3e} mol/L")
        
        return abs(charge_balance) < tolerance