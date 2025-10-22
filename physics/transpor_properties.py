"""
Transport Properties for Solutes
Similar design pattern to SoilMaterial for consistency
Manages chemical-specific transport parameters
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

# ==========================================
# CONTAMINANT DATABASE
# ==========================================

@dataclass
class ContaminantData:
    """Standard transport properties for common contaminants"""
    name: str
    molecular_weight: float  # g/mol
    diffusion_coeff: float   # m²/s in free water at 25°C
    sorption_kd: float = 0.0  # L/kg (linear sorption coefficient)
    decay_rate: float = 0.0   # 1/s (first-order decay)
    charge: int = 0           # For ionic species
    
    @classmethod
    def get_chloride(cls):
        """Chloride (Cl⁻) - conservative tracer"""
        return cls(
            name="Chloride",
            molecular_weight=35.45,
            diffusion_coeff=2.03e-9,  # Well-established value
            sorption_kd=0.0,  # Conservative (no sorption)
            decay_rate=0.0,   # No decay
            charge=-1
        )
    
    @classmethod
    def get_sodium(cls):
        """Sodium (Na⁺)"""
        return cls(
            name="Sodium",
            molecular_weight=22.99,
            diffusion_coeff=1.33e-9,
            sorption_kd=0.5,  # Weak sorption via cation exchange
            decay_rate=0.0,
            charge=+1
        )
    
    @classmethod
    def get_calcium(cls):
        """Calcium (Ca²⁺)"""
        return cls(
            name="Calcium",
            molecular_weight=40.08,
            diffusion_coeff=0.79e-9,
            sorption_kd=2.0,  # Moderate sorption
            decay_rate=0.0,
            charge=+2
        )
    
    @classmethod
    def get_nitrate(cls):
        """Nitrate (NO₃⁻) - relevant for fertilizers"""
        return cls(
            name="Nitrate",
            molecular_weight=62.00,
            diffusion_coeff=1.90e-9,
            sorption_kd=0.0,  # Conservative
            decay_rate=1.0e-7,  # Slow denitrification
            charge=-1
        )


# ==========================================
# DISPERSIVITY MODELS
# ==========================================

class DispersivityModel:
    """
    Estimate dispersivity from spatial scale
    Based on Gelhar (1992) and Xu & Eckstein (1995)
    
    Key insight: αL increases with measurement scale
    """
    
    @staticmethod
    def gelhar_relation(length_scale: float) -> tuple:
        """
        Empirical relation: αL ≈ 0.1 × L^0.8 to 0.2 × L^0.9
        
        Parameters:
        -----------
        length_scale : float
            Characteristic length (m) - typically distance to observation point
        
        Returns:
        --------
        (alpha_L, alpha_T) : tuple of floats (m)
        """
        # Conservative estimate
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
            Distance from source (m)
        grain_size : float
            Median grain diameter (m)
        
        Returns:
        --------
        (alpha_L, alpha_T) : tuple
        """
        if distance < 1.0:
            # Near-field: grain-scale dispersion
            alpha_L = max(grain_size, 0.001)
        else:
            # Far-field: macrodispersion
            alpha_L = 0.1 * distance
        
        alpha_T = alpha_L / 10.0
        
        return alpha_L, alpha_T


# ==========================================
# TRANSPORT MATERIAL CLASS
# ==========================================

class TransportMaterial:
    """
    Complete transport description for a contaminant-soil pair
    Similar to SoilMaterial design pattern
    """
    
    def __init__(self, 
                 contaminant: ContaminantData,
                 alpha_L: float,
                 alpha_T: float,
                 bulk_density: float = 1600.0,
                 tortuosity_model: str = 'millington_quirk'):
        """
        Parameters:
        -----------
        contaminant : ContaminantData
            Chemical species to transport
        alpha_L : float
            Longitudinal dispersivity (m)
        alpha_T : float
            Transverse dispersivity (m)
        bulk_density : float
            Dry bulk density of soil (kg/m³)
        tortuosity_model : str
            'millington_quirk', 'bruggeman', or 'simple'
        """
        self.contaminant = contaminant
        self.alpha_L = alpha_L
        self.alpha_T = alpha_T
        self.rho_b = bulk_density
        self.tortuosity_model = tortuosity_model
    
    def effective_diffusion(self, porosity: float, saturation: float) -> float:
        """
        Calculate effective diffusion coefficient
        
        D_eff = D₀ × τ(θ)
        
        Parameters:
        -----------
        porosity : float (0-1)
        saturation : float (0-1)
        
        Returns:
        --------
        D_eff : float (m²/s)
        """
        theta = porosity * saturation
        
        if self.tortuosity_model == 'millington_quirk':
            tau = theta**(10.0/3.0) / (porosity**2)
        elif self.tortuosity_model == 'bruggeman':
            tau = theta**1.5
        else:  # simple
            tau = theta / porosity
        
        return self.contaminant.diffusion_coeff * tau
    
    def retardation_factor(self, porosity: float, saturation: float) -> float:
        """
        Calculate retardation factor (accounts for sorption)
        
        R = 1 + (ρb/θ) × Kd
        
        Where:
            ρb = bulk density (kg/m³)
            θ = water content
            Kd = sorption coefficient (L/kg = m³/kg × 10⁻³)
        
        R = 1 : conservative (no sorption)
        R > 1 : retarded (sorbs to solids)
        
        Returns:
        --------
        R : float (≥ 1)
        """
        theta = porosity * saturation
        
        if theta < 1e-6:  # Avoid division by zero
            return 1.0
        
        R = 1.0 + (self.rho_b / theta) * (self.contaminant.sorption_kd * 1e-3)
        
        return max(R, 1.0)
    
    def decay_coefficient(self) -> float:
        """
        First-order decay coefficient (1/s)
        For degradable contaminants
        """
        return self.contaminant.decay_rate
    
    @classmethod
    def from_contaminant(cls, name: str, alpha_L: float = None, 
                        alpha_T: float = None, **kwargs):
        """
        Create transport material from database contaminant
        
        Parameters:
        -----------
        name : str
            'chloride', 'sodium', 'calcium', 'nitrate'
        alpha_L, alpha_T : float
            Dispersivities (m). If None, use default values
        **kwargs : additional parameters for TransportMaterial
        
        Example:
        --------
        cl_transport = TransportMaterial.from_contaminant('chloride', alpha_L=0.05)
        """
        # Load contaminant data
        contaminant_map = {
            'chloride': ContaminantData.get_chloride,
            'sodium': ContaminantData.get_sodium,
            'calcium': ContaminantData.get_calcium,
            'nitrate': ContaminantData.get_nitrate
        }
        
        if name.lower() not in contaminant_map:
            raise ValueError(f"Unknown contaminant: {name}. Available: {list(contaminant_map.keys())}")
        
        contaminant = contaminant_map[name.lower()]()
        
        # Default dispersivities if not provided
        if alpha_L is None:
            alpha_L = 0.01  # 1 cm for near-field
        if alpha_T is None:
            alpha_T = alpha_L / 10.0
        
        return cls(contaminant, alpha_L, alpha_T, **kwargs)


# ==========================================
# MULTI-SPECIES TRANSPORT
# ==========================================

class MultiSpeciesTransport:
    """
    Manage transport of multiple contaminants simultaneously
    Useful for complete de-icing salt analysis (Cl⁻, Na⁺, Ca²⁺, Mg²⁺)
    """
    
    def __init__(self, species: list):
        """
        Parameters:
        -----------
        species : list of TransportMaterial
            All species to transport
        """
        self.species = species
        self.n_species = len(species)
    
    def get_species(self, name: str) -> Optional[TransportMaterial]:
        """Get specific species by name"""
        for sp in self.species:
            if sp.contaminant.name.lower() == name.lower():
                return sp
        return None
    
    def total_ionic_strength(self, concentrations: dict) -> float:
        """
        Calculate ionic strength (important for activity corrections)
        
        I = 0.5 × Σ(cᵢ × zᵢ²)
        
        Parameters:
        -----------
        concentrations : dict
            {species_name: concentration (mol/L)}
        
        Returns:
        --------
        ionic_strength : float (mol/L)
        """
        I = 0.0
        for sp in self.species:
            if sp.contaminant.name in concentrations:
                c = concentrations[sp.contaminant.name]
                z = sp.contaminant.charge
                I += c * z**2
        
        return 0.5 * I
    
    def check_electroneutrality(self, concentrations: dict, tolerance: float = 0.01):
        """
        Check charge balance (quality control)
        
        Σ(zᵢ × cᵢ) ≈ 0
        """
        charge_balance = 0.0
        for sp in self.species:
            if sp.contaminant.name in concentrations:
                c = concentrations[sp.contaminant.name]
                z = sp.contaminant.charge
                charge_balance += z * c
        
        if abs(charge_balance) > tolerance:
            print(f"WARNING: Charge imbalance = {charge_balance:.3e} mol/L")
        
        return abs(charge_balance) < tolerance


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRANSPORT PROPERTIES MODULE")
    print("=" * 60)
    
    # Simple chloride transport (conservative tracer)
    print("\n1. CHLORIDE (Conservative Tracer):")
    cl_mat = TransportMaterial.from_contaminant(
        'chloride',
        alpha_L=0.05,  # 5 cm
        alpha_T=0.005  # 5 mm
    )
    
    porosity = 0.35
    saturation = 1.0
    
    D_eff = cl_mat.effective_diffusion(porosity, saturation)
    R = cl_mat.retardation_factor(porosity, saturation)
    
    print(f"   Molecular diffusion: {cl_mat.contaminant.diffusion_coeff:.2e} m²/s")
    print(f"   Effective diffusion: {D_eff:.2e} m²/s")
    print(f"   Retardation factor: {R:.2f}")
    print(f"   Longitudinal dispersivity: {cl_mat.alpha_L} m")
    
    # Multi-species de-icing salt
    print("\n2. MULTI-SPECIES DE-ICING SALT:")
    species = [
        TransportMaterial.from_contaminant('chloride', alpha_L=0.05),
        TransportMaterial.from_contaminant('sodium', alpha_L=0.05),
        TransportMaterial.from_contaminant('calcium', alpha_L=0.05)
    ]
    
    multi = MultiSpeciesTransport(species)
    
    # Example concentrations after salt application
    conc = {
        'Chloride': 0.1,  # mol/L
        'Sodium': 0.08,   # mol/L
        'Calcium': 0.01   # mol/L
    }
    
    I = multi.total_ionic_strength(conc)
    balanced = multi.check_electroneutrality(conc)
    
    print(f"   Species: {[sp.contaminant.name for sp in multi.species]}")
    print(f"   Ionic strength: {I:.3f} mol/L")
    print(f"   Electroneutral: {balanced}")
    
    # Scale-dependent dispersivity
    print("\n3. SCALE-DEPENDENT DISPERSIVITY:")
    for distance in [0.1, 1.0, 10.0, 100.0]:
        alpha_L, alpha_T = DispersivityModel.gelhar_relation(distance)
        print(f"   Distance {distance:>5.1f}m → αL={alpha_L:.4f}m, αT={alpha_T:.5f}m")
    
    print("\n" + "=" * 60)