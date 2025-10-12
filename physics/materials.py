"""
Soil material properties
Bundles hydraulic model with material-specific parameters
"""
from dataclasses import dataclass
from physics.van_genuchten import VanGenuchtenModel, VanGenuchtenParams
import numpy as np

@dataclass
class SoilMaterial:
    """
    Complete soil material description
    Combines hydraulic model with saturated conductivity
    """
    name: str
    hydraulic_model: VanGenuchtenModel
    Ks: float  # Saturated hydraulic conductivity (m/s)
    
    @property
    def porosity(self):
        """Total porosity (same as saturated water content)"""
        return self.hydraulic_model.params.theta_s
    
    @property
    def theta_s(self):
        """Saturated water content"""
        return self.hydraulic_model.params.theta_s
    
    @property
    def theta_r(self):
        """Residual water content"""
        return self.hydraulic_model.params.theta_r
    
    # Delegate hydraulic property calculations to the model
    def water_content(self, Hp: float) -> float:
        """Calculate water content at given pressure head"""
        return self.hydraulic_model.water_content(Hp)
    
    def moisture_capacity(self, Hp: float) -> float:
        """Calculate moisture capacity at given pressure head"""
        return self.hydraulic_model.moisture_capacity(Hp)
    
    def relative_permeability(self, Hp: float) -> float:
        """Calculate relative permeability at given pressure head"""
        return self.hydraulic_model.relative_permeability(Hp)
    
    def hydraulic_conductivity(self, Hp: float) -> float:
        """Calculate absolute hydraulic conductivity K = kr * Ks"""
        return self.relative_permeability(Hp) * self.Ks
    
    def effective_saturation(self, Hp: float) -> float:
        """Calculate effective saturation at given pressure head"""
        return self.hydraulic_model.effective_saturation(Hp)
    
    def compute_fields(self, pressure_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Cm, kr, and K fields for array of pressures
        
        Returns:
            Tuple of (Cm_array, kr_array, K_array)
        """
        Cm, kr = self.hydraulic_model.compute_fields(pressure_array)
        K = kr * self.Ks
        return Cm, kr, K
    
    @classmethod
    def create_default(cls, epsilon: float = 0.01, kr_min: float = 1e-8, Ss: float = 1e-4):
        """Create default soil material matching original simulation"""
        vg_params = VanGenuchtenParams(
            theta_r=0.05,
            theta_s=0.40,
            alpha=2.0,
            n=2.0,
            l_param=0.5
        )
        vg_model = VanGenuchtenModel(vg_params, epsilon=epsilon, kr_min=kr_min, Ss=Ss)
        return cls(name="Default Soil", hydraulic_model=vg_model, Ks=1e-5)
    
    @classmethod
    def create_sandy_loam(cls, epsilon: float = 0.01, kr_min: float = 1e-8, Ss: float = 1e-4):
        """Create sandy loam soil"""
        vg_params = VanGenuchtenParams.sandy_loam()
        vg_model = VanGenuchtenModel(vg_params, epsilon=epsilon, kr_min=kr_min, Ss=Ss)
        return cls(name="Sandy Loam", hydraulic_model=vg_model, Ks=4.42e-6)
    
    @classmethod
    def create_clay(cls, epsilon: float = 0.01, kr_min: float = 1e-8, Ss: float = 1e-4):
        """Create clay soil"""
        vg_params = VanGenuchtenParams.clay()
        vg_model = VanGenuchtenModel(vg_params, epsilon=epsilon, kr_min=kr_min, Ss=Ss)
        return cls(name="Clay", hydraulic_model=vg_model, Ks=1.67e-7)