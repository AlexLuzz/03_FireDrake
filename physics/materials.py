# physics/materials.py
from dataclasses import dataclass
import numpy as np
from .van_genuchten import HydraulicModel, VanGenuchtenModel, CurveBasedModel
from scipy.interpolate import interp1d

# ==========================================
# DEFAULT SOIL CURVE DATA
# ==========================================

@dataclass
class SoilCurveData:
    """Standard soil hydraulic curve data"""
    name: str
    pressure_heads: list  # Pressure head (m)
    theta_values: list    # Water content (m³/m³)
    kr_values: list       # Relative permeability (-)
    Ks: float            # Saturated hydraulic conductivity (m/s)
    
    @classmethod
    def get_till(cls):
        # Original data in MPa, convert to meters of head
        pressure_MPa = [-9.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.050, -0.020, -0.010, 0.000]
        pressure_m = [p * 1e6 / 9810 for p in pressure_MPa]  # Convert MPa to meters
        
        return cls(
            name="Till",
            pressure_heads=pressure_m,
            theta_values=[0.011, 0.014, 0.019, 0.023, 0.027, 0.034, 0.039, 0.045, 0.059, 0.083, 0.140],
            kr_values=[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.010, 0.010, 0.070, 1.000],
            Ks=1e-7
        )
    
    @classmethod
    def get_terreau(cls):
        """Engineered soil mix for green infrastructure"""
        # Original data in MPa, convert to meters of head
        pressure_MPa = [-9.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.050, -0.020, -0.010, 0.000]
        pressure_m = [p * 1e6 / 9810 for p in pressure_MPa]  # Convert MPa to meters
        
        return cls(
            name="Terreau",
            pressure_heads=pressure_m,
            theta_values=[0.017, 0.021, 0.028, 0.035, 0.042, 0.052, 0.061, 0.076, 0.138, 0.283, 0.430],
            kr_values=[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.010, 0.100, 1.000],
            Ks=5e-6
        )

@dataclass
class SoilMaterial:
    """
    Complete description of a soil material
    Bundles hydraulic model with material properties
    """
    name: str
    hydraulic_model: HydraulicModel  # Can be VanGenuchtenModel or CurveBasedModel
    Ks: float  # Saturated hydraulic conductivity (m/s)
    porosity: float = None
    
    def __post_init__(self):
        if self.porosity is None:
            self.porosity = self.hydraulic_model.theta_s
    
    # Delegate hydraulic functions to the model
    def water_content(self, Hp: float) -> float:
        return self.hydraulic_model.water_content(Hp)
    
    def moisture_capacity(self, Hp: float) -> float:
        return self.hydraulic_model.moisture_capacity(Hp)
    
    def relative_permeability(self, Hp: float) -> float:
        return self.hydraulic_model.relative_permeability(Hp)
    
    def hydraulic_conductivity(self, Hp: float) -> float:
        """Absolute hydraulic conductivity K = kr * Ks"""
        return self.relative_permeability(Hp) * self.Ks
    
    def compute_fields(self, pressure_array):
        """Compute Cm, kr, K for array of pressures"""
        Cm, kr = self.hydraulic_model.compute_fields(pressure_array)
        K = kr * self.Ks
        return Cm, kr, K
    
    @classmethod
    def from_curves(cls, name: str, pressure_heads: list = None, theta_values: list = None, 
                    kr_values: list = None, Ks: float = None, theta_r: float = None, 
                    theta_s: float = None, smooth_window: int = 1):
        """
        Create material from empirical curve data with interpolation
        
        Args:
            name: Material name ("Till" or "Terreau" for defaults, or custom name)
            pressure_heads: List of pressure head values (m, negative for unsaturated)
            theta_values: List of water content values (m³/m³)
            kr_values: List of relative permeability values (0-1)
            Ks: Saturated hydraulic conductivity (m/s)
            theta_r: Residual water content (inferred if None)
            theta_s: Saturated water content (inferred if None)
            smooth_window: Window size for moving average smoothing (1 = no smoothing)
        
        Example:
            # Use default curves (no smoothing by default)
            till = SoilMaterial.from_curves(name="Till")
            terreau = SoilMaterial.from_curves(name="Terreau")
            
            # Custom material
            custom = SoilMaterial.from_curves(
                name="Custom Soil",
                pressure_heads=[...],
                theta_values=[...],
                kr_values=[...],
                Ks=1e-6
            )
        """
        # Load default curve data if available
        if pressure_heads is None or theta_values is None or kr_values is None or Ks is None:
            if name.lower() == "till":
                curve_data = SoilCurveData.get_till()
            elif name.lower() == "terreau":
                curve_data = SoilCurveData.get_terreau()
            else:
                raise ValueError(f"No default curves for '{name}'. Please provide pressure_heads, theta_values, kr_values, and Ks.")
            
            pressure_heads = curve_data.pressure_heads
            theta_values = curve_data.theta_values
            kr_values = curve_data.kr_values
            Ks = curve_data.Ks
        
        # Convert to numpy arrays and sort by pressure head
        pressure_heads = np.array(pressure_heads, dtype=float)
        theta_values = np.array(theta_values, dtype=float)
        kr_values = np.array(kr_values, dtype=float)
        
        sort_idx = np.argsort(pressure_heads)
        pressure_heads = pressure_heads[sort_idx]
        theta_values = theta_values[sort_idx]
        kr_values = kr_values[sort_idx]
        
        # Apply moving average smoothing only if requested
        if smooth_window > 1:
            theta_values = moving_average(theta_values, smooth_window)
            kr_values = moving_average(kr_values, smooth_window)
        
        # Infer theta_r and theta_s if not provided
        if theta_r is None:
            theta_r = theta_values[0]  # Minimum value
        if theta_s is None:
            theta_s = theta_values[-1]  # Maximum value
        
        # Create interpolation-based model
        hydraulic_model = CurveBasedModel(
            pressure_heads, theta_values, kr_values, theta_r, theta_s
        )
        
        return cls(name=name, hydraulic_model=hydraulic_model, Ks=Ks)


def moving_average(data, window_size):
    """Apply moving average smoothing"""
    if window_size <= 1:
        return data
    
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed[i] = np.mean(data[start:end])
    
    return smoothed


class CurveBasedModel:
    """
    Hydraulic model based on empirical curve data
    Uses interpolation for property evaluation
    """
    
    def __init__(self, pressure_heads, theta_values, kr_values, theta_r, theta_s, 
                 epsilon=0.01, Ss=1e-4, kr_min=0.001):
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.epsilon = epsilon
        self.Ss = Ss
        self.kr_min = kr_min
        
        # Use simple linear interpolation (already smoothed by moving_average)
        self.theta_interp = interp1d(pressure_heads, theta_values, 
                                     kind='linear', fill_value='extrapolate', 
                                     bounds_error=False)
        self.kr_interp = interp1d(pressure_heads, kr_values, 
                                  kind='linear', fill_value='extrapolate', 
                                  bounds_error=False)
        
        # Store range for extrapolation handling
        self.p_min = pressure_heads[0]
        self.p_max = pressure_heads[-1]
    
    def water_content(self, Hp: float) -> float:
        """Get water content at pressure head Hp"""
        theta = float(self.theta_interp(Hp))
        return np.clip(theta, self.theta_r, self.theta_s)
    
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
        """Get relative permeability at pressure head Hp"""
        if Hp >= 0:
            return 1.0
        else:
            kr = float(self.kr_interp(Hp))
            return np.clip(kr, self.kr_min, 1.0)
    
    def compute_fields(self, pressure_array):
        """Vectorized computation"""
        Cm = np.array([self.moisture_capacity(p) for p in pressure_array])
        kr = np.array([self.relative_permeability(p) for p in pressure_array])
        return Cm, kr


# ==========================================
# VISUALIZATION FUNCTION
# ==========================================

def visualize_soil_properties(materials, pressure_range=(-10, 0.1), n_points=200, 
                              save_path='soil_hydraulic_curves.png'):
    """
    Visualize hydraulic properties for multiple soil materials
    
    Args:
        materials: List of SoilMaterial objects or single SoilMaterial
        pressure_range: Tuple of (min, max) pressure head values (m)
        n_points: Number of points for curve plotting
        save_path: Path to save figure (None to skip saving)
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Ensure materials is a list
    if not isinstance(materials, list):
        materials = [materials]
    
    # Create pressure head range for plotting
    Hp_range = np.linspace(pressure_range[0], pressure_range[1], n_points)
    
    # Colors for different materials
    colors = ['#8B4513', '#2E8B57', '#4169E1', '#DC143C', '#FF8C00']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, mat in enumerate(materials):
        color = colors[i % len(colors)]
        
        # Get original curve data for scatter points
        curve_data = None
        if mat.name.lower() == "till":
            curve_data = SoilCurveData.get_till()
        elif mat.name.lower() == "terreau":
            curve_data = SoilCurveData.get_terreau()
        
        # Water content curve (θ vs Hp)
        theta = np.array([mat.water_content(p) for p in Hp_range])
        axes[0].plot(Hp_range, theta, label=mat.name, color=color, linewidth=2)
        if curve_data:
            axes[0].scatter(curve_data.pressure_heads, curve_data.theta_values,
                          color=color, s=50, alpha=0.6, zorder=5)
        
        # Relative permeability curve (kr vs Hp)
        kr = np.array([mat.relative_permeability(p) for p in Hp_range])
        axes[1].plot(Hp_range, kr, label=mat.name, color=color, linewidth=2)
        if curve_data:
            axes[1].scatter(curve_data.pressure_heads, curve_data.kr_values,
                          color=color, s=50, alpha=0.6, zorder=5)
        
        # Effective saturation curve (Se vs Hp)
        Se = np.array([(mat.water_content(p) - mat.hydraulic_model.theta_r) / 
                      (mat.hydraulic_model.theta_s - mat.hydraulic_model.theta_r) 
                      for p in Hp_range])
        axes[2].plot(Hp_range, Se, label=mat.name, color=color, linewidth=2)
    
    # Format theta plot
    axes[0].set_xlabel('Pressure Head Hp (m)', fontsize=11)
    axes[0].set_ylabel('Water Content θ (m³/m³)', fontsize=11)
    axes[0].set_title('Water Content Curve', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(pressure_range)
    axes[0].set_ylim([0, 0.5])  # Consistent scale for water content
    
    # Format kr plot (linear scale, not log)
    axes[1].set_xlabel('Pressure Head Hp (m)', fontsize=11)
    axes[1].set_ylabel('Relative Permeability kr (-)', fontsize=11)
    axes[1].set_title('Relative Permeability Curve', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(pressure_range)
    axes[1].set_ylim([0, 1.05])  # Linear scale 0 to 1
    
    # Format Se plot
    axes[2].set_xlabel('Pressure Head Hp (m)', fontsize=11)
    axes[2].set_ylabel('Effective Saturation Se (-)', fontsize=11)
    axes[2].set_title('Van Genuchten Curve (Se)', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(pressure_range)
    axes[2].set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved figure: {save_path}")
    
    return fig


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("SOIL MATERIAL DEFINITIONS FOR PROJECT")
    print("=" * 60)
    
    # Load default materials using simple syntax
    print("\nLoading default materials...")
    till = SoilMaterial.from_curves(name="Till")
    terreau = SoilMaterial.from_curves(name="Terreau")
    
    print(f"  ✓ Till: θs={till.porosity:.3f}, Ks={till.Ks:.2e} m/s")
    print(f"  ✓ Terreau: θs={terreau.porosity:.3f}, Ks={terreau.Ks:.2e} m/s")
    
    # Visualize hydraulic properties
    print("\nGenerating hydraulic property curves...")
    fig = visualize_soil_properties([till, terreau], pressure_range=(-10, 0.1))
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Material definitions complete!")
    print("=" * 60)