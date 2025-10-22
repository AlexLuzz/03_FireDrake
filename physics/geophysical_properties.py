"""
Waxman-Smits Model for Montreal Till
Simplified for ERT-based calibration with clay-rich soils

Key workflow:
1. Richards solver → saturation field
2. Transport solver → chloride concentration field  
3. This module: saturation + chloride → predict bulk resistivity
4. Compare with measured ERT → calibrate uncertain parameters
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import minimize

# ==========================================
# WAXMAN-SMITS PARAMETERS
# ==========================================

@dataclass
class WaxmanSmitsParams:
    """
    Waxman-Smits parameters for clay-rich Montreal till
    
    Key equation: σ_bulk = (S^n / F) × (σ_w + B×Qv×S)
    
    Where:
        σ_bulk = bulk electrical conductivity (S/m)
        σ_w = pore fluid conductivity (S/m) - from chloride
        S = water saturation (0-1) - from Richards
        F = formation factor = a / φ^m
        φ = porosity (0-1)
        a = tortuosity factor (0.5-2.5, typically ~1 for till)
        m = cementation exponent (1.3-2.5, typically ~1.5-2 for till)
        n = saturation exponent (1.5-2.5, typically ~2)
        B = equivalent conductance of Na+ counterions (S·m²/meq)
        Qv = cation exchange capacity per pore volume (meq/mL)
    
    UNCERTAIN PARAMETERS (fit from ERT time-series):
        - a, m, n: Formation factor parameters
        - B × Qv: Combined clay conductivity term
    
    MEASURED/KNOWN:
        - φ: From soil samples or initial ERT inversion
        - σ_w: From chloride transport model + water samples
        - S: From Richards equation solver
    """
    # Formation factor parameters (UNCERTAIN - calibrate!)
    a: float = 1.0      # Tortuosity factor
    m: float = 1.8      # Cementation exponent (lower for till vs sandstone)
    n: float = 2.0      # Saturation exponent
    
    # Clay parameters (UNCERTAIN - calibrate!)
    B: float = 4.6e-8   # Na+ equivalent conductance (S·m²/meq)
    Qv: float = 0.15    # CEC per pore volume (meq/mL)
                        # For 15% clay with CEC~30 meq/100g, φ~0.35: Qv ≈ 0.1-0.2
    
    def get_clay_conductivity_term(self) -> float:
        """Get combined B×Qv term (easier to calibrate as single parameter)"""
        return self.B * self.Qv


# ==========================================
# WAXMAN-SMITS MODEL
# ==========================================

class WaxmanSmits:
    """
    Waxman-Smits petrophysical model for Montreal till
    Links hydraulic state → electrical resistivity
    """
    
    def __init__(self, params: Optional[WaxmanSmitsParams] = None):
        self.params = params or WaxmanSmitsParams()
    
    def formation_factor(self, porosity: float) -> float:
        """
        Calculate formation factor: F = a / φ^m
        
        Physical meaning: How much pore geometry restricts current flow
        Higher F → more resistive
        """
        return self.params.a / (porosity ** self.params.m)
    
    def fluid_conductivity_from_chloride(self, 
                                        cl_concentration: float,
                                        temperature: float = 25.0,
                                        background_sigma: float = 0.01) -> float:
        """
        Calculate pore fluid conductivity from chloride concentration
        
        Parameters:
        -----------
        cl_concentration : float
            Chloride concentration (mg/L)
        temperature : float
            Temperature (°C)
        background_sigma : float
            Background water conductivity (S/m) - from measurements
            Montreal groundwater: typically 0.01-0.05 S/m
        
        Returns:
        --------
        sigma_w : float (S/m)
        
        Notes:
        ------
        Empirical relation for NaCl: Δσ ≈ 10.4 × ΔC (S/m per mol/L)
        This assumes NaCl is dominant (valid for road salt)
        """
        # Convert mg/L to mol/L (Cl⁻ molecular weight ≈ 35.5 g/mol)
        molarity = cl_concentration / 35500.0
        
        # Temperature correction factor
        temp_factor = 1 + 0.02 * (temperature - 25.0)
        
        # Conductivity from chloride + background
        sigma_chloride = 10.4 * molarity * temp_factor
        sigma_w = background_sigma + sigma_chloride
        
        return sigma_w
    
    def bulk_conductivity(self,
                         porosity: float,
                         saturation: float,
                         fluid_conductivity: float) -> float:
        """
        Calculate bulk electrical conductivity using Waxman-Smits
        
        FORWARD MODEL: (φ, S, σ_w) → σ_bulk
        
        Equation: σ_bulk = (S^n / F) × (σ_w + B×Qv×S)
                         = (S^n / F) × σ_w + (S^(n+1) / F) × B×Qv
                           ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
                           Archie term            Clay surface term
        
        Parameters:
        -----------
        porosity : float (0-1)
        saturation : float (0-1)
        fluid_conductivity : float (S/m)
        
        Returns:
        --------
        sigma_bulk : float (S/m)
        """
        F = self.formation_factor(porosity)
        clay_term = self.params.get_clay_conductivity_term()
        
        # Waxman-Smits equation
        sigma_bulk = (saturation ** self.params.n / F) * \
                     (fluid_conductivity + clay_term * saturation)
        
        return sigma_bulk
    
    def bulk_resistivity(self, porosity: float, saturation: float,
                        fluid_conductivity: float) -> float:
        """
        Calculate bulk resistivity (Ω·m) from Waxman-Smits
        
        ρ_bulk = 1 / σ_bulk
        """
        sigma = self.bulk_conductivity(porosity, saturation, fluid_conductivity)
        return 1.0 / sigma
    
    def predict_resistivity_field(self,
                                 porosity_field: np.ndarray,
                                 saturation_field: np.ndarray,
                                 chloride_field: np.ndarray,
                                 temperature: float = 25.0,
                                 background_sigma: float = 0.01) -> np.ndarray:
        """
        Predict resistivity field for entire domain
        USE THIS to compare model predictions with ERT measurements
        
        Parameters:
        -----------
        porosity_field : np.ndarray
            Porosity at each node (from soil characterization)
        saturation_field : np.ndarray
            Saturation at each node (from Richards solver)
        chloride_field : np.ndarray
            Chloride concentration at each node (from transport solver, mg/L)
        temperature : float
            Soil temperature (°C)
        background_sigma : float
            Background water conductivity (S/m)
        
        Returns:
        --------
        resistivity_field : np.ndarray (Ω·m)
        """
        n_nodes = len(porosity_field)
        resistivity = np.zeros(n_nodes)
        
        for i in range(n_nodes):
            # Get fluid conductivity from chloride
            sigma_w = self.fluid_conductivity_from_chloride(
                chloride_field[i], temperature, background_sigma
            )
            
            # Compute bulk resistivity
            resistivity[i] = self.bulk_resistivity(
                porosity_field[i],
                saturation_field[i],
                sigma_w
            )
        
        return resistivity


# ==========================================
# CALIBRATION FROM ERT TIME-SERIES
# ==========================================

class ERTCalibrator:
    """
    Calibrate Waxman-Smits parameters using time-lapse ERT data
    
    Strategy:
    1. Use time-series at fixed locations
    2. Fit uncertain parameters (a, m, n, B×Qv) to minimize misfit
    3. Constraints: reasonable parameter ranges based on literature
    """
    
    def __init__(self, model: WaxmanSmits):
        self.model = model
    
    def calibrate(self,
                 measured_resistivity: np.ndarray,  # (n_times, n_locations)
                 porosity: np.ndarray,              # (n_locations,)
                 saturation: np.ndarray,            # (n_times, n_locations)
                 chloride: np.ndarray,              # (n_times, n_locations) mg/L
                 background_sigma: float = 0.01,
                 bounds: dict = None) -> Tuple[WaxmanSmitsParams, dict]:
        """
        Calibrate Waxman-Smits parameters from ERT time-series
        
        Parameters:
        -----------
        measured_resistivity : np.ndarray (n_times, n_locations)
            Measured resistivity from ERT (Ω·m)
        porosity : np.ndarray (n_locations,)
            Porosity at measurement locations (assumed known)
        saturation : np.ndarray (n_times, n_locations)
            Saturation from Richards model
        chloride : np.ndarray (n_times, n_locations)
            Chloride from transport model (mg/L)
        background_sigma : float
            Background water conductivity (S/m)
        bounds : dict
            Parameter bounds: {'a': (min, max), 'm': (min, max), ...}
        
        Returns:
        --------
        calibrated_params : WaxmanSmitsParams
        results : dict with fit statistics
        
        Example:
        --------
        # After running Richards + Transport for several timesteps
        calibrator = ERTCalibrator(ws_model)
        
        optimal_params, stats = calibrator.calibrate(
            measured_resistivity=ert_data,  # Your ERT measurements
            porosity=porosity_field,
            saturation=saturation_timeseries,
            chloride=chloride_timeseries
        )
        
        print(f"Optimal a={optimal_params.a:.3f}, m={optimal_params.m:.3f}")
        print(f"RMSE={stats['rmse']:.2f} Ω·m")
        """
        # Default bounds (literature values for till)
        if bounds is None:
            bounds = {
                'a': (0.5, 2.5),      # Tortuosity factor
                'm': (1.3, 2.5),      # Cementation exponent
                'n': (1.5, 2.5),      # Saturation exponent
                'BQv': (1e-9, 1e-7)   # Combined clay term (S/m)
            }
        
        # Initial guess (current parameters)
        x0 = np.array([
            self.model.params.a,
            self.model.params.m,
            self.model.params.n,
            self.model.params.get_clay_conductivity_term()
        ])
        
        # Bounds for optimizer
        opt_bounds = [
            bounds['a'],
            bounds['m'],
            bounds['n'],
            bounds['BQv']
        ]
        
        # Objective function: minimize RMSE between predicted and measured
        def objective(x):
            a, m, n, BQv = x
            
            # Update model parameters
            self.model.params.a = a
            self.model.params.m = m
            self.model.params.n = n
            # Back-calculate B or Qv (keep B fixed, adjust Qv)
            self.model.params.Qv = BQv / self.model.params.B
            
            # Predict resistivity at all times and locations
            predicted = np.zeros_like(measured_resistivity)
            n_times, n_locs = measured_resistivity.shape
            
            for t in range(n_times):
                for loc in range(n_locs):
                    sigma_w = self.model.fluid_conductivity_from_chloride(
                        chloride[t, loc], background_sigma=background_sigma
                    )
                    predicted[t, loc] = self.model.bulk_resistivity(
                        porosity[loc],
                        saturation[t, loc],
                        sigma_w
                    )
            
            # Compute RMSE
            residuals = predicted - measured_resistivity
            rmse = np.sqrt(np.mean(residuals**2))
            
            return rmse
        
        # Optimize
        print("Calibrating Waxman-Smits parameters...")
        print(f"  Initial RMSE: {objective(x0):.2f} Ω·m")
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=opt_bounds,
            options={'maxiter': 100}
        )
        
        # Extract optimal parameters
        a_opt, m_opt, n_opt, BQv_opt = result.x
        optimal_params = WaxmanSmitsParams(
            a=a_opt,
            m=m_opt,
            n=n_opt,
            B=self.model.params.B,
            Qv=BQv_opt / self.model.params.B
        )
        
        # Compute final statistics
        final_rmse = result.fun
        
        print(f"  Optimized RMSE: {final_rmse:.2f} Ω·m")
        print(f"  Optimal parameters:")
        print(f"    a = {a_opt:.3f}")
        print(f"    m = {m_opt:.3f}")
        print(f"    n = {n_opt:.3f}")
        print(f"    B×Qv = {BQv_opt:.2e} S/m")
        
        return optimal_params, {
            'rmse': final_rmse,
            'success': result.success,
            'message': result.message,
            'iterations': result.nit
        }