"""
sensitivity_analysis.py

Two approaches to compute parameter sensitivities:
1. JAX - Fast, works on van Genuchten equations directly
2. FireDrake-adjoint - Works with full Richards solver
"""

import sys
from pathlib import Path

# Add parent directory to Python path so we can import from physics, solver, etc.
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from firedrake import *
from firedrake_adjoint import *

from physics import VanGenuchtenParams


# ============================================================
# 1. JAX SENSITIVITY (Fast, local analysis)
# ============================================================

class JaxSensitivity:
    """Compute sensitivities using JAX automatic differentiation"""
    
    @staticmethod
    @jax.jit
    def van_genuchten_theta(h: jnp.ndarray, alpha: float, n: float,
                            theta_s: float = 0.45, theta_r: float = 0.067) -> jnp.ndarray:
        """Van Genuchten water content (JAX version)"""
        m = 1.0 - 1.0 / n
        Se = jnp.where(
            h < 0,
            jnp.power(1.0 + jnp.power(alpha * jnp.abs(h), n), -m),
            1.0
        )
        return theta_r + (theta_s - theta_r) * Se
    
    def compute_sensitivities(self, h_values: np.ndarray, params: VanGenuchtenParams) -> dict:
        """
        Compute ∂θ/∂p for all parameters
        
        Parameters:
        -----------
        h_values : array
            Pressure head values to evaluate at
        params : VanGenuchtenParams
            Current parameter values
        
        Returns:
        --------
        sensitivities : dict
            {'alpha': dtheta_dalpha, 'n': dtheta_dn, ...}
        """
        h_jax = jnp.array(h_values)
        
        # Define functions for each parameter
        def theta_alpha(alpha):
            return self.van_genuchten_theta(h_jax, alpha, params.n, params.theta_s, params.theta_r)
        
        def theta_n(n):
            return self.van_genuchten_theta(h_jax, params.alpha, n, params.theta_s, params.theta_r)
        
        # Compute Jacobians (∂θ/∂p at each point)
        sens_alpha = np.array(jacfwd(theta_alpha)(params.alpha))
        sens_n = np.array(jacfwd(theta_n)(params.n))
        
        return {
            'alpha': sens_alpha,
            'n': sens_n
        }


# ============================================================
# 2. FIREDRAKE-ADJOINT SENSITIVITY (Full solver)
# ============================================================

class AdjointSensitivity:
    """Compute sensitivities using firedrake-adjoint"""
    
    def __init__(self, solver):
        """
        Parameters:
        -----------
        solver : RichardsSolver
            Your Richards equation solver
        """
        self.solver = solver
        self.tape = get_working_tape()
        self.tape.clear_tape()
    
    def compute_sensitivities(self, target_field: Function, 
                             control_params: dict) -> dict:
        """
        Compute sensitivities using adjoint method
        
        Parameters:
        -----------
        target_field : Function
            Field to compute sensitivity of (e.g., water table)
        control_params : dict
            {'alpha': Constant(...), 'n': Constant(...)}
            Parameters to compute sensitivity with respect to
        
        Returns:
        --------
        sensitivities : dict
            {param_name: sensitivity_value}
        
        Example:
        --------
        # In your solver, make parameters into Controls:
        alpha_control = Control(Constant(2.0))
        n_control = Control(Constant(1.5))
        
        # Run forward solve (this gets recorded on tape)
        solver.solve_timestep(dt)
        
        # Define functional (what to minimize)
        J = assemble(target_field**2 * dx)
        
        # Compute sensitivities
        sens = adjoint_sens.compute_sensitivities(
            target_field,
            {'alpha': alpha_control, 'n': n_control}
        )
        """
        # Define functional
        J = assemble(target_field**2 * dx)
        
        # Compute sensitivities for each control
        sensitivities = {}
        for name, control in control_params.items():
            # Compute dJ/d(param)
            dJdm = compute_gradient(J, control)
            sensitivities[name] = float(dJdm)
        
        return sensitivities
    
    def setup_controls(self, params_dict: dict) -> dict:
        """
        Helper to create Controls from parameter values
        
        Parameters:
        -----------
        params_dict : dict
            {'alpha': 2.0, 'n': 1.5}
        
        Returns:
        --------
        controls : dict
            {'alpha': Control(Constant(2.0)), ...}
        """
        controls = {}
        for name, value in params_dict.items():
            const = Constant(value)
            controls[name] = Control(const)
        return controls


# ============================================================
# COMPARISON & UTILITIES
# ============================================================

def compare_sensitivities(jax_sens: dict, adjoint_sens: dict, 
                         param_names: list) -> None:
    """
    Compare JAX vs Adjoint sensitivities
    
    Parameters:
    -----------
    jax_sens : dict
        From JaxSensitivity
    adjoint_sens : dict
        From AdjointSensitivity
    param_names : list
        Parameters to compare
    """
    print("\n" + "="*60)
    print("SENSITIVITY COMPARISON")
    print("="*60)
    print(f"{'Parameter':<15} {'JAX':<20} {'Adjoint':<20} {'Rel. Diff':<15}")
    print("-"*60)
    
    for name in param_names:
        jax_val = np.mean(np.abs(jax_sens[name]))  # Average absolute sensitivity
        adj_val = abs(adjoint_sens[name])
        
        if adj_val > 1e-12:
            rel_diff = abs(jax_val - adj_val) / adj_val * 100
        else:
            rel_diff = 0.0
        
        print(f"{name:<15} {jax_val:<20.6e} {adj_val:<20.6e} {rel_diff:<14.2f}%")
    
    print("="*60)


def save_sensitivities(filename: str, sensitivities: dict, method: str = "JAX"):
    """Save sensitivity results to file"""
    import json
    
    # Convert arrays to lists for JSON
    data = {
        'method': method,
        'sensitivities': {}
    }
    
    for name, value in sensitivities.items():
        if isinstance(value, np.ndarray):
            data['sensitivities'][name] = {
                'mean': float(np.mean(value)),
                'max': float(np.max(np.abs(value))),
                'values': value.tolist()
            }
        else:
            data['sensitivities'][name] = float(value)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {method} sensitivities to {filename}")


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("Sensitivity Analysis Example")
    print("="*60)
    
    # Test JAX sensitivity
    print("\n1. Testing JAX Sensitivity...")
    
    jax_sens = JaxSensitivity()
    
    # Pressure heads to evaluate
    h_values = np.linspace(-5, 0, 50)
    
    # Parameters
    params = VanGenuchtenParams(
        alpha=2.0,
        n=1.5,
        theta_s=0.45,
        theta_r=0.067
    )
    
    # Compute sensitivities
    sens = jax_sens.compute_sensitivities(h_values, params)
    
    print(f"  Computed sensitivities at {len(h_values)} pressure head values")
    print(f"  ∂θ/∂α: mean = {np.mean(np.abs(sens['alpha'])):.4e}, max = {np.max(np.abs(sens['alpha'])):.4e}")
    print(f"  ∂θ/∂n: mean = {np.mean(np.abs(sens['n'])):.4e}, max = {np.max(np.abs(sens['n'])):.4e}")
    
    # Save results
    save_sensitivities('jax_sensitivities.json', sens, method="JAX")
    
    print("\n✓ JAX sensitivity computation successful!")
    print("\nNote: FireDrake-adjoint sensitivity requires running with your solver.")
    print("      See run_optimization.py for full example.")