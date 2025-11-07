"""
Adjoint-Based Parameter Optimization for Firedrake
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from firedrake import Function, assemble, Constant, FunctionSpace, dx
from firedrake.adjoint import Control, ReducedFunctional, minimize

@dataclass
class ObservationData:
    """Container for observation data"""
    times: np.ndarray
    locations: List[Tuple]
    values: np.ndarray
    weights: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones_like(self.values)

@dataclass
class ParameterBounds:
    """Parameter bounds as dictionary {param_name: (min, max)}"""
    bounds: Dict[str, Tuple[float, float]]

class AdjointOptimizer:
    """
    Parameter optimizer using Firedrake adjoint
    """
    
    def __init__(
        self,
        observations: ObservationData,
        bounds: ParameterBounds
    ):
        self.observations = observations
        self.bounds = bounds
        self.reduced_functional = None
        self.controls = None
        self.control_names = None
        self.control_functions = None
        
        # Track optimization progress
        self.loss_history = []
        self.param_history = []
        self.iteration_count = 0
    
    def setup_optimization(
        self,
        probe_manager: 'ProbeManager',  # Pass the ProbeManager!
        parameter_controls: Dict[str, Control],
        param_functions: Dict[str, 'Function'],
        initial_params: Dict[str, float]
    ):
        """
        Setup the reduced functional for optimization
        
        Args:
            probe_manager: ProbeManager with recorded Functions
            parameter_controls: Dict of {param_name: Control(Function)}
            param_functions: Dict of {param_name: Function} - the underlying Function objects
            initial_params: Dict of initial parameter values
        """
        # Compute loss from recorded Functions (stays on tape!)
        loss = self._compute_loss_from_functions(probe_manager)
        
        # Store control information
        self.control_names = list(parameter_controls.keys())
        self.controls = [parameter_controls[name] for name in self.control_names]
        self.control_functions = [param_functions[name] for name in self.control_names]
        
        # Create reduced functional
        self.reduced_functional = ReducedFunctional(loss, self.controls)
        
        # Store initial state
        self.loss_history = [float(loss)]
        self.param_history = [initial_params.copy()]
        
        print(f"Optimization setup complete:")
        print(f"  Initial loss: {float(loss):.6e}")
        print(f"  Parameters: {', '.join(self.control_names)}")
    
    def _compute_loss_from_functions(self, probe_manager):
        """Compute loss"""
        recorded_functions = probe_manager.get_recorded_functions()
        if not recorded_functions:
            raise RuntimeError("No Functions recorded!")

        n_times = len(recorded_functions)
        cutoff = int(0.2 * n_times)
        
        P0 = probe_manager.P0DG
        total_loss = 0.0
        n_obs = 0
        
        print(f"  Using timesteps {cutoff} to {n_times-1}")

        for i, (t, field_name, field_at_probes) in enumerate(recorded_functions):
            if i < cutoff:
                continue

            obs_func = Function(P0)
            obs_func.dat.data[:] = self.observations.values[i, :]
            
            weight_func = Function(P0)
            weight_func.dat.data[:] = self.observations.weights[i, :]
            
            diff = field_at_probes - obs_func
            weighted_diff = weight_func * diff
            
            total_loss += weighted_diff * weighted_diff * dx            
            n_obs += len(probe_manager.probe_positions)

        loss = assemble(Constant(1.0 / float(n_obs)) * total_loss)
        print(f"  Total observations: {n_obs}")
        return loss
    
    def optimize(
        self,
        method: str = 'L-BFGS-B',
        maxiter: int = 10,
        gtol: float = 1e-3,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run optimization with proper error handling
        """
        if self.reduced_functional is None:
            raise RuntimeError("Must call setup_optimization() first!")
        
        # Get bounds
        min_vals = [self.bounds.bounds[name][0] for name in self.control_names]
        max_vals = [self.bounds.bounds[name][1] for name in self.control_names]
        if len(self.controls) == 1:
            bounds_list = [min_vals[0], max_vals[0]]
        else:
            bounds_list = [min_vals, max_vals]
        
        print(f"=== DEBUG BOUNDS ===")
        print(f"Number of controls: {len(self.controls)}")
        print(f"Control names: {self.control_names}")
        print(f"Min vals: {min_vals}")
        print(f"Max vals: {max_vals}")
        print(f"Bounds list: {bounds_list}")
        print(f"===================\n")

        print(f"\n=== OPTIMIZATION ({method}) ===")
        print(f"Max iterations: {maxiter}, gtol: {gtol}")
        
        # Callback for monitoring
        def callback(controls_values):
            self.iteration_count += 1

            # Evaluate loss with current control values
            #current_loss = float(self.reduced_functional(controls_values))
            current_loss = float(self.loss_history[-1]) if self.loss_history else np.nan
            self.loss_history.append(current_loss)
            
            # Get gradients and store params
            grads = self.reduced_functional.derivative()
            grad_info = []
            params = {}
            
            for i, name in enumerate(self.control_names):
                grad_val = float(grads[i].dat.data[0]) if hasattr(grads[i], 'dat') else float(grads[i])
                param_val = float(controls_values[i])  # controls_values should already be numpy array
                
                params[name] = param_val
                grad_info.append((name, abs(grad_val)))
            
            self.param_history.append(params)
            grad_info.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nIteration {self.iteration_count}: Loss = {current_loss:.6e}")
            print(f"  Top gradients: {grad_info[0][0]}={grad_info[0][1]:.2e}, "
                f"{grad_info[1][0]}={grad_info[1][1]:.2e}")
            print(f"  Loss history: {self.loss_history}")
            if len(self.loss_history) > 1:
                rel_change = abs(self.loss_history[-1] - self.loss_history[-2]) / abs(self.loss_history[-2])
                print(f"  Relative change: {rel_change:.2e}")
        
        # Initialize optimal_values to None
        optimal_values = None
        
        # Try optimization with error handling
        try:
            optimal_values = minimize(
                self.reduced_functional,
                method=method,
                bounds=bounds_list,
                callback=callback if verbose else None,
                options={
                    'maxiter': maxiter,
                    'gtol': gtol,
                    'ftol': 1e-5,
                    'maxls': 100,  # More line search steps
                    'maxcor': 20,
                    'disp': verbose
                }
            )
            
            # Extract results successfully
            result = {}
            for i, name in enumerate(self.control_names):
                val = float(optimal_values[i].dat.data[0]) if hasattr(optimal_values[i], 'dat') else float(optimal_values[i])
                result[name] = val
            
        except Exception as e:
            print(f"⚠️ Optimization didn't converge: {e}")
            
            # Check if we got partial results
            if optimal_values is not None:
                # We have some results, try to extract them
                if not isinstance(optimal_values, list):
                    print("  Got single value, expected list — wrapping it for safety.")
                    optimal_values = [optimal_values]
                
                result = {}
                for i, name in enumerate(self.control_names):
                    try:
                        val = float(optimal_values[i].dat.data[0]) if hasattr(optimal_values[i], 'dat') else float(optimal_values[i])
                        result[name] = val
                    except:
                        # Fallback to current control value
                        result[name] = float(self.control_functions[i].dat.data[0])
            else:
                # No optimal_values at all, use last parameters from history or current values
                print("  Returning best parameters found so far...")
                if self.param_history:
                    result = self.param_history[-1]
                else:
                    result = {name: float(func.dat.data[0]) for name, func in 
                            zip(self.control_names, self.control_functions)}
        
        print(f"\n{'='*40}")
        print(f"Final loss: {self.loss_history[-1]:.6e}")
        print(f"Improvement: {(1 - self.loss_history[-1]/self.loss_history[0])*100:.2f}%")
        
        return result

def create_parameter_controls(params: Dict[str, float], mesh) -> Tuple[Dict[str, Control], Dict[str, Function]]:
    """
    Helper function to create scalar Function parameters with Controls
    
    Uses "R" (Real) function space - a special 0D space with exactly 1 DOF per parameter.
    This gives us scalar values that work correctly with pyadjoint's serialization.
    
    For log-scale parameters: The Function stores log(value), and you must use exp()
    at the point of use in the physics to convert to physical space.
    Example: If 'Ks' is in log_scale_params, use:  Ks_physical = exp(params['Ks'])
    
    Args:
        params: Dictionary of parameter names and initial values (always in physical space)
        mesh: The mesh to create the R space on
        log_scale_params: List of parameter names to store in log-space (e.g., ['Ks_till'])
        
    Returns:
        (controls_dict, functions_dict) where functions_dict contains scalar Functions
        For log-scale params, the Functions contain log(value) - use exp() when using them!
    """
    
    controls = {}
    functions = {}
    
    # Create a "Real" space - this is a 0-dimensional space with 1 DOF (scalar)
    R = FunctionSpace(mesh, "R", 0)
    
    for name, value in params.items():
        f_param = Function(R, name=name)
        
        f_param.assign(value)
            
        functions[name] = f_param
        controls[name] = Control(f_param)
    
    return controls, functions

def create_tight_bounds(
    base_params: Dict[str, float],
    variation_pct: float = 20.0,
    custom_bounds: Optional[Dict[str, tuple]] = None,
    absolute_mins: Optional[Dict[str, float]] = None
) -> ParameterBounds:
    """Create bounds"""
    bounds_dict = {}
    
    for name, base_val in base_params.items():
        if custom_bounds and name in custom_bounds:
            min_val, max_val = custom_bounds[name]
            print(f"   Custom bounds for {name}: [{min_val:.2e}, {max_val:.2e}]")
        else:
            variation = variation_pct / 100.0
            min_val = base_val * (1 - variation)
            max_val = base_val * (1 + variation)
        
        if absolute_mins and name in absolute_mins:
            min_val = max(absolute_mins[name], min_val)
        
        bounds_dict[name] = (min_val, max_val)
    
    return ParameterBounds(bounds_dict)