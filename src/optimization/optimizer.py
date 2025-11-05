"""
Adjoint-Based Parameter Optimization for Firedrake
Uses pyadjoint for automatic differentiation through the PDE solve
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from firedrake import *
from firedrake.adjoint import *
from pyadjoint import Control, ReducedFunctional, minimize


def get_constant_value(obj) -> float:
    """
    Safely extract float value from a Constant or Control
    
    Args:
        obj: Can be a Constant, Control, or already a float
    
    Returns:
        Float value
    """
    if isinstance(obj, float) or isinstance(obj, int):
        return float(obj)
    
    # If it's a Control, get the Constant first
    if isinstance(obj, Control):
        obj = obj.data()
    
    # Now extract value from Constant
    try:
        # For Firedrake Constant
        return float(obj.values()[0])
    except (AttributeError, IndexError):
        try:
            # Alternative: direct float conversion
            return float(obj)
        except:
            raise TypeError(f"Cannot extract float from {type(obj)}")


@dataclass
class ObservationData:
    """Container for observation data"""
    times: np.ndarray              # Time points
    locations: List[Tuple]         # Coordinates where observations were made
    values: np.ndarray             # Observed values [n_times, n_locations]
    weights: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones_like(self.values)


@dataclass
class ParameterBounds:
    """Parameter bounds as dictionary {param_name: (min, max)}"""
    bounds: Dict[str, Tuple[float, float]]
    
    def get_bounds_lists(self) -> Tuple[List[float], List[float]]:
        """Return (min_values, max_values) lists in consistent order"""
        names = list(self.bounds.keys())
        mins = [self.bounds[name][0] for name in names]
        maxs = [self.bounds[name][1] for name in names]
        return mins, maxs


class AdjointOptimizer:
    """
    Parameter optimizer using Firedrake adjoint
    
    The key insight: Instead of running simulation many times, we:
    1. Run simulation ONCE with parameter Constants
    2. Build a loss functional that's recorded on the adjoint tape
    3. Use automatic differentiation to compute exact gradients
    4. Optimize using scipy with these gradients
    
    Usage:
        controls, constants = create_parameter_controls(initial_params)
        
        continue_annotation()
        simulated_data = your_simulation(constants)
        
        optimizer = AdjointOptimizer(observations, bounds)
        optimizer.setup_optimization(simulated_data, controls)
        best = optimizer.optimize()
    """
    
    def __init__(
        self,
        observations: ObservationData,
        bounds: ParameterBounds,
    ):
        self.observations = observations
        self.bounds = bounds
        self.reduced_functional = None
        self.controls = None
        self.control_names = None
        
        # Track optimization progress
        self.loss_history = []
        self.param_history = []
        self.iteration_count = 0
        
    def compute_loss_value(self, simulated: np.ndarray) -> float:
        """
        Compute weighted MSE loss (for evaluation, not optimization)
        Cuts off first 20% to avoid initial transients
        """
        cutoff = int(0.2 * len(simulated))
        residuals = (simulated[cutoff:] - self.observations.values[cutoff:]) * \
                    self.observations.weights[cutoff:]
        return float(np.sum(residuals ** 2) / np.sum(self.observations.weights[cutoff:]))
    
    def setup_optimization(
        self,
        probe_manager: 'ProbeManager',  # Pass the ProbeManager!
        parameter_controls: Dict[str, Control],
        initial_params: Dict[str, float]
    ):
        """
        Setup the reduced functional for optimization
        
        Args:
            probe_manager: ProbeManager with recorded Functions
            parameter_controls: Dict of {param_name: Control(constant)}
            initial_params: Dict of initial parameter values
        """
        # Compute loss from recorded Functions (stays on tape!)
        loss = self._compute_loss_from_functions(probe_manager)
        
        # Store control information
        self.control_names = list(parameter_controls.keys())
        self.controls = [parameter_controls[name] for name in self.control_names]
        
        # Create reduced functional
        self.reduced_functional = ReducedFunctional(loss, self.controls)
        
        # Store initial state
        self.loss_history = [float(loss)]
        self.param_history = [initial_params.copy()]
        
        print(f"Optimization setup complete:")
        print(f"  Initial loss: {float(loss):.6e}")
        print(f"  Parameters: {', '.join(self.control_names)}")
    
    def _compute_loss_from_functions(self, probe_manager):
        """
        Compute loss using assemble() to get proper OverloadedType
        """
        recorded_functions = probe_manager.get_recorded_functions()
        if not recorded_functions:
            raise RuntimeError("No Functions recorded in ProbeManager!")

        n_times = len(recorded_functions)
        cutoff = int(0.2 * n_times)
        
        P0 = probe_manager.P0DG
        probe_y = np.array([pos[1] for pos in probe_manager.probe_positions])

        # Accumulate loss as assembled scalars
        total_loss = 0.0  # Start with float
        n_obs = 0

        for i, (t, field_name, field_at_probes) in enumerate(recorded_functions):
            if i < cutoff:
                continue

            # field_at_probes is Function on VertexOnlyMesh
            # Create observation Function
            obs_func = Function(P0)
            obs_func.dat.data[:] = self.observations.values[i, :] - probe_y  # Store as pressure
            
            # Weight Function  
            weight_func = Function(P0)
            weight_func.dat.data[:] = self.observations.weights[i, :]
            
            # Compute weighted residual (UFL)
            # field_at_probes already has pressure, obs_func also has pressure
            diff = field_at_probes - obs_func
            weighted_diff = weight_func * diff
            
            # Assemble squared residual (this creates OverloadedType!)
            loss_contribution = assemble(weighted_diff * weighted_diff * dx)
            total_loss = total_loss + loss_contribution
            
            n_obs += len(probe_manager.probe_positions)

        if n_obs == 0:
            raise RuntimeError("No observations in loss computation")

        # Normalize (scale the final OverloadedType)
        total_loss = total_loss / float(n_obs)
        
        return total_loss
    
    def _callback(self, controls_values):
        """Callback to track optimization progress"""
        self.iteration_count += 1
        
        # Evaluate loss at current parameters
        current_loss = float(self.reduced_functional(controls_values))
        self.loss_history.append(current_loss)
        
        # Store parameters
        params = {name: float(val) 
                 for name, val in zip(self.control_names, controls_values)}
        self.param_history.append(params)
        
        print(f"Iteration {self.iteration_count}: Loss = {current_loss:.6e}")
    
    def optimize(
        self,
        method: str = 'L-BFGS-B',
        maxiter: int = 10,
        gtol: float = 1e-3,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run optimization using scipy's minimize
        
        HOW IT WORKS:
        1. scipy asks for loss at current parameters
        2. ReducedFunctional "replays" the forward solve with new params
        3. scipy asks for gradient
        4. ReducedFunctional runs adjoint solve (backward through tape)
        5. Returns exact gradient via chain rule
        6. scipy updates parameters and repeats
        
        Args:
            method: Optimization method ('L-BFGS-B', 'SLSQP', 'TNC')
            maxiter: Maximum iterations
            gtol: Gradient tolerance
            verbose: Print progress
         
        Returns:
            Dictionary of optimized parameters
        """
        if self.reduced_functional is None:
            raise RuntimeError("Must call setup_optimization() first!")
        
        # Get bounds as scipy-style list of (min, max) tuples per parameter
        min_vals, max_vals = self.bounds.get_bounds_lists()
        # Get bounds in the SAME ORDER as self.control_names
        min_vals = [self.bounds.bounds[name][0] for name in self.control_names]
        max_vals = [self.bounds.bounds[name][1] for name in self.control_names]
        bounds_list = [min_vals, max_vals]

        # DEBUG: Print what we're passing
        print("\n=== DEBUG BOUNDS ===")
        print(f"Number of controls: {len(self.controls)}")
        print(f"Control names: {self.control_names}")
        print(f"Min vals ({len(min_vals)}): {min_vals}")
        print(f"Max vals ({len(max_vals)}): {max_vals}")
        print(f"Bounds list: {bounds_list}")
        print("===================\n")
        
        # Reset iteration counter
        self.iteration_count = 0
        
        # Run optimization with callback
        optimal_values = minimize(
            self.reduced_functional,
            method=method,
            bounds=bounds_list,
            callback=self._callback if verbose else None,
            options={
                'maxiter': maxiter,
                'gtol': gtol,
                'disp': verbose
            }
        )
        
        # Convert to dictionary
        result = {name: float(val) for name, val in zip(self.control_names, optimal_values)}
        
        if verbose:
            print("\nOptimization complete!")
            print("Optimized parameters:")
            for name, value in result.items():
                print(f"  {name:15s} = {value:.6e}")
            print(f"\nFinal loss: {self.loss_history[-1]:.6e}")
            print(f"Improvement: {(1 - self.loss_history[-1]/self.loss_history[0])*100:.2f}%")
        
        return result


def create_parameter_controls(params: Dict[str, float]) -> Tuple[Dict[str, Control], Dict[str, Constant]]:
    """
    Helper function to create Constant parameters with Controls
    """
    controls = {}
    constants = {}
    
    for name, value in params.items():
        const = Constant(value)  # Remove name parameter!

        # DIAGNOSTIC: Check if Constant has adjoint support
        print(f"\nDEBUG {name}:")
        print(f"  Type: {type(const)}")
        print(f"  Has ufl_element: {hasattr(const, 'ufl_element')}")
        print(f"  Module: {type(const).__module__}")
        print(f"  Value: {const.values()}")

        constants[name] = const
        controls[name] = Control(const)
    
    return controls, constants


def create_tight_bounds(
    base_params: Dict[str, float],
    variation_pct: float = 20.0,
    absolute_mins: Optional[Dict[str, float]] = None
) -> ParameterBounds:
    """
    Create bounds around base parameter estimates
    
    Args:
        base_params: Your best parameter estimates
        variation_pct: Allowed variation percentage (default ±20%)
        absolute_mins: Optional absolute minimum values (e.g., {'n': 1.05})
    
    Returns:
        ParameterBounds object
    """
    bounds_dict = {}
    
    for name, base_val in base_params.items():
        variation = variation_pct / 100.0
        min_val = base_val * (1 - variation)
        max_val = base_val * (1 + variation)
        
        # Apply absolute constraints
        if absolute_mins and name in absolute_mins:
            min_val = max(absolute_mins[name], min_val)
        
        bounds_dict[name] = (min_val, max_val)
    
    return ParameterBounds(bounds_dict)