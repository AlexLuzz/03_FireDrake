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
    
    def build_loss_functional(
        self,
        simulated_functions: List[Function],
        observation_functions: List[Function]
    ) -> AdjFloat:
        """
        Build differentiable loss functional using Firedrake functions
        
        CRITICAL: Loss must use Firedrake operations to be on the tape!
        
        Args:
            simulated_functions: List of Function objects from simulation
            observation_functions: List of Function objects with observations
        
        Returns:
            Differentiable loss value (AdjFloat)
        """
        loss = Constant(0.0)
        
        for sim_fn, obs_fn in zip(simulated_functions, observation_functions):
            # Compute squared difference using Firedrake operations
            diff = sim_fn - obs_fn
            loss += assemble(diff * diff * dx)
        
        return loss
    
    def setup_optimization(
        self,
        loss_functional: AdjFloat,
        parameter_controls: Dict[str, Control],
        initial_params: Dict[str, float]
    ):
        """
        Setup the reduced functional for optimization
        
        Args:
            loss_functional: Loss computed during annotated forward solve (AdjFloat)
            parameter_controls: Dict of {param_name: Control(constant)}
            initial_params: Dict of initial parameter values (for history tracking)
        """
        # Store control information
        self.control_names = list(parameter_controls.keys())
        self.controls = [parameter_controls[name] for name in self.control_names]
        
        # Create reduced functional
        # The loss_functional is already on the tape from the forward solve!
        self.reduced_functional = ReducedFunctional(loss_functional, self.controls)
        
        # Store initial state
        self.loss_history = [float(loss_functional)]
        self.param_history = [initial_params.copy()]
        
        print(f"Optimization setup complete:")
        print(f"  Initial loss: {float(loss_functional):.6e}")
        print(f"  Parameters: {', '.join(self.control_names)}")
        print(f"\nInitial parameter values:")
        for name, val in initial_params.items():
            print(f"  {name:20s} = {val:.6e}")
    
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
        
        # Get bounds in correct format for scipy
        min_vals, max_vals = self.bounds.get_bounds_lists()
        bounds_list = [min_vals, max_vals]
        
        if verbose:
            print(f"\nStarting adjoint-based optimization:")
            print(f"  Method: {method}")
            print(f"  Max iterations: {maxiter}")
            print(f"  Parameters: {len(self.controls)}")
            print(f"  Adjoint tape blocks: {len(get_working_tape().get_blocks())}")
            print()
        
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