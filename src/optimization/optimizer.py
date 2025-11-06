"""
Adjoint-Based Parameter Optimization for Firedrake
Uses pyadjoint for automatic differentiation through the PDE solve
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from firedrake import Function, assemble, Constant, FunctionSpace, dx
# Use the Firedrake-adjoint wrappers for pyadjoint primitives to ensure
# correct Constant/Control handling in this environment.
from firedrake.adjoint import Control, ReducedFunctional, minimize


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
        """
        Compute loss using assemble() to get proper OverloadedType
        """
        recorded_functions = probe_manager.get_recorded_functions()
        if not recorded_functions:
            raise RuntimeError("No Functions recorded in ProbeManager!")

        n_times = len(recorded_functions)
        cutoff = int(0.2 * n_times)
        
        P0 = probe_manager.P0DG

        # Accumulate loss as assembled scalars
        total_loss = 0.0  # Start with float
        n_obs = 0

        for i, (t, field_name, field_at_probes) in enumerate(recorded_functions):
            if i < cutoff:
                continue

            # field_at_probes is Function on VertexOnlyMesh (already includes elevation!)
            # Create observation Function (observations are total head/elevation)
            obs_func = Function(P0)
            obs_func.dat.data[:] = self.observations.values[i, :]
            
            # Weight Function  
            weight_func = Function(P0)
            weight_func.dat.data[:] = self.observations.weights[i, :]
            
            # Compute weighted residual (UFL)
            # Both field_at_probes and obs_func are total heads (elevation)
            diff = field_at_probes - obs_func
            weighted_diff = weight_func * diff
            
            # Assemble squared residual (this creates OverloadedType!)
            total_loss += weighted_diff * weighted_diff * dx            
            n_obs += len(probe_manager.probe_positions)

        if n_obs == 0:
            raise RuntimeError("No observations in loss computation")

        # Assemble symbolic loss at the end
        loss = assemble(Constant(1.0 / float(n_obs)) * total_loss)
        return loss
    
    def _callback(self, controls_values):
        """Callback to track optimization progress"""
        self.iteration_count += 1
        
        # Convert numpy array to list
        if hasattr(controls_values, 'tolist'):
            values_list = controls_values.tolist()
        else:
            values_list = list(controls_values)
        
        # Update Function objects with new values and pass to ReducedFunctional
        # ReducedFunctional expects Function objects, not scalar values
        funcs_list = []
        for func, val in zip(self.control_functions, values_list):
            # Create a copy and assign the new scalar value
            func_copy = func.copy(deepcopy=True)
            func_copy.dat.data[:] = val
            funcs_list.append(func_copy)
        
        # Evaluate loss at current parameters
        current_loss = float(self.reduced_functional(funcs_list))
        self.loss_history.append(current_loss)
        
        # Store parameters
        params = {name: float(val) 
                 for name, val in zip(self.control_names, values_list)}
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
        
        # Get bounds in the SAME ORDER as self.control_names
        min_vals = [self.bounds.bounds[name][0] for name in self.control_names]
        max_vals = [self.bounds.bounds[name][1] for name in self.control_names]
        # Pyadjoint expects:
        #  - single parameter: [lower, upper]
        #  - multiple parameters: [[lower1, ...], [upper1, ...]]
        if len(self.controls) == 1:
            bounds_list = [min_vals[0], max_vals[0]]
        else:
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
        
        # Extract scalar values from Function objects (R-space Functions have single DOF)
        scalar_values = []
        for func in optimal_values:
            if hasattr(func, 'dat'):
                # It's a Function - extract the single scalar value
                scalar_values.append(float(func.dat.data[0]))
            else:
                # Already a scalar
                scalar_values.append(float(func))
        
        # Convert to dictionary
        result = {name: val for name, val in zip(self.control_names, scalar_values)}
        
        if verbose:
            print("\nOptimization complete!")
            print("Optimized parameters:")
            for name, value in result.items():
                print(f"  {name:15s} = {value:.6e}")
            print(f"\nFinal loss: {self.loss_history[-1]:.6e}")
            print(f"Improvement: {(1 - self.loss_history[-1]/self.loss_history[0])*100:.2f}%")
        
        return result


def create_parameter_controls(params: Dict[str, float], mesh) -> Tuple[Dict[str, Control], Dict[str, Function]]:
    """
    Helper function to create scalar Function parameters with Controls
    
    Uses "R" (Real) function space - a special 0D space with exactly 1 DOF per parameter.
    This gives us scalar values that work correctly with pyadjoint's serialization.
    
    Args:
        params: Dictionary of parameter names and initial values
        mesh: The mesh to create the R space on
        
    Returns:
        (controls_dict, functions_dict) where functions_dict contains scalar Functions
    """
    controls = {}
    functions = {}
    
    # Create a "Real" space - this is a 0-dimensional space with 1 DOF (scalar)
    R = FunctionSpace(mesh, "R", 0)
    
    for name, value in params.items():
        # Create a scalar Function on R space
        f_param = Function(R, name=name)
        f_param.assign(value)
        functions[name] = f_param
        controls[name] = Control(f_param)
    
    return controls, functions


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