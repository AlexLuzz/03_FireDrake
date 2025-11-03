"""
Generic Parameter Optimization Framework
Optimizes parameters by fitting simulation outputs to observations
Uses PyTorch for optimization with finite differences or adjoint gradients
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Callable, Optional
from dataclasses import dataclass
from firedrake.adjoint import *
from pyadjoint import get_working_tape, Control, ReducedFunctional

@dataclass
class ObservationData:
    """Container for observation data"""
    times: np.ndarray              # Time points (seconds)
    locations: List[Tuple]         # Coordinates where observations were made
    values: np.ndarray             # Observed values [n_times, n_locations]
    weights: Optional[np.ndarray] = None  # Optional weights for each observation
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones_like(self.values)


@dataclass
class ParameterBounds:
    """Parameter bounds as dictionary {param_name: (min, max)}"""
    bounds: Dict[str, Tuple[float, float]]
    
    def get(self, name: str) -> Tuple[float, float]:
        return self.bounds[name]
    
    @property
    def names(self) -> List[str]:
        return list(self.bounds.keys())


class ParameterOptimizer:
    """
    Generic parameter optimizer using PyTorch
    
    Usage:
        optimizer = ParameterOptimizer(
            forward_model=your_simulation_function,
            observations=obs_data,
            bounds=param_bounds,
            initial_params={'param1': 1.0, 'param2': 2.0}
        )
        
        best_params = optimizer.optimize(n_iterations=50)
    """
    
    def __init__(
        self,
        forward_model: Callable,
        observations: ObservationData,
        bounds: ParameterBounds,
        initial_params: Dict[str, float]
    ):
        """
        Args:
            forward_model: Function that takes params dict and returns simulated values
                          Signature: forward_model(params) -> np.ndarray [n_times, n_locations]
            observations: ObservationData with times, locations, values
            bounds: ParameterBounds with min/max for each parameter
            initial_params: Starting parameter values
        """
        self.forward_model = forward_model
        self.observations = observations
        self.bounds = bounds
        self.param_names = bounds.names
        
        # Transform initial params to optimization space
        self.current_params = self._to_opt_space(initial_params)
        
        # History
        self.loss_history = []
        self.param_history = []
        
    def _to_opt_space(self, params: Dict[str, float]) -> torch.Tensor:
        """Transform physical parameters to unbounded optimization space using sigmoid inverse"""
        opt_params = []
        for name in self.param_names:
            val = params[name]
            min_val, max_val = self.bounds.get(name)
            
            # Normalize to [0,1] then inverse sigmoid
            normalized = (val - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 1e-6, 1 - 1e-6)
            opt_val = np.log(normalized / (1 - normalized))
            opt_params.append(opt_val)
            
        return torch.tensor(opt_params, dtype=torch.float64, requires_grad=True)
    
    def _to_physical_space(self, opt_params: torch.Tensor) -> Dict[str, float]:
        """Transform optimization parameters back to physical space"""
        params = {}
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.bounds.get(name)
            normalized = torch.sigmoid(opt_params[i])
            val = min_val + (max_val - min_val) * normalized
            params[name] = float(val)
        return params
    
    def compute_loss(self, simulated: np.ndarray) -> float:
        """Weighted mean squared error"""

        # Cut off the first 20% of the data (to avoid initial transient effects)
        cutoff = int(0.2 * len(simulated))
        residuals = (simulated[cutoff:] - self.observations.values[cutoff:]) * self.observations.weights[cutoff:]
        return np.sum(residuals ** 2) / np.sum(self.observations.weights[cutoff:])
    
    def compute_gradient_fd(self, params: Dict[str, float], epsilon: float = 1e-5) -> np.ndarray:
        """Compute gradient using finite differences"""
        baseline_sim = self.forward_model(params)
        baseline_loss = self.compute_loss(baseline_sim)
        
        gradients = []
        for name in self.param_names:
            perturbed = params.copy()
            perturbed[name] += epsilon
            
            perturbed_sim = self.forward_model(perturbed)
            perturbed_loss = self.compute_loss(perturbed_sim)
            
            grad = (perturbed_loss - baseline_loss) / epsilon
            gradients.append(grad)
        
        return np.array(gradients)
    
    def optimize(
        self,
        n_iterations: int = 50,
        learning_rate: float = 0.05,
        optimizer_type: str = 'adam',
        use_finite_diff: bool = True,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run optimization
        
        Args:
            n_iterations: Number of iterations
            learning_rate: Learning rate for optimizer
            optimizer_type: 'adam', 'lbfgs', or 'sgd'
            use_finite_diff: Use finite differences (True) or adjoint (False)
            verbose: Print progress
            
        Returns:
            Optimized parameters
        """
        # Setup PyTorch optimizer
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam([self.current_params], lr=learning_rate)
        elif optimizer_type == 'lbfgs':
            optimizer = torch.optim.LBFGS([self.current_params], lr=learning_rate, max_iter=20)
        else:
            optimizer = torch.optim.SGD([self.current_params], lr=learning_rate)
        
        if verbose:
            print(f"Starting optimization: {len(self.param_names)} parameters, {n_iterations} iterations")
            print(f"Method: {'Finite Differences' if use_finite_diff else 'Adjoint'}")
            print()
        
        for iteration in range(n_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Get physical params and run simulation
                params = self._to_physical_space(self.current_params)
                simulated = self.forward_model(params)
                loss = self.compute_loss(simulated)
                
                # Store history
                self.loss_history.append(loss)
                self.param_history.append(params.copy())
                
                if verbose and (iteration % 10 == 0 or iteration == 0):
                    print(f"Iter {iteration:3d}: Loss = {loss:.6e}")
                
                # Compute gradients
                if use_finite_diff:
                    grad_physical = self.compute_gradient_fd(params)
                    
                    # Chain rule: d_loss/d_opt = d_loss/d_phys * d_phys/d_opt
                    sigmoid_vals = torch.sigmoid(self.current_params)
                    transform_grad = sigmoid_vals * (1 - sigmoid_vals)
                    
                    scales = torch.tensor([self.bounds.get(name)[1] - self.bounds.get(name)[0] 
                                          for name in self.param_names])
                    
                    self.current_params.grad = torch.tensor(grad_physical) * transform_grad * scales
                else:
                    # For adjoint, user needs to implement in their forward_model
                    raise NotImplementedError("Adjoint requires pyadjoint integration")
                
                return torch.tensor(loss)
            
            optimizer.step(closure)
            
            # Check convergence
            if len(self.loss_history) > 1:
                rel_change = abs(self.loss_history[-1] - self.loss_history[-2]) / (self.loss_history[-2] + 1e-10)
                if bool(rel_change < 1e-2):  # Explicitly convert to Python bool
                    if verbose:
                        print(f"\nConverged after {iteration + 1} iterations (rel_change={rel_change:.6e})")
                    break
        
        # Return best parameters
        best_idx = np.argmin(self.loss_history)
        if verbose:
            print(f"\nOptimization complete. Best loss: {self.loss_history[best_idx]:.6e}")
        
        return self.param_history[best_idx]


class MultiStartOptimizer:
    """Run optimization from multiple starting points to avoid local minima"""
    
    def __init__(self, base_optimizer: ParameterOptimizer):
        self.base_optimizer = base_optimizer
    
    def optimize(
        self,
        n_starts: int = 5,
        n_iterations: int = 30,
        **kwargs
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Run multi-start optimization
        
        Returns:
            (best_params, all_results)
        """
        results = []
        
        print(f"Multi-start optimization: {n_starts} starting points\n")
        
        for i in range(n_starts):
            print(f"--- Start {i+1}/{n_starts} ---")
            
            # Random initial parameters within bounds
            initial = {}
            for name in self.base_optimizer.param_names:
                min_val, max_val = self.base_optimizer.bounds.get(name)
                initial[name] = np.random.uniform(min_val, max_val)
            
            # Reset optimizer with new initial guess
            self.base_optimizer.current_params = self.base_optimizer._to_opt_space(initial)
            self.base_optimizer.loss_history = []
            self.base_optimizer.param_history = []
            
            # Optimize
            best_params = self.base_optimizer.optimize(n_iterations=n_iterations, verbose=False, **kwargs)
            final_loss = self.base_optimizer.loss_history[-1]
            
            results.append({
                'initial': initial,
                'optimized': best_params,
                'loss': final_loss
            })
            
            print(f"Final loss: {final_loss:.6e}\n")
        
        # Find best
        best_result = min(results, key=lambda x: x['loss'])
        print(f"Best result: loss = {best_result['loss']:.6e}")
        
        return best_result['optimized'], results

def create_tight_bounds(
    base_params: Dict[str, float],
    variation_pct: float = 20.0,
    min_constraints: Optional[Dict[str, float]] = None
) -> Tuple[Dict[str, float], ParameterBounds]:
    """
    Create tight bounds around base parameter estimates
    
    Args:
        base_params: Your best parameter estimates
        variation_pct: Allowed variation percentage (e.g., 20 for ±20%)
        min_constraints: Optional minimum values (e.g., {'n': 1.05})
    
    Returns:
        (initial_params, bounds)
    """
    bounds_dict = {}
    initial = {}
    
    for name, base_val in base_params.items():
        variation = variation_pct / 100.0
        min_val = base_val * (1 - variation)
        max_val = base_val * (1 + variation)
        
        # Apply constraints if provided
        if min_constraints and name in min_constraints:
            min_val = max(min_constraints[name], min_val)
        
        bounds_dict[name] = (min_val, max_val)
        
        # Initial guess: slightly perturbed
        perturbation = np.random.uniform(-0.1, 0.1) * variation * base_val
        initial[name] = np.clip(base_val + perturbation, min_val, max_val)
    
    return initial, ParameterBounds(bounds_dict)