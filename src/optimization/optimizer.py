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
        
        # Store parameter scales for normalization
        self.param_scales = self._compute_scales()
        
        # Transform to optimization space
        self.current_params = self._to_opt_space(initial_params)
        
        self.loss_history = []
        self.param_history = []
        self.gradient_history = []
    
    def _compute_scales(self) -> Dict[str, float]:
        """Compute characteristic scale for each parameter"""
        scales = {}
        for name in self.param_names:
            min_val, max_val = self.bounds.get(name)
            scales[name] = (max_val - min_val) / 2.0
        return scales
    
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
    
    def compute_gradient_fd(self, params: Dict[str, float], 
                           epsilon_relative: float = 1e-3) -> np.ndarray:
        """
        Improved finite differences with relative epsilon
        
        Args:
            epsilon_relative: Perturbation as fraction of parameter scale (e.g., 0.001 = 0.1%)
        """
        baseline_sim = self.forward_model(params)
        baseline_loss = self.compute_loss(baseline_sim)
        
        gradients = []
        for name in self.param_names:
            # Use RELATIVE epsilon scaled to parameter magnitude
            scale = self.param_scales[name]
            epsilon = epsilon_relative * scale
            
            perturbed = params.copy()
            perturbed[name] += epsilon
            
            # Ensure within bounds
            min_val, max_val = self.bounds.get(name)
            perturbed[name] = np.clip(perturbed[name], min_val, max_val)
            
            perturbed_sim = self.forward_model(perturbed)
            perturbed_loss = self.compute_loss(perturbed_sim)
            
            grad = (perturbed_loss - baseline_loss) / epsilon
            gradients.append(grad)
            
            print(f"  {name:15s}: ε={epsilon:.2e}, Δloss={perturbed_loss-baseline_loss:.2e}, grad={grad:.2e}")
        
        return np.array(gradients)
    
    def optimize(
        self,
        n_iterations: int = 50,
        learning_rate: float = 0.1,
        optimizer_type: str = 'adam',
        use_finite_diff: bool = True,
        epsilon_relative: float = 1e-3,  # NEW
        adaptive_lr: bool = True,  # NEW
        gradient_clip: float = 10.0,  # NEW: prevent explosion
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Enhanced optimization with better finite difference handling
        
        Args:
            epsilon_relative: FD perturbation as fraction of param range (0.001 = 0.1%)
            adaptive_lr: Reduce learning rate if loss increases
            gradient_clip: Clip gradient norm to prevent explosion
        """
        # Setup optimizer
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam([self.current_params], lr=learning_rate)
        elif optimizer_type == 'lbfgs':
            optimizer = torch.optim.LBFGS([self.current_params], lr=learning_rate, 
                                         max_iter=20, line_search_fn='strong_wolfe')
        else:
            optimizer = torch.optim.SGD([self.current_params], lr=learning_rate, 
                                       momentum=0.9)
        
        if verbose:
            print(f"Starting optimization: {len(self.param_names)} parameters, {n_iterations} iterations")
            print(f"Learning rate: {learning_rate}, Epsilon: {epsilon_relative}")
            print()
        
        current_lr = learning_rate
        
        for iteration in range(n_iterations):
            def closure():
                nonlocal current_lr
                
                optimizer.zero_grad()
                
                # Get physical params
                params = self._to_physical_space(self.current_params)
                
                # Run simulation
                simulated = self.forward_model(params)
                loss = self.compute_loss(simulated)
                
                # Store history
                self.loss_history.append(loss)
                self.param_history.append(params.copy())
                
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Iteration {iteration:3d}: Loss = {loss:.6e}")
                    print(f"Parameters:")
                    for name, val in params.items():
                        print(f"  {name:15s} = {val:.6e}")
                
                # Compute gradients (finite differences)
                if use_finite_diff:
                    print(f"\nComputing finite difference gradients (ε_rel={epsilon_relative}):")
                    grad_physical = self.compute_gradient_fd(params, epsilon_relative)
                    
                    # Store for diagnostics
                    self.gradient_history.append(grad_physical.copy())
                    
                    # Check if gradients are reasonable
                    grad_norm = np.linalg.norm(grad_physical)
                    print(f"\nGradient norm: {grad_norm:.2e}")
                    
                    if grad_norm < 1e-10:
                        print("⚠️  WARNING: Gradients very small! Try:")
                        print("   - Increase epsilon_relative")
                        print("   - Check if loss is changing")
                        print("   - Verify forward model is sensitive to parameters")
                    
                    # Clip gradients to prevent explosion
                    if grad_norm > gradient_clip:
                        grad_physical = grad_physical * (gradient_clip / grad_norm)
                        print(f"⚠️  Clipped gradients from {grad_norm:.2e} to {gradient_clip:.2e}")
                    
                    # Transform to optimization space with better scaling
                    sigmoid_vals = torch.sigmoid(self.current_params)
                    transform_grad = sigmoid_vals * (1 - sigmoid_vals)
                    
                    # Scale by parameter ranges
                    scales = torch.tensor([self.bounds.get(name)[1] - self.bounds.get(name)[0] 
                                          for name in self.param_names])
                    
                    # Apply chain rule
                    self.current_params.grad = torch.tensor(grad_physical) * transform_grad * scales
                    
                    print(f"Optimization space gradient norm: {self.current_params.grad.norm():.2e}")
                
                # Adaptive learning rate
                if adaptive_lr and len(self.loss_history) > 1:
                    if self.loss_history[-1] > self.loss_history[-2]:
                        current_lr *= 0.8
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        print(f"⚠️  Loss increased! Reducing learning rate to {current_lr:.2e}")
                    elif self.loss_history[-1] < self.loss_history[-2] * 0.99:
                        current_lr *= 1.05
                        current_lr = min(current_lr, learning_rate * 2)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        print(f"✓ Good progress! Increasing learning rate to {current_lr:.2e}")
                
                return torch.tensor(loss)
            
            optimizer.step(closure)
            
            # Convergence check
            if len(self.loss_history) > 2:
                recent_losses = self.loss_history[-3:]
                if max(recent_losses) - min(recent_losses) < 1e-6 * min(recent_losses):
                    if verbose:
                        print(f"\n✓ Converged after {iteration + 1} iterations")
                    break
        
        # Return best parameters
        best_idx = np.argmin(self.loss_history)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimization complete!")
            print(f"Best loss: {self.loss_history[best_idx]:.6e} at iteration {best_idx}")
            print(f"Final loss: {self.loss_history[-1]:.6e}")
        
        return self.param_history[best_idx]
    
    def diagnose_gradients(self, params: Dict[str, float], epsilon_range: List[float] = None):
        """
        Diagnostic tool: test different epsilon values
        
        Usage:
            optimizer.diagnose_gradients(initial_params)
        """
        if epsilon_range is None:
            epsilon_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        
        print("\n" + "="*70)
        print("GRADIENT DIAGNOSTICS")
        print("="*70)
        
        baseline_sim = self.forward_model(params)
        baseline_loss = self.compute_loss(baseline_sim)
        print(f"Baseline loss: {baseline_loss:.6e}\n")
        
        for eps in epsilon_range:
            print(f"\nTesting ε_relative = {eps:.1e}")
            print("-" * 70)
            
            gradients = []
            for name in self.param_names:
                scale = self.param_scales[name]
                epsilon = eps * scale
                
                perturbed = params.copy()
                perturbed[name] += epsilon
                
                perturbed_sim = self.forward_model(perturbed)
                perturbed_loss = self.compute_loss(perturbed_sim)
                
                grad = (perturbed_loss - baseline_loss) / epsilon
                gradients.append(grad)
                
                print(f"  {name:12s}: Δloss={perturbed_loss-baseline_loss:+.3e}, grad={grad:+.3e}")
            
            grad_norm = np.linalg.norm(gradients)
            print(f"  Gradient norm: {grad_norm:.3e}")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS:")
        print("  - Choose ε where gradients are stable and reasonable magnitude")
        print("  - Too small ε → noisy gradients (numerical errors)")
        print("  - Too large ε → biased gradients (nonlinearity)")
        print("="*70 + "\n")


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