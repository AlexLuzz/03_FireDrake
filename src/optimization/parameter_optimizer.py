"""
Parameter Optimization Framework for Richards Equation
Uses PyTorch optimizer with Firedrake/pyadjoint for gradient computation
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Callable
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
from firedrake import *
from firedrake_adjoint import *
import matplotlib.pyplot as plt


@dataclass
class ParameterBounds:
    """Bounds for van Genuchten parameters"""
    theta_r: Tuple[float, float] = (0.01, 0.15)
    theta_s: Tuple[float, float] = (0.30, 0.55)
    alpha: Tuple[float, float] = (0.001, 0.1)    # 1/cm
    n: Tuple[float, float] = (1.1, 3.0)
    Ks: Tuple[float, float] = (1.0, 500.0)        # cm/day


@dataclass
class ObservationData:
    """Container for observed water level data"""
    times: np.ndarray          # Time points (seconds from start)
    locations: List[Tuple[float, float]]  # (x, y) coordinates
    water_levels: np.ndarray   # Shape: (n_times, n_locations)
    weights: np.ndarray = None  # Optional weights for each observation
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones_like(self.water_levels)


class ParameterVector:
    """
    Handles parameter transformation between physical and optimization space
    Uses log-transform for positive parameters to improve optimization
    """
    def __init__(self, bounds: ParameterBounds):
        self.bounds = bounds
        self.param_names = ['theta_r', 'theta_s', 'alpha', 'n', 'Ks']
        
    def to_optimization_space(self, params: Dict[str, float]) -> torch.Tensor:
        """Transform physical parameters to optimization space (log-scale for some)"""
        opt_params = []
        for name in self.param_names:
            val = params[name]
            min_val, max_val = getattr(self.bounds, name)
            
            # Normalize to [0, 1] then apply inverse sigmoid
            normalized = (val - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 1e-6, 1 - 1e-6)
            opt_val = np.log(normalized / (1 - normalized))
            opt_params.append(opt_val)
            
        return torch.tensor(opt_params, dtype=torch.float64, requires_grad=True)
    
    def to_physical_space(self, opt_params: torch.Tensor) -> Dict[str, float]:
        """Transform optimization parameters back to physical space"""
        params = {}
        for i, name in enumerate(self.param_names):
            min_val, max_val = getattr(self.bounds, name)
            # Apply sigmoid to get [0, 1] then scale to bounds
            normalized = torch.sigmoid(opt_params[i])
            val = min_val + (max_val - min_val) * normalized
            params[name] = float(val)
        return params


class RichardsParameterOptimizer:
    """
    Main optimization class that wraps Firedrake simulation in PyTorch
    """
    def __init__(
        self,
        simulation_builder: Callable,  # Function that builds and runs simulation
        observations: ObservationData,
        bounds: ParameterBounds,
        initial_params: Dict[str, float],
        material_zone: str = "base"  # Which material zone to optimize
    ):
        self.simulation_builder = simulation_builder
        self.observations = observations
        self.bounds = bounds
        self.material_zone = material_zone
        self.param_vector = ParameterVector(bounds)
        
        # Convert initial parameters to optimization space
        self.current_params = self.param_vector.to_optimization_space(initial_params)
        
        # History tracking
        self.loss_history = []
        self.param_history = []
        
    def run_forward_simulation(self, params: Dict[str, float]) -> np.ndarray:
        """
        Run Firedrake simulation with given parameters
        Returns: simulated water levels at observation points and times
        """
        # Build and run simulation with current parameters
        probe_manager = self.simulation_builder(params, self.material_zone)
        
        # Extract water levels at observation times and locations
        simulated = np.zeros((len(self.observations.times), len(self.observations.locations)))
        
        for i, (t, loc) in enumerate(zip(self.observations.times, self.observations.locations)):
            for j, (x, y) in enumerate(self.observations.locations):
                # Interpolate from probe data
                simulated[i, j] = probe_manager.get_value_at(t, x, y)
        
        return simulated
    
    def compute_loss(self, simulated: np.ndarray) -> float:
        """
        Compute weighted least squares loss
        """
        residuals = (simulated - self.observations.water_levels) * self.observations.weights
        loss = np.sum(residuals ** 2) / np.sum(self.observations.weights)
        return loss
    
    def compute_gradient_fd(self, params_dict: Dict[str, float], epsilon: float = 1e-5) -> torch.Tensor:
        """
        Compute gradient using finite differences
        Useful when adjoint is not available or for verification
        """
        baseline_sim = self.run_forward_simulation(params_dict)
        baseline_loss = self.compute_loss(baseline_sim)
        
        gradients = []
        for i, name in enumerate(self.param_vector.param_names):
            # Perturb parameter
            perturbed_params = params_dict.copy()
            perturbed_params[name] += epsilon
            
            # Run simulation
            perturbed_sim = self.run_forward_simulation(perturbed_params)
            perturbed_loss = self.compute_loss(perturbed_sim)
            
            # Finite difference gradient
            grad = (perturbed_loss - baseline_loss) / epsilon
            gradients.append(grad)
        
        return torch.tensor(gradients, dtype=torch.float64)
    
    def objective_function(self, opt_params: torch.Tensor) -> torch.Tensor:
        """
        Objective function for PyTorch optimizer
        """
        # Convert to physical parameters
        params_dict = self.param_vector.to_physical_space(opt_params)
        
        # Run forward simulation
        simulated = self.run_forward_simulation(params_dict)
        
        # Compute loss
        loss = self.compute_loss(simulated)
        
        # Store history
        self.loss_history.append(loss)
        self.param_history.append(params_dict.copy())
        
        print(f"Iteration {len(self.loss_history)}: Loss = {loss:.6f}")
        print(f"  Parameters: {params_dict}")
        
        return torch.tensor(loss, dtype=torch.float64)
    
    def optimize(
        self,
        max_iterations: int = 50,
        learning_rate: float = 0.01,
        optimizer_type: str = 'adam',
        use_finite_diff: bool = True
    ):
        """
        Run optimization using PyTorch optimizer
        
        Args:
            max_iterations: Maximum number of optimization iterations
            learning_rate: Learning rate for optimizer
            optimizer_type: 'adam', 'lbfgs', or 'sgd'
            use_finite_diff: Use finite differences for gradients (vs adjoint)
        """
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam([self.current_params], lr=learning_rate)
        elif optimizer_type.lower() == 'lbfgs':
            optimizer = torch.optim.LBFGS([self.current_params], lr=learning_rate, max_iter=20)
        else:
            optimizer = torch.optim.SGD([self.current_params], lr=learning_rate)
        
        print("=" * 60)
        print("Starting Parameter Optimization")
        print("=" * 60)
        
        for iteration in range(max_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Compute objective
                loss = self.objective_function(self.current_params)
                
                # Compute gradients
                if use_finite_diff:
                    params_dict = self.param_vector.to_physical_space(self.current_params)
                    grad = self.compute_gradient_fd(params_dict)
                    
                    # Account for parameter transformation chain rule
                    # d_loss/d_opt = d_loss/d_phys * d_phys/d_opt
                    sigmoid_vals = torch.sigmoid(self.current_params)
                    transform_grad = sigmoid_vals * (1 - sigmoid_vals)
                    
                    min_vals = torch.tensor([getattr(self.bounds, name)[0] for name in self.param_vector.param_names])
                    max_vals = torch.tensor([getattr(self.bounds, name)[1] for name in self.param_vector.param_names])
                    scale = max_vals - min_vals
                    
                    self.current_params.grad = grad * transform_grad * scale
                else:
                    loss.backward()
                
                return loss
            
            optimizer.step(closure)
            
            # Check convergence
            if len(self.loss_history) > 2:
                rel_change = abs(self.loss_history[-1] - self.loss_history[-2]) / (self.loss_history[-2] + 1e-10)
                if rel_change < 1e-6:
                    print(f"\nConverged after {iteration + 1} iterations")
                    break
        
        print("=" * 60)
        print("Optimization Complete")
        print("=" * 60)
        
        # Return best parameters
        best_idx = np.argmin(self.loss_history)
        return self.param_history[best_idx]
    
    def plot_results(self, save_path: str = None):
        """Plot optimization history and fit quality"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss history
        axes[0, 0].semilogy(self.loss_history, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss (log scale)')
        axes[0, 0].set_title('Optimization Convergence')
        axes[0, 0].grid(True)
        
        # Parameter evolution
        for param_name in self.param_vector.param_names:
            values = [p[param_name] for p in self.param_history]
            axes[0, 1].plot(values, label=param_name, linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Parameter Value')
        axes[0, 1].set_title('Parameter Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Best fit comparison
        best_params = self.param_history[np.argmin(self.loss_history)]
        best_sim = self.run_forward_simulation(best_params)
        
        # Plot for first location
        axes[1, 0].plot(self.observations.times / 3600, self.observations.water_levels[:, 0], 
                       'ko', label='Observed', markersize=6)
        axes[1, 0].plot(self.observations.times / 3600, best_sim[:, 0], 
                       'r-', label='Simulated', linewidth=2)
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Water Level (m)')
        axes[1, 0].set_title(f'Fit at Location {self.observations.locations[0]}')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Residuals
        residuals = best_sim[:, 0] - self.observations.water_levels[:, 0]
        axes[1, 1].plot(self.observations.times / 3600, residuals, 'bo-', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--')
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Residual (m)')
        axes[1, 1].set_title('Residuals')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_observations_from_csv(csv_path: str, location: Tuple[float, float]) -> ObservationData:
    """
    Load observed water level data from CSV
    Expected format: datetime, water_level
    """
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    
    # Convert to seconds from first observation
    start_time = df['datetime'].iloc[0]
    times = (df['datetime'] - start_time).dt.total_seconds().values
    
    # Assuming single location for now
    water_levels = df['water_level'].values.reshape(-1, 1)
    
    return ObservationData(
        times=times,
        locations=[location],
        water_levels=water_levels
    )