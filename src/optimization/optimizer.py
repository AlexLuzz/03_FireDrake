"""
optimizer.py

Simple parameter optimization framework:
- Cost functions (L1, L2)
- Parameter bounds
- Scipy optimization
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import Callable, Dict, List
import json
from pathlib import Path


# ============================================================
# PARAMETER DEFINITION
# ============================================================

@dataclass
class Parameter:
    """Single parameter to optimize"""
    name: str
    value: float
    lower: float
    upper: float
    log_scale: bool = False  # Optimize in log10 space
    
    def to_normalized(self) -> float:
        """Convert current value to [0, 1]"""
        if self.log_scale:
            log_val = np.log10(self.value)
            log_min = np.log10(self.lower)
            log_max = np.log10(self.upper)
            return (log_val - log_min) / (log_max - log_min)
        else:
            return (self.value - self.lower) / (self.upper - self.lower)
    
    def from_normalized(self, x_norm: float) -> float:
        """Convert from [0, 1] to physical value"""
        if self.log_scale:
            log_min = np.log10(self.lower)
            log_max = np.log10(self.upper)
            log_val = log_min + x_norm * (log_max - log_min)
            return 10**log_val
        else:
            return self.lower + x_norm * (self.upper - self.lower)


# ============================================================
# COST FUNCTIONS
# ============================================================

def L1_cost(simulated: np.ndarray, observed: np.ndarray, 
            weights: np.ndarray = None) -> float:
    """
    L1 norm (sum of absolute differences)
    
    Cost = Σ w_i |sim_i - obs_i|
    """
    if weights is None:
        weights = np.ones_like(simulated)
    
    diff = np.abs(simulated - observed)
    return np.sum(weights * diff)


def L2_cost(simulated: np.ndarray, observed: np.ndarray,
            weights: np.ndarray = None) -> float:
    """
    L2 norm (sum of squared differences)
    
    Cost = Σ w_i (sim_i - obs_i)²
    """
    if weights is None:
        weights = np.ones_like(simulated)
    
    diff = simulated - observed
    return np.sum(weights * diff**2)


def RMSE_cost(simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    Root Mean Square Error
    
    RMSE = sqrt(Σ (sim_i - obs_i)² / N)
    """
    diff = simulated - observed
    return np.sqrt(np.mean(diff**2))


def NSE_cost(simulated: np.ndarray, observed: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency (converted to minimization)
    
    Returns (1 - NSE) so that minimization gives NSE → 1
    NSE = 1 is perfect fit
    """
    numerator = np.sum((observed - simulated)**2)
    denominator = np.sum((observed - np.mean(observed))**2)
    
    nse = 1.0 - numerator / denominator
    return 1.0 - nse  # Convert to minimization problem


# ============================================================
# OPTIMIZER
# ============================================================

class SimpleOptimizer:
    """
    Simple optimization wrapper
    """
    
    def __init__(self, 
                 parameters: List[Parameter],
                 forward_model: Callable,
                 cost_function: Callable = L2_cost):
        """
        Parameters:
        -----------
        parameters : list of Parameter
            Parameters to optimize
        forward_model : callable
            Function: params_dict -> simulated_values
        cost_function : callable
            Function: (simulated, observed, weights) -> cost
        """
        self.parameters = parameters
        self.forward_model = forward_model
        self.cost_function = cost_function
        
        # For storing history
        self.history = {
            'iteration': [],
            'cost': [],
            'parameters': []
        }
        self.iteration = 0
    
    def _params_to_dict(self, x_normalized: np.ndarray) -> Dict[str, float]:
        """Convert normalized parameter array to dictionary"""
        params_dict = {}
        for i, param in enumerate(self.parameters):
            params_dict[param.name] = param.from_normalized(x_normalized[i])
        return params_dict
    
    def objective(self, x_normalized: np.ndarray, 
                  observed: np.ndarray,
                  weights: np.ndarray = None,
                  verbose: bool = False) -> float:
        """
        Objective function to minimize
        
        Parameters:
        -----------
        x_normalized : array
            Normalized parameter values [0, 1]
        observed : array
            Observed data
        weights : array
            Observation weights (optional)
        
        Returns:
        --------
        cost : float
        """
        self.iteration += 1
        
        # Convert to physical parameters
        params_dict = self._params_to_dict(x_normalized)
        
        # Run forward model
        try:
            simulated = self.forward_model(params_dict)
        except Exception as e:
            print(f"Forward model failed: {e}")
            return 1e10  # Large penalty
        
        # Compute cost
        cost = self.cost_function(simulated, observed, weights)
        
        # Store history
        self.history['iteration'].append(self.iteration)
        self.history['cost'].append(cost)
        self.history['parameters'].append(params_dict.copy())
        
        if verbose or self.iteration % 10 == 0:
            print(f"Iteration {self.iteration}: Cost = {cost:.6e}")
            for name, value in params_dict.items():
                print(f"  {name} = {value:.6e}")
        
        return cost
    
    def optimize(self, 
                 observed: np.ndarray,
                 weights: np.ndarray = None,
                 method: str = 'L-BFGS-B',
                 maxiter: int = 100) -> Dict:
        """
        Run optimization
        
        Parameters:
        -----------
        observed : array
            Observed data to match
        weights : array
            Observation weights (optional)
        method : str
            'L-BFGS-B' (local, fast) or 'differential_evolution' (global)
        maxiter : int
            Maximum iterations
        
        Returns:
        --------
        results : dict
            Optimization results
        """
        # Initial guess (current parameter values)
        x0 = np.array([p.to_normalized() for p in self.parameters])
        
        print(f"\n{'='*60}")
        print(f"Starting Optimization: {method}")
        print(f"{'='*60}")
        print(f"Parameters: {[p.name for p in self.parameters]}")
        print(f"Initial values: {self._params_to_dict(x0)}")
        
        if method.upper() == 'DIFFERENTIAL_EVOLUTION':
            # Global optimization
            bounds = [(0, 1) for _ in self.parameters]
            
            result = differential_evolution(
                func=lambda x: self.objective(x, observed, weights),
                bounds=bounds,
                maxiter=maxiter,
                popsize=10,
                disp=True
            )
        else:
            # Local optimization (L-BFGS-B, SLSQP, TNC)
            bounds = [(0, 1) for _ in self.parameters]
            
            result = minimize(
                fun=lambda x: self.objective(x, observed, weights),
                x0=x0,
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter, 'disp': True}
            )
        
        # Extract results
        optimal_params = self._params_to_dict(result.x)
        
        results = {
            'optimal_parameters': optimal_params,
            'optimal_cost': result.fun,
            'success': result.success,
            'message': result.message if hasattr(result, 'message') else str(result),
            'n_iterations': self.iteration,
            'history': self.history
        }
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Success: {results['success']}")
        print(f"Final cost: {results['optimal_cost']:.6e}")
        print(f"Iterations: {results['n_iterations']}")
        print("\nOptimal parameters:")
        for name, value in optimal_params.items():
            print(f"  {name:15s} = {value:.6e}")
        
        return results
    
    def save_results(self, results: Dict, filename: str):
        """Save optimization results to JSON"""
        # Convert to JSON-serializable format
        output = {
            'optimal_parameters': results['optimal_parameters'],
            'optimal_cost': float(results['optimal_cost']),
            'success': bool(results['success']),
            'n_iterations': results['n_iterations'],
            'parameter_bounds': {
                p.name: {'lower': p.lower, 'upper': p.upper}
                for p in self.parameters
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {filename}")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_optimization_history(history: Dict, filename: str = None):
    """Plot cost and parameter evolution"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Cost evolution
    axes[0].semilogy(history['iteration'], history['cost'], 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Cost (log scale)')
    axes[0].set_title('Convergence')
    axes[0].grid(True, alpha=0.3)
    
    # Parameter evolution
    param_names = list(history['parameters'][0].keys())
    for name in param_names:
        values = [p[name] for p in history['parameters']]
        axes[1].plot(history['iteration'], values, label=name, linewidth=2)
    
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Parameter Value')
    axes[1].set_title('Parameter Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    else:
        plt.show()


def plot_fit_quality(simulated: np.ndarray, 
                    observed: np.ndarray,
                    times: np.ndarray = None,
                    filename: str = None):
    """Plot simulated vs observed data"""
    import matplotlib.pyplot as plt
    
    if times is None:
        times = np.arange(len(simulated))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time series
    ax1.plot(times, observed, 'o', label='Observed', markersize=6, alpha=0.7)
    ax1.plot(times, simulated, '-', label='Simulated', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot (1:1 line)
    ax2.scatter(observed, simulated, alpha=0.6, s=50)
    
    # 1:1 line
    min_val = min(observed.min(), simulated.min())
    max_val = max(observed.max(), simulated.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
    
    ax2.set_xlabel('Observed')
    ax2.set_ylabel('Simulated')
    ax2.set_title('Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    else:
        plt.show()


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":
    print("Simple Optimizer Test")
    print("="*60)
    
    # Define parameters
    parameters = [
        Parameter(name='alpha', value=2.0, lower=0.5, upper=10.0, log_scale=True),
        Parameter(name='n', value=1.5, lower=1.1, upper=3.0, log_scale=False),
    ]
    
    # Create synthetic observed data
    observed = np.array([1.2, 1.5, 1.8, 2.0, 2.1])
    
    # Simple forward model (dummy for testing)
    def forward_model(params):
        # Just return something that depends on parameters
        alpha = params['alpha']
        n = params['n']
        return observed + 0.1 * (alpha - 2.0) + 0.05 * (n - 1.5)
    
    # Create optimizer
    optimizer = SimpleOptimizer(
        parameters=parameters,
        forward_model=forward_model,
        cost_function=L2_cost
    )
    
    # Run optimization
    results = optimizer.optimize(
        observed=observed,
        method='L-BFGS-B',
        maxiter=50
    )
    
    # Save results
    optimizer.save_results(results, 'optimization_results.json')
    
    # Plot
    plot_optimization_history(results['history'], 'optimization_history.png')
    
    print("\n✓ Test complete!")