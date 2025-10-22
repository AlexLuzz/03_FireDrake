"""
JAX-accelerated sensitivity analysis and gradient computation
for parameter optimization
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev
from jax.experimental import sparse
import numpy as np
from typing import Callable, Dict, Tuple, List
import matplotlib.pyplot as plt
from functools import partial


# ============================================================
# JAX-BASED SENSITIVITY ANALYSIS
# ============================================================

class JaxSensitivityAnalyzer:
    """
    Compute parameter sensitivities using JAX automatic differentiation
    
    Note: This works for post-processing and local sensitivity analysis.
    For full gradient-based optimization with FireDrake, we need adjoint methods.
    """
    
    def __init__(self):
        self.cache = {}
    
    @staticmethod
    @jit
    def van_genuchten_saturation(h: jnp.ndarray, 
                                 alpha: float, 
                                 n: float,
                                 theta_s: float = 0.45,
                                 theta_r: float = 0.067) -> jnp.ndarray:
        """
        Van Genuchten saturation function (JAX-compatible)
        
        S_e = [1 + (alpha * |h|)^n]^(-m)
        theta = theta_r + (theta_s - theta_r) * S_e
        """
        m = 1.0 - 1.0 / n
        
        # Effective saturation
        Se = jnp.where(
            h < 0,
            jnp.power(1.0 + jnp.power(alpha * jnp.abs(h), n), -m),
            1.0
        )
        
        # Actual saturation
        theta = theta_r + (theta_s - theta_r) * Se
        return theta
    
    @staticmethod
    @jit
    def van_genuchten_conductivity(h: jnp.ndarray,
                                   alpha: float,
                                   n: float,
                                   K_s: float = 1e-5) -> jnp.ndarray:
        """Van Genuchten hydraulic conductivity (JAX-compatible)"""
        m = 1.0 - 1.0 / n
        
        Se = jnp.where(
            h < 0,
            jnp.power(1.0 + jnp.power(alpha * jnp.abs(h), n), -m),
            1.0
        )
        
        # Mualem model
        K_rel = jnp.sqrt(Se) * jnp.power(
            1.0 - jnp.power(1.0 - jnp.power(Se, 1.0/m), m), 2
        )
        
        return K_s * K_rel
    
    def compute_parameter_sensitivities(self,
                                       h_profile: np.ndarray,
                                       base_params: Dict[str, float],
                                       param_name: str) -> np.ndarray:
        """
        Compute sensitivity of saturation to parameter changes
        
        ∂θ/∂p where p is the parameter
        
        Parameters:
        -----------
        h_profile : array
            Pressure head profile
        base_params : dict
            Base parameter values
        param_name : str
            Parameter to compute sensitivity for
        
        Returns:
        --------
        sensitivity : array
            ∂θ/∂p at each point in h_profile
        """
        h_jax = jnp.array(h_profile)
        
        # Create sensitivity function
        if param_name == 'alpha':
            def theta_func(alpha):
                return self.van_genuchten_saturation(
                    h_jax, alpha, 
                    base_params['n'],
                    base_params.get('theta_s', 0.45),
                    base_params.get('theta_r', 0.067)
                )
            grad_func = grad(lambda p: jnp.sum(theta_func(p)))
            param_value = base_params['alpha']
            
        elif param_name == 'n':
            def theta_func(n):
                return self.van_genuchten_saturation(
                    h_jax,
                    base_params['alpha'],
                    n,
                    base_params.get('theta_s', 0.45),
                    base_params.get('theta_r', 0.067)
                )
            grad_func = grad(lambda p: jnp.sum(theta_func(p)))
            param_value = base_params['n']
            
        elif param_name == 'K_s':
            def K_func(K_s):
                return self.van_genuchten_conductivity(
                    h_jax,
                    base_params['alpha'],
                    base_params['n'],
                    K_s
                )
            grad_func = grad(lambda p: jnp.sum(K_func(p)))
            param_value = base_params['K_s']
        
        # Compute gradient using JAX AD
        sensitivity = grad_func(param_value)
        
        return np.array(sensitivity)
    
    def compute_jacobian(self,
                        h_profile: np.ndarray,
                        params: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Compute Jacobian: all parameter sensitivities at once
        
        Returns dict of ∂θ/∂p for each parameter p
        """
        jacobian = {}
        
        for param_name in ['alpha', 'n', 'K_s']:
            if param_name in params:
                jacobian[param_name] = self.compute_parameter_sensitivities(
                    h_profile, params, param_name
                )
        
        return jacobian


# ============================================================
# FINITE DIFFERENCE GRADIENTS (Fallback)
# ============================================================

class FiniteDifferenceGradient:
    """
    Compute gradients using finite differences
    Useful when JAX can't be used directly with FireDrake
    """
    
    def __init__(self, forward_model: Callable, epsilon: float = 1e-6):
        """
        Parameters:
        -----------
        forward_model : callable
            Function: params -> cost
        epsilon : float
            Finite difference step size (relative)
        """
        self.forward_model = forward_model
        self.epsilon = epsilon
        self.cache = {}
    
    def compute_gradient(self, 
                        params: Dict[str, float],
                        param_names: List[str]) -> Dict[str, float]:
        """
        Compute gradient using central differences
        
        ∂f/∂p ≈ [f(p + ε) - f(p - ε)] / (2ε)
        """
        base_cost = self.forward_model(params)
        gradients = {}
        
        for param_name in param_names:
            base_value = params[param_name]
            
            # Relative step
            h = base_value * self.epsilon
            if abs(h) < 1e-10:
                h = self.epsilon
            
            # Forward perturbation
            params_plus = params.copy()
            params_plus[param_name] = base_value + h
            cost_plus = self.forward_model(params_plus)
            
            # Backward perturbation
            params_minus = params.copy()
            params_minus[param_name] = base_value - h
            cost_minus = self.forward_model(params_minus)
            
            # Central difference
            gradient = (cost_plus - cost_minus) / (2 * h)
            gradients[param_name] = gradient
        
        return gradients


# ============================================================
# MORRIS SENSITIVITY (Global Screening)
# ============================================================

class MorrisSensitivity:
    """
    Morris One-at-a-Time (OAT) global sensitivity analysis
    
    Efficient screening method to identify important parameters
    """
    
    def __init__(self, 
                 forward_model: Callable,
                 param_bounds: Dict[str, Tuple[float, float]],
                 n_trajectories: int = 10,
                 n_levels: int = 4):
        """
        Parameters:
        -----------
        forward_model : callable
            Function: params_dict -> cost
        param_bounds : dict
            {param_name: (lower, upper)}
        n_trajectories : int
            Number of random trajectories
        n_levels : int
            Number of levels in parameter space
        """
        self.forward_model = forward_model
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.n_trajectories = n_trajectories
        self.n_levels = n_levels
        
        self.results = None
    
    def _generate_trajectory(self) -> np.ndarray:
        """Generate one Morris trajectory"""
        # Random starting point
        x = np.random.randint(0, self.n_levels - 1, self.n_params)
        
        # Random order of parameter changes
        order = np.random.permutation(self.n_params)
        
        # Trajectory: (n_params + 1) x n_params matrix
        trajectory = np.zeros((self.n_params + 1, self.n_params))
        trajectory[0] = x
        
        # Change one parameter at a time
        for i, param_idx in enumerate(order):
            trajectory[i + 1] = trajectory[i].copy()
            delta = np.random.choice([-1, 1])
            trajectory[i + 1, param_idx] += delta
            
            # Keep in bounds
            trajectory[i + 1, param_idx] = np.clip(
                trajectory[i + 1, param_idx], 0, self.n_levels - 1
            )
        
        return trajectory
    
    def _normalize_params(self, x: np.ndarray) -> Dict[str, float]:
        """Convert normalized trajectory point to physical parameters"""
        params = {}
        for i, name in enumerate(self.param_names):
            lower, upper = self.param_bounds[name]
            # Normalize from [0, n_levels-1] to [lower, upper]
            normalized = x[i] / (self.n_levels - 1)
            params[name] = lower + normalized * (upper - lower)
        
        return params
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Run Morris sensitivity analysis
        
        Returns:
        --------
        results : dict
            'mu': mean elementary effects
            'mu_star': mean absolute elementary effects
            'sigma': standard deviation of elementary effects
        """
        if verbose:
            print("Running Morris sensitivity analysis...")
            print(f"Parameters: {self.param_names}")
            print(f"Trajectories: {self.n_trajectories}")
        
        # Storage for elementary effects
        elementary_effects = {name: [] for name in self.param_names}
        
        # Run trajectories
        for traj_idx in range(self.n_trajectories):
            if verbose:
                print(f"\nTrajectory {traj_idx + 1}/{self.n_trajectories}")
            
            trajectory = self._generate_trajectory()
            
            # Evaluate along trajectory
            costs = []
            for point in trajectory:
                params = self._normalize_params(point)
                cost = self.forward_model(params)
                costs.append(cost)
            
            # Compute elementary effects
            for i in range(self.n_params):
                param_name = self.param_names[i]
                
                # Find where this parameter changed
                for j in range(len(trajectory) - 1):
                    if trajectory[j + 1, i] != trajectory[j, i]:
                        # Elementary effect
                        delta_cost = costs[j + 1] - costs[j]
                        delta_param = trajectory[j + 1, i] - trajectory[j, i]
                        
                        ee = delta_cost / delta_param
                        elementary_effects[param_name].append(ee)
                        break
        
        # Compute statistics
        results = {}
        for name in self.param_names:
            ee_array = np.array(elementary_effects[name])
            results[name] = {
                'mu': np.mean(ee_array),
                'mu_star': np.mean(np.abs(ee_array)),
                'sigma': np.std(ee_array),
                'elementary_effects': ee_array
            }
        
        self.results = results
        
        if verbose:
            print("\n" + "="*60)
            print("Morris Sensitivity Results")
            print("="*60)
            print(f"{'Parameter':<15} {'μ*':<12} {'σ':<12} {'μ*/σ':<12}")
            print("-"*60)
            
            for name in self.param_names:
                mu_star = results[name]['mu_star']
                sigma = results[name]['sigma']
                ratio = mu_star / sigma if sigma > 0 else 0
                print(f"{name:<15} {mu_star:>11.4e} {sigma:>11.4e} {ratio:>11.4f}")
        
        return results
    
    def plot_results(self, filename: str = None):
        """Plot Morris sensitivity results"""
        if self.results is None:
            raise ValueError("Run analysis first!")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot of mu_star
        names = list(self.results.keys())
        mu_stars = [self.results[n]['mu_star'] for n in names]
        sigmas = [self.results[n]['sigma'] for n in names]
        
        ax1.barh(names, mu_stars, color='steelblue')
        ax1.set_xlabel('μ* (Mean Absolute Elementary Effect)')
        ax1.set_title('Parameter Importance')
        ax1.grid(axis='x', alpha=0.3)
        
        # Scatter plot: mu_star vs sigma
        ax2.scatter(mu_stars, sigmas, s=100, alpha=0.7, c='coral')
        for i, name in enumerate(names):
            ax2.annotate(name, (mu_stars[i], sigmas[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('μ* (Importance)')
        ax2.set_ylabel('σ (Interactions/Non-linearity)')
        ax2.set_title('Morris Sensitivity Plot')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved Morris plot to {filename}")
        else:
            plt.show()


# ============================================================
# SOBOL SENSITIVITY (Variance-based)
# ============================================================

class SobolSensitivity:
    """
    Sobol variance-based global sensitivity analysis
    
    Computes first-order and total-order sensitivity indices
    """
    
    def __init__(self,
                 forward_model: Callable,
                 param_bounds: Dict[str, Tuple[float, float]],
                 n_samples: int = 1000):
        """
        Parameters:
        -----------
        forward_model : callable
            params_dict -> cost
        param_bounds : dict
            {param_name: (lower, upper)}
        n_samples : int
            Number of samples (actual evaluations = n_samples * (2 + n_params))
        """
        self.forward_model = forward_model
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.n_samples = n_samples
    
    def _generate_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two independent sample matrices"""
        A = np.random.uniform(0, 1, (self.n_samples, self.n_params))
        B = np.random.uniform(0, 1, (self.n_samples, self.n_params))
        return A, B
    
    def _scale_samples(self, samples: np.ndarray) -> List[Dict[str, float]]:
        """Scale samples to parameter bounds"""
        params_list = []
        for sample in samples:
            params = {}
            for i, name in enumerate(self.param_names):
                lower, upper = self.param_bounds[name]
                params[name] = lower + sample[i] * (upper - lower)
            params_list.append(params)
        return params_list
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Run Sobol sensitivity analysis
        
        Returns:
        --------
        results : dict
            'S1': First-order indices
            'ST': Total-order indices
        """
        if verbose:
            print("Running Sobol sensitivity analysis...")
            print(f"Parameters: {self.param_names}")
            print(f"Samples: {self.n_samples}")
            print(f"Total evaluations: {self.n_samples * (2 + self.n_params)}")
        
        # Generate samples
        A, B = self._generate_samples()
        
        # Evaluate model on A and B
        params_A = self._scale_samples(A)
        params_B = self._scale_samples(B)
        
        if verbose:
            print("\nEvaluating model on sample sets A and B...")
        
        f_A = np.array([self.forward_model(p) for p in params_A])
        f_B = np.array([self.forward_model(p) for p in params_B])
        
        # Compute variance
        f_mean = np.mean(np.concatenate([f_A, f_B]))
        f_var = np.var(np.concatenate([f_A, f_B]))
        
        if verbose:
            print(f"Mean output: {f_mean:.4e}")
            print(f"Variance: {f_var:.4e}")
        
        # First-order and total-order indices
        S1 = {}
        ST = {}
        
        for i, name in enumerate(self.param_names):
            if verbose:
                print(f"\nComputing indices for {name}...")
            
            # Create C^i matrix (B with i-th column from A)
            C_i = B.copy()
            C_i[:, i] = A[:, i]
            params_Ci = self._scale_samples(C_i)
            
            f_Ci = np.array([self.forward_model(p) for p in params_Ci])
            
            # First-order index
            S1[name] = np.mean(f_B * (f_Ci - f_A)) / f_var
            
            # Total-order index
            ST[name] = 1 - np.mean(f_A * (f_Ci - f_B)) / f_var
        
        results = {'S1': S1, 'ST': ST, 'variance': f_var}
        
        if verbose:
            print("\n" + "="*60)
            print("Sobol Sensitivity Results")
            print("="*60)
            print(f"{'Parameter':<15} {'S1 (First)':<15} {'ST (Total)':<15}")
            print("-"*60)
            
            for name in self.param_names:
                print(f"{name:<15} {S1[name]:>14.4f} {ST[name]:>14.4f}")
        
        return results


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_jax_sensitivity():
    """Example: JAX-based local sensitivity"""
    
    print("JAX Sensitivity Analysis Example\n")
    
    analyzer = JaxSensitivityAnalyzer()
    
    # Pressure head profile
    h = np.linspace(-5, 0, 100)
    
    # Base parameters
    base_params = {
        'alpha': 2.0,
        'n': 1.5,
        'K_s': 1e-5,
        'theta_s': 0.45,
        'theta_r': 0.067
    }
    
    # Compute saturation
    theta = analyzer.van_genuchten_saturation(
        jnp.array(h),
        base_params['alpha'],
        base_params['n'],
        base_params['theta_s'],
        base_params['theta_r']
    )
    
    # Compute sensitivities
    sens_alpha = analyzer.compute_parameter_sensitivities(h, base_params, 'alpha')
    sens_n = analyzer.compute_parameter_sensitivities(h, base_params, 'n')
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(h, theta, 'b-', linewidth=2)
    ax1.set_xlabel('Pressure head h [m]')
    ax1.set_ylabel('Saturation θ [-]')
    ax1.set_title('Water Retention Curve')
    ax1.grid(alpha=0.3)
    
    ax2.plot(h, sens_alpha, label='∂θ/∂α', linewidth=2)
    ax2.plot(h, sens_n, label='∂θ/∂n', linewidth=2)
    ax2.set_xlabel('Pressure head h [m]')
    ax2.set_ylabel('Sensitivity')
    ax2.set_title('Parameter Sensitivities')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('jax_sensitivity_example.png', dpi=150)
    print("Saved sensitivity plot")


if __name__ == "__main__":
    example_jax_sensitivity()