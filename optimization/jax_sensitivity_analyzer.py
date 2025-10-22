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
from firedrake.ml.jax.fem_operator import FiredrakeJaxOperator, to_jax, from_jax

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
        
        # Create function that returns theta at each point
        if param_name == 'alpha':
            def theta_at_param(alpha):
                return self.van_genuchten_saturation(
                    h_jax, alpha, 
                    base_params['n'],
                    base_params.get('theta_s', 0.45),
                    base_params.get('theta_r', 0.067)
                )
            param_value = base_params['alpha']
            
        elif param_name == 'n':
            def theta_at_param(n):
                return self.van_genuchten_saturation(
                    h_jax,
                    base_params['alpha'],
                    n,
                    base_params.get('theta_s', 0.45),
                    base_params.get('theta_r', 0.067)
                )
            param_value = base_params['n']
            
        elif param_name == 'K_s':
            def theta_at_param(K_s):
                return self.van_genuchten_conductivity(
                    h_jax,
                    base_params['alpha'],
                    base_params['n'],
                    K_s
                )
            param_value = base_params['K_s']
        
        # Use jacfwd to get Jacobian (gradient at each point)
        # This gives us ∂θ_i/∂p for all i
        jac_func = jacfwd(theta_at_param)
        sensitivity = jac_func(param_value)
        
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

def validate_jax_gradients():
    """Validate JAX gradients against finite differences"""
    print("\n" + "="*60)
    print("Validating JAX Automatic Differentiation")
    print("="*60)
    
    analyzer = JaxSensitivityAnalyzer()
    
    # Test point
    h_test = np.array([-2.0])
    base_params = {
        'alpha': 2.0,
        'n': 1.5,
        'theta_s': 0.45,
        'theta_r': 0.067
    }
    
    # JAX gradient
    sens_jax = analyzer.compute_parameter_sensitivities(h_test, base_params, 'alpha')
    
    # Finite difference gradient
    epsilon = 1e-6
    params_plus = base_params.copy()
    params_plus['alpha'] = base_params['alpha'] + epsilon
    
    params_minus = base_params.copy()
    params_minus['alpha'] = base_params['alpha'] - epsilon
    
    theta_plus = analyzer.van_genuchten_saturation(
        jnp.array(h_test), params_plus['alpha'], params_plus['n'],
        params_plus['theta_s'], params_plus['theta_r']
    )
    
    theta_minus = analyzer.van_genuchten_saturation(
        jnp.array(h_test), params_minus['alpha'], params_minus['n'],
        params_minus['theta_s'], params_minus['theta_r']
    )
    
    sens_fd = (theta_plus - theta_minus) / (2 * epsilon)
    
    # Compare
    print(f"\nTest point: h = {h_test[0]} m")
    print(f"Parameters: α={base_params['alpha']}, n={base_params['n']}")
    print(f"\nGradient ∂θ/∂α:")
    print(f"  JAX (autodiff):      {sens_jax[0]:.8e}")
    print(f"  Finite difference:   {sens_fd[0]:.8e}")
    print(f"  Relative difference: {abs(sens_jax[0] - sens_fd[0]) / abs(sens_fd[0]) * 100:.6f}%")
    
    if abs(sens_jax[0] - sens_fd[0]) / abs(sens_fd[0]) < 1e-4:
        print("\n✓ Validation PASSED: JAX gradients are accurate!")
    else:
        print("\n✗ Validation FAILED: Large discrepancy detected!")
    
    print("="*60)


def example_jax_sensitivity():
    """Example: JAX-based local sensitivity"""
    
    print("="*60)
    print("JAX Sensitivity Analysis Example")
    print("="*60)
    
    # First, validate gradients
    validate_jax_gradients()
    
    analyzer = JaxSensitivityAnalyzer()
    
    # Pressure head profile (from saturated to dry)
    h = np.linspace(-5, 0, 100)
    
    # Base parameters (typical sandy loam)
    base_params = {
        'alpha': 2.0,      # 1/m
        'n': 1.5,          # -
        'K_s': 1e-5,       # m/s
        'theta_s': 0.45,   # -
        'theta_r': 0.067   # -
    }
    
    print("\nBase parameters:")
    for key, val in base_params.items():
        print(f"  {key}: {val}")
    
    # Compute saturation and conductivity
    print("\nComputing retention curve...")
    theta = np.array(analyzer.van_genuchten_saturation(
        jnp.array(h),
        base_params['alpha'],
        base_params['n'],
        base_params['theta_s'],
        base_params['theta_r']
    ))
    
    print("Computing hydraulic conductivity...")
    K = np.array(analyzer.van_genuchten_conductivity(
        jnp.array(h),
        base_params['alpha'],
        base_params['n'],
        base_params['K_s']
    ))
    
    # Compute sensitivities
    print("\nComputing sensitivities...")
    print("  ∂θ/∂α...")
    sens_theta_alpha = analyzer.compute_parameter_sensitivities(h, base_params, 'alpha')
    print("  ∂θ/∂n...")
    sens_theta_n = analyzer.compute_parameter_sensitivities(h, base_params, 'n')
    
    # Compute Jacobian (all sensitivities)
    print("\nComputing full Jacobian...")
    jacobian = analyzer.compute_jacobian(h, base_params)
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Retention curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(h, theta, 'b-', linewidth=2)
    ax1.set_xlabel('Pressure head h [m]')
    ax1.set_ylabel('Water content θ [-]')
    ax1.set_title('Water Retention Curve')
    ax1.grid(alpha=0.3)
    ax1.axhline(base_params['theta_s'], color='gray', linestyle='--', alpha=0.5, label='θs')
    ax1.axhline(base_params['theta_r'], color='gray', linestyle='--', alpha=0.5, label='θr')
    ax1.legend()
    
    # 2. Hydraulic conductivity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(h, K, 'r-', linewidth=2)
    ax2.set_xlabel('Pressure head h [m]')
    ax2.set_ylabel('Hydraulic conductivity K [m/s]')
    ax2.set_title('Hydraulic Conductivity')
    ax2.grid(alpha=0.3)
    ax2.axhline(base_params['K_s'], color='gray', linestyle='--', alpha=0.5, label='Ks')
    ax2.legend()
    
    # 3. Sensitivity ∂θ/∂α
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(h, sens_theta_alpha, 'g-', linewidth=2)
    ax3.set_xlabel('Pressure head h [m]')
    ax3.set_ylabel('∂θ/∂α [m]')
    ax3.set_title('Sensitivity to α parameter')
    ax3.grid(alpha=0.3)
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    # 4. Sensitivity ∂θ/∂n
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(h, sens_theta_n, 'orange', linewidth=2)
    ax4.set_xlabel('Pressure head h [m]')
    ax4.set_ylabel('∂θ/∂n [-]')
    ax4.set_title('Sensitivity to n parameter')
    ax4.grid(alpha=0.3)
    ax4.axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    # 5. Combined sensitivities (normalized)
    ax5 = fig.add_subplot(gs[2, :])
    # Normalize sensitivities for comparison
    sens_alpha_norm = sens_theta_alpha / np.max(np.abs(sens_theta_alpha))
    sens_n_norm = sens_theta_n / np.max(np.abs(sens_theta_n))
    
    ax5.plot(h, sens_alpha_norm, 'g-', linewidth=2, label='∂θ/∂α (normalized)')
    ax5.plot(h, sens_n_norm, 'orange', linewidth=2, label='∂θ/∂n (normalized)')
    ax5.set_xlabel('Pressure head h [m]')
    ax5.set_ylabel('Normalized sensitivity [-]')
    ax5.set_title('Relative Parameter Sensitivities')
    ax5.legend()
    ax5.grid(alpha=0.3)
    ax5.axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    # Add text with max sensitivity values
    max_sens_alpha = np.max(np.abs(sens_theta_alpha))
    max_sens_n = np.max(np.abs(sens_theta_n))
    h_max_alpha = h[np.argmax(np.abs(sens_theta_alpha))]
    h_max_n = h[np.argmax(np.abs(sens_theta_n))]
    
    textstr = f'Max |∂θ/∂α| = {max_sens_alpha:.3e} at h = {h_max_alpha:.2f} m\n'
    textstr += f'Max |∂θ/∂n| = {max_sens_n:.3e} at h = {h_max_n:.2f} m'
    ax5.text(0.02, 0.98, textstr, transform=ax5.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Van Genuchten Model - Sensitivity Analysis', fontsize=14, fontweight='bold')
    
    plt.savefig('jax_sensitivity_example.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print("Saved sensitivity plot to 'jax_sensitivity_example.png'")
    print("="*60)
    
    # Print summary statistics
    print("\nSensitivity Summary:")
    print("-"*60)
    print(f"{'Parameter':<15} {'Max |∂θ/∂p|':<15} {'h at max':<15}")
    print("-"*60)
    print(f"{'alpha':<15} {max_sens_alpha:<15.3e} {h_max_alpha:<15.2f}")
    print(f"{'n':<15} {max_sens_n:<15.3e} {h_max_n:<15.2f}")
    print("-"*60)


if __name__ == "__main__":
    example_jax_sensitivity()