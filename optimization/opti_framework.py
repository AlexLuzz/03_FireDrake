"""
Parameter Optimization for Richards Equation
Inverse modeling to calibrate soil hydraulic parameters
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from scipy.optimize import minimize, differential_evolution, Bounds
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from firedrake import *
from physics import SoilMaterial
from solver import RichardsSolver
from config import SimulationConfig


# ============================================================
# PARAMETER DEFINITION
# ============================================================

@dataclass
class OptimizationParameter:
    """Definition of a parameter to optimize"""
    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    log_scale: bool = False  # Optimize in log space
    material_name: Optional[str] = None  # Which material this belongs to
    
    def to_physical(self, normalized_value: float) -> float:
        """Convert from [0,1] normalized to physical space"""
        if self.log_scale:
            log_lower = np.log10(self.lower_bound)
            log_upper = np.log10(self.upper_bound)
            log_value = log_lower + normalized_value * (log_upper - log_lower)
            return 10**log_value
        else:
            return self.lower_bound + normalized_value * (self.upper_bound - self.lower_bound)
    
    def to_normalized(self, physical_value: float) -> float:
        """Convert from physical to [0,1] normalized space"""
        if self.log_scale:
            log_lower = np.log10(self.lower_bound)
            log_upper = np.log10(self.upper_bound)
            log_value = np.log10(physical_value)
            return (log_value - log_lower) / (log_upper - log_lower)
        else:
            return (physical_value - self.lower_bound) / (self.upper_bound - self.lower_bound)


@dataclass
class VanGenuchtenParams:
    """Van Genuchten parameter set"""
    alpha: float  # [1/m]
    n: float      # [-]
    theta_s: float  # [-]
    theta_r: float  # [-]
    K_s: float    # [m/s]
    
    @classmethod
    def create_optimization_params(cls, material_name: str = "soil") -> List[OptimizationParameter]:
        """Create optimization parameters for van Genuchten model"""
        return [
            OptimizationParameter(
                name=f"alpha_{material_name}",
                initial_value=2.0,
                lower_bound=0.5,
                upper_bound=10.0,
                log_scale=True,
                material_name=material_name
            ),
            OptimizationParameter(
                name=f"n_{material_name}",
                initial_value=1.5,
                lower_bound=1.1,
                upper_bound=3.0,
                log_scale=False,
                material_name=material_name
            ),
            OptimizationParameter(
                name=f"K_s_{material_name}",
                initial_value=1e-5,
                lower_bound=1e-7,
                upper_bound=1e-3,
                log_scale=True,
                material_name=material_name
            ),
        ]


# ============================================================
# OBSERVATION DATA
# ============================================================

@dataclass
class ObservationData:
    """Container for observation/measurement data"""
    times: np.ndarray  # Time points [s]
    values: np.ndarray  # Measured values
    locations: List[Tuple[float, float]]  # (x, y) coordinates
    variable: str  # 'water_table', 'saturation', 'pressure_head'
    weights: Optional[np.ndarray] = None  # Observation weights
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones_like(self.values)
        
        # Ensure shapes match
        assert len(self.times) == self.values.shape[0]
        assert self.values.shape[1] == len(self.locations)


# ============================================================
# COST FUNCTION (Using JAX for efficiency)
# ============================================================

class CostFunction:
    """Base class for cost functions"""
    
    def __init__(self, observations: List[ObservationData]):
        self.observations = observations
    
    def compute(self, simulated: Dict[str, np.ndarray], 
                observations: List[ObservationData]) -> float:
        """Compute cost (lower is better)"""
        raise NotImplementedError


class L2Cost(CostFunction):
    """Weighted L2 norm (least squares)"""
    
    def compute(self, simulated: Dict[str, np.ndarray]) -> float:
        """
        Compute weighted L2 cost
        
        Parameters:
        -----------
        simulated : dict
            Dictionary with keys matching observation variable names
            Each value is array of shape (n_times, n_locations)
        """
        total_cost = 0.0
        
        for obs in self.observations:
            sim_values = simulated[obs.variable]
            
            # Ensure dimensions match
            if sim_values.shape != obs.values.shape:
                raise ValueError(f"Shape mismatch: simulated {sim_values.shape} vs observed {obs.values.shape}")
            
            # Weighted squared difference
            diff = sim_values - obs.values
            weighted_diff = diff * obs.weights
            cost = np.sum(weighted_diff**2)
            
            total_cost += cost
        
        # Normalize by number of observations
        n_obs = sum(obs.values.size for obs in self.observations)
        return total_cost / n_obs


class RMSECost(CostFunction):
    """Root Mean Square Error"""
    
    def compute(self, simulated: Dict[str, np.ndarray]) -> float:
        total_squared_error = 0.0
        n_total = 0
        
        for obs in self.observations:
            sim_values = simulated[obs.variable]
            diff = (sim_values - obs.values) * obs.weights
            total_squared_error += np.sum(diff**2)
            n_total += obs.values.size
        
        return np.sqrt(total_squared_error / n_total)


class NashSutcliffeCost(CostFunction):
    """Nash-Sutcliffe Efficiency (NSE) - converted to minimization"""
    
    def compute(self, simulated: Dict[str, np.ndarray]) -> float:
        """Returns (1 - NSE) so minimization → NSE = 1"""
        total_nse = 0.0
        
        for obs in self.observations:
            sim_values = simulated[obs.variable]
            
            # Mean of observations
            obs_mean = np.mean(obs.values)
            
            # NSE = 1 - sum((obs - sim)^2) / sum((obs - mean)^2)
            numerator = np.sum((obs.values - sim_values)**2)
            denominator = np.sum((obs.values - obs_mean)**2)
            
            nse = 1.0 - numerator / denominator
            total_nse += nse
        
        # Return negative (for minimization)
        return -total_nse / len(self.observations)


# ============================================================
# FORWARD MODEL RUNNER
# ============================================================

class ForwardModelRunner:
    """Runs the Richards solver for parameter estimation"""
    
    def __init__(self, 
                 base_config: SimulationConfig,
                 base_domain_setup: Callable,
                 observation_data: List[ObservationData],
                 coarse_mesh: bool = True):
        """
        Parameters:
        -----------
        base_config : SimulationConfig
            Base configuration (will be modified for coarse runs)
        base_domain_setup : callable
            Function that sets up domain: setup(mesh, params) -> domain
        observation_data : list
            List of ObservationData objects
        coarse_mesh : bool
            Use coarser mesh for faster computation
        """
        self.base_config = base_config
        self.base_domain_setup = base_domain_setup
        self.observations = observation_data
        self.coarse_mesh = coarse_mesh
        
        # Cache for results
        self.cache = {}
        self.n_evaluations = 0
    
    def create_config(self, coarse: bool = True) -> SimulationConfig:
        """Create configuration for forward model"""
        config = SimulationConfig(
            dt=self.base_config.dt if not coarse else self.base_config.dt * 2,
            t_end=self.base_config.t_end,
            output_dir=self.base_config.output_dir / "optimization",
            probes_positions=[]  # We'll extract at observation points
        )
        return config
    
    def run(self, parameters: Dict[str, float], verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        Run forward model with given parameters
        
        Parameters:
        -----------
        parameters : dict
            Dictionary of parameter names to values
        
        Returns:
        --------
        results : dict
            Dictionary with same structure as observations
        """
        self.n_evaluations += 1
        
        # Check cache
        param_key = tuple(sorted(parameters.items()))
        if param_key in self.cache:
            if verbose:
                print(f"[Eval {self.n_evaluations}] Using cached result")
            return self.cache[param_key]
        
        if verbose:
            print(f"\n[Eval {self.n_evaluations}] Running forward model with:")
            for name, value in parameters.items():
                print(f"  {name} = {value:.6e}")
        
        # Create mesh
        if self.coarse_mesh:
            mesh = RectangleMesh(40, 20, 20.0, 5.0)  # Coarser
        else:
            mesh = RectangleMesh(80, 40, 20.0, 5.0)  # Original
        
        V = FunctionSpace(mesh, "CG", 1)
        
        # Setup domain with new parameters
        domain = self.base_domain_setup(mesh, parameters)
        
        # Get observation times
        all_times = set()
        for obs in self.observations:
            all_times.update(obs.times)
        observation_times = sorted(all_times)
        
        # Setup solver (simplified - no rain for now)
        from physics import RainScenario
        from solver import BoundaryConditionManager
        
        rain_event = RainScenario(events=[])  # No rain for calibration
        bc_manager = BoundaryConditionManager(V, initial_water_table=1.2)
        
        config = self.create_config(coarse=self.coarse_mesh)
        solver = RichardsSolver(mesh, V, domain, rain_event, bc_manager, config)
        
        # Extract values at observation locations
        from interpolation import PointInterpolator
        results = {obs.variable: [] for obs in self.observations}
        
        # Create interpolators for each observation location
        interpolators = {}
        for obs in self.observations:
            if obs.variable not in interpolators:
                interpolators[obs.variable] = []
            for loc in obs.locations:
                interpolators[obs.variable].append(
                    PointInterpolator(mesh, loc)
                )
        
        # Run simulation and extract at observation times
        t = 0.0
        time_idx = 0
        
        while t <= config.t_end and time_idx < len(observation_times):
            target_time = observation_times[time_idx]
            
            # Step to target time
            while t < target_time:
                dt = min(config.dt, target_time - t)
                solver.solve_timestep(dt)
                t += dt
            
            # Extract values at observation locations
            for obs in self.observations:
                values_at_time = []
                for interp in interpolators[obs.variable]:
                    if obs.variable == 'pressure_head':
                        val = interp.interpolate(solver.h_n)
                    elif obs.variable == 'saturation':
                        val = interp.interpolate(solver.get_saturation())
                    elif obs.variable == 'water_table':
                        # Approximate water table depth
                        val = 5.0 - interp.interpolate(solver.h_n)
                    values_at_time.append(val)
                
                results[obs.variable].append(values_at_time)
            
            time_idx += 1
        
        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])
        
        # Cache results
        self.cache[param_key] = results
        
        if verbose:
            print(f"  Completed in {self.n_evaluations} evaluations")
        
        return results


# ============================================================
# OPTIMIZATION PROBLEM
# ============================================================

class OptimizationProblem:
    """Main optimization problem class"""
    
    def __init__(self,
                 parameters: List[OptimizationParameter],
                 observations: List[ObservationData],
                 forward_model: ForwardModelRunner,
                 cost_function: CostFunction):
        """
        Parameters:
        -----------
        parameters : list
            List of OptimizationParameter objects
        observations : list
            List of ObservationData objects
        forward_model : ForwardModelRunner
            Forward model runner
        cost_function : CostFunction
            Cost function to minimize
        """
        self.parameters = parameters
        self.observations = observations
        self.forward_model = forward_model
        self.cost_function = cost_function
        
        # History
        self.history = {
            'parameters': [],
            'cost': [],
            'iteration': []
        }
        self.iteration = 0
    
    def objective_function(self, x_normalized: np.ndarray, verbose: bool = False) -> float:
        """
        Objective function to minimize
        
        Parameters:
        -----------
        x_normalized : array
            Normalized parameter values in [0, 1]
        
        Returns:
        --------
        cost : float
            Cost function value
        """
        self.iteration += 1
        
        # Convert to physical parameters
        params_dict = {}
        for i, param in enumerate(self.parameters):
            physical_value = param.to_physical(x_normalized[i])
            params_dict[param.name] = physical_value
        
        # Run forward model
        try:
            simulated = self.forward_model.run(params_dict, verbose=verbose)
        except Exception as e:
            print(f"Forward model failed: {e}")
            return 1e10  # Large penalty
        
        # Compute cost
        cost = self.cost_function.compute(simulated)
        
        # Store history
        self.history['parameters'].append(params_dict.copy())
        self.history['cost'].append(cost)
        self.history['iteration'].append(self.iteration)
        
        if verbose or self.iteration % 10 == 0:
            print(f"\nIteration {self.iteration}: Cost = {cost:.6e}")
            for name, value in params_dict.items():
                print(f"  {name} = {value:.6e}")
        
        return cost
    
    def get_bounds(self) -> Bounds:
        """Get parameter bounds for scipy optimizers"""
        return Bounds(
            lb=np.zeros(len(self.parameters)),
            ub=np.ones(len(self.parameters))
        )
    
    def get_initial_guess(self) -> np.ndarray:
        """Get initial parameter guess (normalized)"""
        return np.array([
            param.to_normalized(param.initial_value)
            for param in self.parameters
        ])


# ============================================================
# OPTIMIZATION ALGORITHMS
# ============================================================

class Optimizer:
    """Base optimizer class"""
    
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
    
    def optimize(self) -> Dict:
        """Run optimization"""
        raise NotImplementedError


class ScipyOptimizer(Optimizer):
    """Scipy-based optimization"""
    
    def __init__(self, problem: OptimizationProblem, method: str = 'L-BFGS-B'):
        """
        Parameters:
        -----------
        method : str
            Scipy optimization method: 'L-BFGS-B', 'SLSQP', 'TNC'
        """
        super().__init__(problem)
        self.method = method
    
    def optimize(self, maxiter: int = 100) -> Dict:
        """Run scipy optimization"""
        print(f"\n{'='*60}")
        print(f"Starting {self.method} optimization")
        print(f"{'='*60}")
        
        x0 = self.problem.get_initial_guess()
        bounds = self.problem.get_bounds()
        
        result = minimize(
            fun=self.problem.objective_function,
            x0=x0,
            method=self.method,
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': True}
        )
        
        # Convert to physical parameters
        optimal_params = {}
        for i, param in enumerate(self.problem.parameters):
            optimal_params[param.name] = param.to_physical(result.x[i])
        
        return {
            'optimal_parameters': optimal_params,
            'optimal_cost': result.fun,
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit,
            'history': self.problem.history
        }


class DifferentialEvolutionOptimizer(Optimizer):
    """Differential Evolution (global optimization)"""
    
    def optimize(self, maxiter: int = 50, popsize: int = 15) -> Dict:
        """
        Run differential evolution
        
        Parameters:
        -----------
        maxiter : int
            Maximum generations
        popsize : int
            Population size (popsize * n_parameters)
        """
        print(f"\n{'='*60}")
        print(f"Starting Differential Evolution")
        print(f"{'='*60}")
        
        bounds = [(0, 1) for _ in self.problem.parameters]
        
        result = differential_evolution(
            func=self.problem.objective_function,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            disp=True,
            workers=1,  # Parallel evaluation possible
            updating='deferred'
        )
        
        # Convert to physical parameters
        optimal_params = {}
        for i, param in enumerate(self.problem.parameters):
            optimal_params[param.name] = param.to_physical(result.x[i])
        
        return {
            'optimal_parameters': optimal_params,
            'optimal_cost': result.fun,
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit,
            'history': self.problem.history
        }


# ============================================================
# RESULTS VISUALIZATION
# ============================================================

class OptimizationVisualizer:
    """Visualize optimization results"""
    
    @staticmethod
    def plot_convergence(history: Dict, filename: Optional[Path] = None):
        """Plot convergence history"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        iterations = history['iteration']
        costs = history['cost']
        
        # Cost evolution
        axes[0].plot(iterations, costs, 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Cost Function')
        axes[0].set_title('Optimization Convergence')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Parameter evolution
        param_names = list(history['parameters'][0].keys())
        for param_name in param_names:
            values = [p[param_name] for p in history['parameters']]
            axes[1].plot(iterations, values, label=param_name, linewidth=2)
        
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Parameter Value')
        axes[1].set_title('Parameter Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved convergence plot to {filename}")
        else:
            plt.show()
    
    @staticmethod
    def plot_fit_quality(observations: List[ObservationData],
                        simulated: Dict[str, np.ndarray],
                        filename: Optional[Path] = None):
        """Plot observed vs simulated data"""
        n_obs = len(observations)
        fig, axes = plt.subplots(n_obs, 1, figsize=(12, 4*n_obs))
        
        if n_obs == 1:
            axes = [axes]
        
        for i, obs in enumerate(observations):
            ax = axes[i]
            
            sim_data = simulated[obs.variable]
            
            # Plot each location
            for j, loc in enumerate(obs.locations):
                ax.plot(obs.times / 3600, obs.values[:, j], 
                       'o', label=f'Observed (x={loc[0]:.1f}m)', 
                       markersize=8, alpha=0.7)
                ax.plot(obs.times / 3600, sim_data[:, j], 
                       '-', linewidth=2, 
                       label=f'Simulated (x={loc[0]:.1f}m)')
            
            ax.set_xlabel('Time [hours]')
            ax.set_ylabel(obs.variable.replace('_', ' ').title())
            ax.set_title(f'{obs.variable} - Observations vs Simulation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved fit quality plot to {filename}")
        else:
            plt.show()


# ============================================================
# UTILITY: POINT INTERPOLATOR
# ============================================================

class PointInterpolator:
    """Interpolate field values at specific points"""
    
    def __init__(self, mesh, point: Tuple[float, float]):
        """
        Parameters:
        -----------
        mesh : Mesh
            FireDrake mesh
        point : tuple
            (x, y) coordinates
        """
        self.point = point
        self.mesh = mesh
    
    def interpolate(self, field: Function) -> float:
        """Interpolate field value at point"""
        try:
            return field.at(self.point)
        except:
            # Fallback: nearest neighbor
            V = field.function_space()
            coords = V.mesh().coordinates.dat.data
            values = field.dat.data
            
            # Find nearest point
            distances = np.sum((coords - np.array(self.point))**2, axis=1)
            nearest_idx = np.argmin(distances)
            
            return values[nearest_idx]


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_optimization():
    """Example: Optimize van Genuchten parameters"""
    
    # 1. Create synthetic observations (in practice, load from file)
    times = np.array([0, 12, 24, 48, 96]) * 3600  # hours to seconds
    locations = [(5.0, 2.5), (10.0, 2.5), (15.0, 2.5)]
    
    # Synthetic water table observations
    observations = [
        ObservationData(
            times=times,
            values=np.array([
                [1.2, 1.2, 1.2],  # t=0
                [1.5, 1.4, 1.3],  # t=12h
                [1.8, 1.6, 1.4],  # t=24h
                [2.0, 1.8, 1.6],  # t=48h
                [2.1, 1.9, 1.7],  # t=96h
            ]),
            locations=locations,
            variable='water_table'
        )
    ]
    
    # 2. Define parameters to optimize
    parameters = VanGenuchtenParams.create_optimization_params("Till")
    
    # 3. Setup forward model
    def domain_setup(mesh, params):
        """Create domain with given parameters"""
        # Extract parameters
        alpha = params.get('alpha_Till', 2.0)
        n = params.get('n_Till', 1.5)
        K_s = params.get('K_s_Till', 1e-5)
        
        # Create material (simplified)
        from physics import Domain
        material = SoilMaterial.from_curves(name="Till")
        # Update parameters (you'll need to modify SoilMaterial to accept custom params)
        
        domain = Domain.homogeneous(mesh, material)
        return domain
    
    base_config = SimulationConfig(dt=3600, t_end=96*3600)
    
    forward_model = ForwardModelRunner(
        base_config=base_config,
        base_domain_setup=domain_setup,
        observation_data=observations,
        coarse_mesh=True
    )
    
    # 4. Setup optimization problem
    cost_function = L2Cost(observations)
    
    problem = OptimizationProblem(
        parameters=parameters,
        observations=observations,
        forward_model=forward_model,
        cost_function=cost_function
    )
    
    # 5. Run optimization
    optimizer = DifferentialEvolutionOptimizer(problem)
    # optimizer = ScipyOptimizer(problem, method='L-BFGS-B')
    
    results = optimizer.optimize(maxiter=30)
    
    # 6. Visualize results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Optimal cost: {results['optimal_cost']:.6e}")
    print("\nOptimal parameters:")
    for name, value in results['optimal_parameters'].items():
        print(f"  {name} = {value:.6e}")
    
    # Plot convergence
    OptimizationVisualizer.plot_convergence(
        results['history'],
        filename=Path('optimization_convergence.png')
    )
    
    # Run final simulation with optimal parameters
    final_simulated = forward_model.run(
        results['optimal_parameters'],
        verbose=True
    )
    
    # Plot fit quality
    OptimizationVisualizer.plot_fit_quality(
        observations,
        final_simulated,
        filename=Path('optimization_fit.png')
    )
    
    # Save results
    output = {
        'optimal_parameters': results['optimal_parameters'],
        'optimal_cost': float(results['optimal_cost']),
        'n_iterations': results['n_iterations'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open('optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to optimization_results.json")


if __name__ == "__main__":
    example_optimization()