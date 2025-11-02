"""
Advanced Parameter Optimization using Firedrake Adjoint (pyadjoint)
This uses automatic differentiation instead of finite differences - much faster!
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Callable
from datetime import datetime, timedelta
from firedrake import *
from firedrake_adjoint import *
import matplotlib.pyplot as plt


class AdjointParameterOptimizer:
    """
    Parameter optimizer using Firedrake's tape-based automatic differentiation
    This is much more efficient than finite differences for large parameter spaces
    """
    
    def __init__(
        self,
        simulation_config: dict,
        observations: 'ObservationData',
        bounds: 'ParameterBounds',
        initial_params: Dict[str, float],
        material_zone: str = "base"
    ):
        self.config = simulation_config
        self.observations = observations
        self.bounds = bounds
        self.material_zone = material_zone
        
        # Convert parameters to Firedrake Controls
        self.controls = self._create_controls(initial_params)
        
        # History
        self.loss_history = []
        self.param_history = []
        
    def _create_controls(self, params: Dict[str, float]) -> Dict[str, Control]:
        """Create Firedrake Control objects for each parameter"""
        controls = {}
        
        for param_name, value in params.items():
            # Create a Constant for each parameter
            param_constant = Constant(value)
            # Wrap in Control for adjoint
            controls[param_name] = Control(param_constant)
        
        return controls
    
    def build_adjoint_simulation(self, controls: Dict[str, Control]) -> Tuple:
        """
        Build Richards simulation with Control objects for adjoint
        This version is tape-aware for automatic differentiation
        """
        from src import *
        
        # Extract control values
        params = {name: float(ctrl.control) for name, ctrl in controls.items()}
        
        # Setup simulation (similar to your existing code)
        config = SimulationConfig(**self.config)
        
        rain_zones = [
            {'name': 'grass', 'x_min': 0.0, 'x_max': 8.0, 'multiplier': 1.0},
            {'name': 'green_infrastructure', 'x_min': 9.0, 'x_max': 11.0, 'multiplier': 6.0},
        ]
        
        rain_source = rainfall_scenario(
            from_date=config.start_datetime,
            to_date=config.end_datetime,
            csv_path=config.data_input_dir / "BB_METEO.csv",
            rain_unit="mm/day",
            zones=rain_zones
        )
        
        domain = Domain(nx=140, ny=70, Lx=20.0, Ly=5.0)
        domain.add_rectangle("GI", 9.0, 11.0, 4.0, 5.0)

        # Create material with Control objects
        # NOTE: You may need to modify your Material class to accept Constants
        optimized_material = Material(
            theta_r=controls['theta_r'].control,  # Pass the Constant directly
            theta_s=controls['theta_s'].control,
            alpha=controls['alpha'].control,
            n=controls['n'].control,
            Ks=controls['Ks'].control
        )
        
        if self.material_zone == "base":
            domain.assign("base", optimized_material)
            domain.assign("GI", Material.terreau())
        else:
            domain.assign("base", Material.till())
            domain.assign("GI", optimized_material)

        V = FunctionSpace(domain.mesh, "CG", 1)
        field_map = MaterialField(domain, V)
        
        bc_manager = BoundaryConditionManager(V, left_wt=1.2, right_wt=1.2)
        
        probe_manager = ProbeManager(domain.mesh)
        snapshot_manager = SnapshotManager([0.0, config.t_end])
        
        solver = RichardsSolver(
            domain=domain, V=V, field_map=field_map,
            source_scenario=rain_source,
            bc_manager=bc_manager, config=config
        )
        
        # Run simulation (tape is recording!)
        solver.run(probe_manager, snapshot_manager)
        
        return solver, probe_manager, domain
    
    def compute_functional(self, probe_manager: 'ProbeManager') -> float:
        """
        Compute objective functional (misfit with observations)
        This must be differentiable through the tape
        """
        total_misfit = 0.0
        
        for i, t in enumerate(self.observations.times):
            for j, (x, y) in enumerate(self.observations.locations):
                # Get simulated value
                sim_value = probe_manager.get_value_at(t, x, y)
                obs_value = self.observations.water_levels[i, j]
                weight = self.observations.weights[i, j]
                
                # Squared misfit
                misfit = weight * (sim_value - obs_value)**2
                total_misfit += misfit
        
        # Normalize
        n_obs = len(self.observations.times) * len(self.observations.locations)
        return total_misfit / n_obs
    
    def optimize_with_adjoint(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6
    ) -> Dict[str, float]:
        """
        Optimize using Firedrake's adjoint capabilities
        """
        print("="*60)
        print("Adjoint-based Parameter Optimization")
        print("="*60)
        
        # Initial forward run
        print("\nRunning initial simulation...")
        continue_annotation()  # Start tape recording
        
        solver, probe_manager, domain = self.build_adjoint_simulation(self.controls)
        
        # Compute functional
        J_val = self.compute_functional(probe_manager)
        
        # Create ReducedFunctional
        # This wraps the functional and enables gradient computation
        J = ReducedFunctional(
            Constant(J_val),
            list(self.controls.values())
        )
        
        # Setup optimization bounds
        bounds_list = []
        for name in self.controls.keys():
            min_val, max_val = getattr(self.bounds, name)
            bounds_list.append((min_val, max_val))
        
        # Optimize using scipy's L-BFGS-B through pyadjoint
        print("\nStarting optimization with L-BFGS-B...")
        
        optimal_controls = minimize(
            J,
            method='L-BFGS-B',
            bounds=bounds_list,
            options={
                'maxiter': max_iterations,
                'ftol': tolerance,
                'disp': True
            }
        )
        
        # Extract optimized parameters
        optimized_params = {}
        for name, ctrl in self.controls.items():
            optimized_params[name] = float(ctrl.control)
        
        print("\n" + "="*60)
        print("Optimization Complete")
        print("="*60)
        
        return optimized_params
    
    def compute_hessian(self) -> np.ndarray:
        """
        Compute Hessian matrix for uncertainty quantification
        Hessian provides information about parameter sensitivity and correlations
        """
        print("\nComputing Hessian for uncertainty analysis...")
        
        # Run forward simulation
        continue_annotation()
        solver, probe_manager, domain = self.build_adjoint_simulation(self.controls)
        J_val = self.compute_functional(probe_manager)
        J = ReducedFunctional(Constant(J_val), list(self.controls.values()))
        
        # Compute Hessian
        # This uses second-order adjoint
        hessian = compute_hessian(J, list(self.controls.values()))
        
        return hessian


class MultiObjectiveOptimizer(AdjointParameterOptimizer):
    """
    Extended optimizer that handles multiple objectives
    e.g., fit water levels + match discharge + regularization
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = kwargs.get('objective_weights', {'water_level': 1.0})
        
    def compute_functional(self, probe_manager: 'ProbeManager', solver: 'RichardsSolver') -> float:
        """
        Multi-objective functional
        """
        J_total = 0.0
        
        # 1. Water level misfit
        if 'water_level' in self.weights:
            J_wl = super().compute_functional(probe_manager)
            J_total += self.weights['water_level'] * J_wl
        
        # 2. Discharge misfit (if you have discharge observations)
        if 'discharge' in self.weights and hasattr(self, 'discharge_obs'):
            J_q = self._compute_discharge_misfit(solver)
            J_total += self.weights['discharge'] * J_q
        
        # 3. Regularization (Tikhonov)
        if 'regularization' in self.weights:
            J_reg = self._compute_regularization()
            J_total += self.weights['regularization'] * J_reg
        
        # 4. Physical constraints penalty
        if 'physical_penalty' in self.weights:
            J_phys = self._compute_physical_penalty()
            J_total += self.weights['physical_penalty'] * J_phys
        
        return J_total
    
    def _compute_discharge_misfit(self, solver: 'RichardsSolver') -> float:
        """Compute misfit with discharge observations"""
        # Implement based on your discharge calculation
        pass
    
    def _compute_regularization(self) -> float:
        """Tikhonov regularization to prefer parameters close to prior"""
        J_reg = 0.0
        
        if hasattr(self, 'prior_params'):
            for name, ctrl in self.controls.items():
                prior_val = self.prior_params[name]
                current_val = float(ctrl.control)
                J_reg += (current_val - prior_val)**2
        
        return J_reg
    
    def _compute_physical_penalty(self) -> float:
        """Penalty for physically unrealistic parameter combinations"""
        J_penalty = 0.0
        
        # Example: theta_r should be less than theta_s
        theta_r = float(self.controls['theta_r'].control)
        theta_s = float(self.controls['theta_s'].control)
        
        if theta_r >= theta_s:
            J_penalty += 1000 * (theta_r - theta_s + 0.01)**2
        
        # Example: n should be > 1
        n = float(self.controls['n'].control)
        if n <= 1.0:
            J_penalty += 1000 * (1.01 - n)**2
        
        return J_penalty


def sensitivity_analysis(
    optimizer: AdjointParameterOptimizer,
    param_name: str,
    perturbation: float = 0.01
) -> Dict:
    """
    Perform sensitivity analysis for a specific parameter
    Shows how changes in one parameter affect the objective
    """
    print(f"\nSensitivity analysis for parameter: {param_name}")
    
    base_params = {name: float(ctrl.control) for name, ctrl in optimizer.controls.items()}
    
    # Perturb parameter
    perturbed_params = base_params.copy()
    perturbed_params[param_name] *= (1 + perturbation)
    
    # Run simulations
    continue_annotation()
    _, probe_base, _ = optimizer.build_adjoint_simulation(optimizer.controls)
    J_base = optimizer.compute_functional(probe_base)
    
    # Update control
    optimizer.controls[param_name].control.assign(perturbed_params[param_name])
    
    continue_annotation()
    _, probe_pert, _ = optimizer.build_adjoint_simulation(optimizer.controls)
    J_pert = optimizer.compute_functional(probe_pert)
    
    # Compute sensitivity
    sensitivity = (J_pert - J_base) / (perturbed_params[param_name] - base_params[param_name])
    
    # Reset
    optimizer.controls[param_name].control.assign(base_params[param_name])
    
    return {
        'parameter': param_name,
        'base_value': base_params[param_name],
        'perturbed_value': perturbed_params[param_name],
        'base_objective': J_base,
        'perturbed_objective': J_pert,
        'sensitivity': sensitivity,
        'relative_sensitivity': sensitivity * base_params[param_name] / J_base
    }


def uncertainty_quantification(
    optimizer: AdjointParameterOptimizer,
    observation_error: float = 0.01
) -> Dict:
    """
    Compute parameter uncertainties using Hessian
    Based on linear approximation around optimum
    """
    # Compute Hessian at optimal point
    H = optimizer.compute_hessian()
    
    # Convert to numpy array if needed
    if hasattr(H, 'array'):
        H = H.array()
    
    # Covariance matrix: C = (H / (2 * σ²))^{-1}
    # where σ² is the observation error variance
    try:
        C = np.linalg.inv(H / (2 * observation_error**2))
        
        # Parameter standard deviations
        param_std = np.sqrt(np.diag(C))
        
        # Correlation matrix
        D = np.diag(1.0 / param_std)
        correlation = D @ C @ D
        
        param_names = list(optimizer.controls.keys())
        
        results = {
            'covariance': C,
            'correlation': correlation,
            'std_dev': {name: std for name, std in zip(param_names, param_std)},
            'confidence_95': {name: 1.96 * std for name, std in zip(param_names, param_std)}
        }
        
        print("\nParameter Uncertainties (95% confidence):")
        for name, conf in results['confidence_95'].items():
            value = float(optimizer.controls[name].control)
            print(f"  {name:10s}: {value:.6f} ± {conf:.6f}")
        
        return results
        
    except np.linalg.LinAlgError:
        print("Warning: Hessian is singular - parameters may not be identifiable")
        return None


if __name__ == "__main__":
    """Example usage"""
    
    # This would be integrated with your existing simulation setup
    print("Advanced adjoint-based optimization example")
    print("Requires integration with your RichardsSolver class")