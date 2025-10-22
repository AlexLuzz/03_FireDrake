"""
main_optimization.py

Main script to run parameter optimization:
1. Define data window for calibration
2. Load measured data
3. Run optimization
4. Validate results
"""

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from datetime import datetime, timedelta
from firedrake import FunctionSpace, RectangleMesh

# Import from local modules
from physics import SoilMaterial, Domain, VanGenuchtenParams, VanGenuchtenModel
from solver import RichardsSolver, BoundaryConditionManager
from setup import SimulationConfig, RainScenario, ProbeManager
from optimization import SimpleOptimizer, Parameter, L2_cost, RMSE_cost, NSE_cost
from optimization import plot_optimization_history, plot_fit_quality
from tools.data_import import load_and_align_data

# ============================================================
# FORWARD MODEL
# ============================================================

class ForwardModel:
    """
    Run Richards solver and extract simulated water table
    """
    
    def __init__(self, 
                 mesh,
                 base_material_name: str = "Till",
                 probe_location: tuple = (5.0, 2.5),
                 observation_times: np.ndarray = None,
                 coarse: bool = True):
        """
        Parameters:
        -----------
        mesh : Mesh
            FireDrake mesh
        base_material_name : str
            Material to optimize parameters for
        probe_location : tuple
            (x, y) location to extract water table
        observation_times : array
            Times to extract data at (seconds)
        coarse : bool
            Use coarse mesh/timestep for speed
        """
        self.mesh = mesh
        self.base_material_name = base_material_name
        self.probe_location = probe_location
        self.observation_times = observation_times
        self.coarse = coarse
        
        # Cache for avoiding redundant solves
        self.cache = {}
        self.n_evaluations = 0
    
    def __call__(self, params_dict: dict) -> np.ndarray:
        """
        Run forward model with given parameters
        
        Parameters:
        -----------
        params_dict : dict
            {'alpha': 2.0, 'n': 1.5, ...}
        
        Returns:
        --------
        simulated : array
            Water table depths at observation times
        """
        self.n_evaluations += 1
        
        # Check cache
        param_key = tuple(sorted(params_dict.items()))
        if param_key in self.cache:
            print(f"  [Eval {self.n_evaluations}] Using cached result")
            return self.cache[param_key]
        
        print(f"\n[Eval {self.n_evaluations}] Running forward model:")
        for name, value in params_dict.items():
            print(f"  {name} = {value:.6e}")
        
        # Create material with new parameters
        material = self._create_material(params_dict)
        
        # Setup domain
        domain = Domain.homogeneous(self.mesh, material)
        
        # Setup solver
        V = FunctionSpace(self.mesh, "CG", 1)
        
        # Simple config for calibration
        dt = 7200 if self.coarse else 3600  # 2 or 1 hour
        t_end = max(self.observation_times)
        
        config = SimulationConfig(
            dt=dt,
            t_end=t_end,
            output_dir=Path("results/calibration")
        )
        
        # No rain for simple calibration
        rain_event = RainScenario(events=[])
        
        bc_manager = BoundaryConditionManager(V, initial_water_table=1.2)
        
        solver = RichardsSolver(self.mesh, V, domain, rain_event, bc_manager, config)
        
        # Run and extract at observation times
        simulated = self._extract_water_table(solver, config)
        
        # Cache result
        self.cache[param_key] = simulated
        
        print(f"  → Water table range: [{simulated.min():.3f}, {simulated.max():.3f}] m")
        
        return simulated
    
    def _create_material(self, params_dict: dict):
        """Create material with optimized parameters"""
        # Create new parameters
        vg_params = VanGenuchtenParams(
            alpha=params_dict.get('alpha', 2.0),
            n=params_dict.get('n', 1.5),
            theta_s=params_dict.get('theta_s', 0.45),
            theta_r=params_dict.get('theta_r', 0.067),
            l_param=0.5
        )
        
        # Create hydraulic model
        hydraulic_model = VanGenuchtenModel(vg_params)
        
        # Create soil material
        K_s = params_dict.get('K_s', 1e-5)
        material = SoilMaterial(
            name=self.base_material_name,
            K_s=K_s,
            hydraulic_model=hydraulic_model
        )
        
        return material
    
    def _extract_water_table(self, solver, config) -> np.ndarray:
        """Extract water table depth at observation times"""
        simulated = []
        t = 0.0
        obs_idx = 0
        
        while t <= config.t_end and obs_idx < len(self.observation_times):
            target_time = self.observation_times[obs_idx]
            
            # Advance to target time
            while t < target_time:
                dt_step = min(config.dt, target_time - t)
                solver.solve_timestep(dt_step)
                t += dt_step
            
            # Extract water table at probe location
            h = solver.h_n.at(self.probe_location)
            water_table = 5.0 - h  # Surface elevation - pressure head
            simulated.append(water_table)
            
            obs_idx += 1
        
        return np.array(simulated)


# ============================================================
# MAIN OPTIMIZATION ROUTINE
# ============================================================

def run_calibration(calibration_window: tuple = None,
                   use_synthetic_data: bool = True,
                   method: str = 'L-BFGS-B'):
    """
    Main calibration routine
    
    Parameters:
    -----------
    calibration_window : tuple
        (start_datetime, end_datetime) for calibration
    use_synthetic_data : bool
        Use synthetic data if True, else load from CSV
    method : str
        Optimization method: 'L-BFGS-B' or 'differential_evolution'
    """
    
    print("="*70)
    print("PARAMETER CALIBRATION FOR RICHARDS EQUATION")
    print("="*70)
    
    # ========================================
    # 1. Define calibration window
    # ========================================
    if calibration_window is None:
        start_datetime = datetime(2024, 8, 16)
        end_datetime = datetime(2024, 8, 21)  # 5 days
    else:
        start_datetime, end_datetime = calibration_window
    
    print(f"\nCalibration window:")
    print(f"  Start: {start_datetime}")
    print(f"  End:   {end_datetime}")
    print(f"  Duration: {(end_datetime - start_datetime).total_seconds() / 86400:.1f} days")
    
    # ========================================
    # 2. Load observation data
    # ========================================
    print("\nLoading measured data...")
    data_file = Path("data_input/MEASURED_PZ_CG.csv")
    
    # Calculate simulation duration in days
    sim_duration_days = (end_datetime - start_datetime).total_seconds() / 86400.0
    
    # Load and align measured data
    measured_data = load_and_align_data(
        csv_path=data_file,
        start_from=0.0,
        sim_duration_days=sim_duration_days,
        data_type='Measured',
        ref_date=start_datetime,
        offset=0.6,  # 60cm offset if needed
        apply_smoothing=True
    )
    
    if measured_data is None:
        print("⚠️  Failed to load measured data!")
        return None
    
    # Extract data for LTC 101 (or first available)
    available_ltcs = [k for k in measured_data.keys() if k.startswith('LTC')]
    if not available_ltcs:
        print("⚠️  No LTC data found in measured data!")
        return None
    
    ltc_name = available_ltcs[0]
    obs_times = measured_data['times'] * 86400.0  # Convert days to seconds
    obs_values = measured_data[ltc_name]
    
    print(f"Observations: {len(obs_times)} points from {ltc_name}")
    print(f"  Time range: {obs_times[0]/3600:.1f} to {obs_times[-1]/3600:.1f} hours")
    print(f"  Value range: {obs_values.min():.3f} to {obs_values.max():.3f} m")
    
    # ========================================
    # 3. Define parameters to optimize
    # ========================================
    print("\nDefining parameters...")
    
    parameters = [
        Parameter(
            name='alpha',
            value=2.0,       # Initial guess
            lower=0.5,       # Lower bound
            upper=10.0,      # Upper bound
            log_scale=True   # Optimize in log space
        ),
        Parameter(
            name='n',
            value=1.5,
            lower=1.1,
            upper=3.0,
            log_scale=False
        ),
        # Uncomment to also optimize K_s:
        # Parameter(
        #     name='K_s',
        #     value=1e-5,
        #     lower=1e-7,
        #     upper=1e-3,
        #     log_scale=True
        # ),
    ]
    
    for p in parameters:
        print(f"  {p.name}: [{p.lower:.2e}, {p.upper:.2e}], initial = {p.value:.2e}")
    
    # ========================================
    # 4. Create mesh and forward model
    # ========================================
    print("\nCreating mesh and forward model...")
    
    # Use coarse mesh for calibration
    mesh = RectangleMesh(40, 20, 20.0, 5.0)
    
    forward_model = ForwardModel(
        mesh=mesh,
        base_material_name="Till",
        probe_location=(5.0, 2.5),
        observation_times=obs_times,
        coarse=True
    )
    
    # ========================================
    # 5. Run sensitivity analysis (optional)
    # ========================================
    print("\nSkipping sensitivity analysis (optional feature)...")
    # Uncomment below if you have JaxSensitivity implemented:
    # from optimization.sensitivity_analysis import JaxSensitivity
    # jax_sens = JaxSensitivity()
    # h_test = np.linspace(-5, 0, 50)
    # test_params = VanGenuchtenParams(alpha=2.0, n=1.5)
    # sens = jax_sens.compute_sensitivities(h_test, test_params)
    # print("  Sensitivity summary:")
    # for name, values in sens.items():
    #     print(f"    ∂θ/∂{name}: max = {np.max(np.abs(values)):.4e}")
    
    # ========================================
    # 6. Create optimizer
    # ========================================
    print("\nSetting up optimizer...")
    
    optimizer = SimpleOptimizer(
        parameters=parameters,
        forward_model=forward_model,
        cost_function=L2_cost  # or L1_cost, RMSE_cost, NSE_cost
    )
    
    # ========================================
    # 7. Run optimization
    # ========================================
    print("\nStarting optimization...")
    
    results = optimizer.optimize(
        observed=obs_values,
        method=method,
        maxiter=50
    )
    
    # ========================================
    # 8. Save and visualize results
    # ========================================
    output_dir = Path("results/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    optimizer.save_results(
        results,
        output_dir / f'calibration_results_{timestamp}.json'
    )
    
    # Plot convergence
    plot_optimization_history(
        results['history'],
        output_dir / f'convergence_{timestamp}.png'
    )
    
    # Run final simulation with optimal parameters
    print("\nRunning final simulation with optimal parameters...")
    final_simulated = forward_model(results['optimal_parameters'])
    
    # Plot fit quality
    plot_fit_quality(
        final_simulated,
        obs_values,
        times=obs_times / 3600,  # Convert to hours
        filename=output_dir / f'fit_quality_{timestamp}.png'
    )
    
    # ========================================
    # 9. Print summary
    # ========================================
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)
    print(f"Final cost (L2): {results['optimal_cost']:.6e}")
    print(f"RMSE: {RMSE_cost(final_simulated, obs_values):.4f} m")
    
    # Compute NSE
    nse_cost = NSE_cost(final_simulated, obs_values)
    nse = 1.0 - nse_cost
    print(f"NSE: {nse:.4f}")
    
    print(f"\nForward model evaluations: {forward_model.n_evaluations}")
    print(f"Success: {results['success']}")
    
    print("\nOptimal parameters:")
    for name, value in results['optimal_parameters'].items():
        print(f"  {name:10s} = {value:.6e}")
    
    print("\n" + "="*70)
    
    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    
    # Option 1: Quick test with synthetic data
    results = run_calibration(
        calibration_window=None,
        use_synthetic_data=True,
        method='L-BFGS-B'
    )
    
    # Option 2: Real calibration with measured data
    # results = run_calibration(
    #     calibration_window=(
    #         datetime(2024, 8, 16),
    #         datetime(2024, 8, 21)
    #     ),
    #     use_synthetic_data=False,
    #     method='differential_evolution'  # Global search
    # )