"""
Practical example: Parameter optimization integrated with your existing code
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from firedrake import *
from physics import *
from solver import *
from jax_sensitivity_analyzer import *


# ============================================================
# 1. LOAD MEASURED DATA
# ============================================================

def load_observation_data(csv_file: str) -> List[ObservationData]:
    """
    Load observation data from CSV file
    
    Expected CSV format:
    time_hours, location_x, location_y, water_table_depth
    0.0, 5.0, 2.5, 1.2
    12.0, 5.0, 2.5, 1.5
    ...
    """
    df = pd.read_csv(csv_file)
    
    # Get unique locations
    locations = df[['location_x', 'location_y']].drop_duplicates().values
    locations = [tuple(loc) for loc in locations]
    
    # Get unique times
    times_hours = df['time_hours'].unique()
    times = times_hours * 3600  # Convert to seconds
    
    # Reshape data to (n_times, n_locations)
    values = np.zeros((len(times), len(locations)))
    
    for i, t in enumerate(times_hours):
        for j, loc in enumerate(locations):
            mask = (df['time_hours'] == t) & \
                   (df['location_x'] == loc[0]) & \
                   (df['location_y'] == loc[1])
            values[i, j] = df.loc[mask, 'water_table_depth'].values[0]
    
    return [ObservationData(
        times=times,
        values=values,
        locations=locations,
        variable='water_table',
        weights=None  # Equal weights
    )]


def create_synthetic_observations() -> List[ObservationData]:
    """
    Create synthetic observations for testing
    (Use this if you don't have real data yet)
    """
    # Observation times (every 12 hours for 5 days)
    times = np.arange(0, 5*24+1, 12) * 3600  # seconds
    
    # Observation locations
    locations = [
        (5.0, 2.5),   # Left side
        (10.0, 2.5),  # Center
        (15.0, 2.5),  # Right side
    ]
    
    # Synthetic water table evolution (meters below surface)
    # Starting at 1.2m, gradually rising
    n_times = len(times)
    n_locs = len(locations)
    values = np.zeros((n_times, n_locs))
    
    for i in range(n_times):
        t_hours = times[i] / 3600
        for j, (x, y) in enumerate(locations):
            # Synthetic pattern: water table rises faster on left
            base_rise = 0.01 * t_hours  # 1 cm per hour
            spatial_factor = 1.0 + 0.2 * (20 - x) / 20  # Varies with x
            values[i, j] = 1.2 + base_rise * spatial_factor
    
    return [ObservationData(
        times=times,
        values=values,
        locations=locations,
        variable='water_table'
    )]


# ============================================================
# 2. CUSTOM FORWARD MODEL (Integrated with your solver)
# ============================================================

class CustomForwardModel(ForwardModelRunner):
    """
    Custom forward model that matches your existing setup
    """
    
    def __init__(self, observations: List[ObservationData], 
                 rain_scenario: RainScenario,
                 coarse: bool = True):
        """
        Parameters:
        -----------
        observations : list
            Observation data
        rain_scenario : RainScenario
            Rain events to simulate
        coarse : bool
            Use coarse mesh/timestep
        """
        self.observations = observations
        self.rain_scenario = rain_scenario
        self.coarse = coarse
        self.n_evaluations = 0
        self.cache = {}
    
    def run(self, parameters: Dict[str, float], verbose: bool = False) -> Dict[str, np.ndarray]:
        """Run forward model with parameters"""
        self.n_evaluations += 1
        
        # Check cache
        param_key = tuple(sorted(parameters.items()))
        if param_key in self.cache:
            return self.cache[param_key]
        
        if verbose:
            print(f"\n[Eval {self.n_evaluations}] Forward model with:")
            for k, v in parameters.items():
                print(f"  {k} = {v:.4e}")
        
        # Create mesh
        if self.coarse:
            mesh = RectangleMesh(40, 20, 20.0, 5.0)  # Coarser
            dt = 7200  # 2 hours
        else:
            mesh = RectangleMesh(80, 40, 20.0, 5.0)
            dt = 3600  # 1 hour
        
        # Extract observation times
        all_times = set()
        for obs in self.observations:
            all_times.update(obs.times)
        obs_times_sorted = sorted(all_times)
        
        # Determine simulation end time
        t_end = max(obs_times_sorted)
        
        config = SimulationConfig(
            dt=dt,
            t_end=t_end,
            output_dir=Path("results/optimization")
        )
        
        # Setup materials with optimized parameters
        till_params = {
            'alpha': parameters.get('alpha_Till', 2.0),
            'n': parameters.get('n_Till', 1.5),
            'K_s': parameters.get('K_s_Till', 1e-5),
        }
        
        terreau_params = {
            'alpha': parameters.get('alpha_Terreau', 5.0),
            'n': parameters.get('n_Terreau', 1.8),
            'K_s': parameters.get('K_s_Terreau', 5e-5),
        }
        
        # Create materials (you may need to add a method to SoilMaterial)
        till = self._create_material("Till", till_params)
        terreau = self._create_material("Terreau", terreau_params)
        
        # Create domain (matching your setup)
        domain = Domain.homogeneous(mesh, till)
        domain.add_rectangle(
            material=terreau,
            x_min=9.0, x_max=11.0,
            y_min=4.0, y_max=5.0,
            name="green_infrastructure"
        )
        
        # Setup solver
        V = FunctionSpace(mesh, "CG", 1)
        bc_manager = BoundaryConditionManager(V, initial_water_table=1.2)
        
        solver = RichardsSolver(mesh, V, domain, self.rain_scenario, 
                               bc_manager, config)
        
        # Run simulation and extract at observation points
        results = self._extract_observations(solver, config, obs_times_sorted)
        
        # Cache
        self.cache[param_key] = results
        
        return results
    
    def _create_material(self, name: str, params: Dict) -> SoilMaterial:
        """Create soil material with custom parameters"""
        # Start with default
        material = SoilMaterial.from_curves(name=name)
        
        # Override with optimization parameters
        # You'll need to add setters to SoilMaterial or recreate it
        material.alpha = params.get('alpha', material.alpha)
        material.n = params.get('n', material.n)
        material.K_s = params.get('K_s', material.K_s)
        
        return material
    
    def _extract_observations(self, solver: RichardsSolver, 
                            config: SimulationConfig,
                            obs_times: List[float]) -> Dict[str, np.ndarray]:
        """Extract simulated values at observation points/times"""
        
        # Create interpolators
        interpolators = {}
        for obs in self.observations:
            interpolators[obs.variable] = []
            for loc in obs.locations:
                # Simple point evaluation
                interpolators[obs.variable].append(loc)
        
        # Storage
        results = {obs.variable: [] for obs in self.observations}
        
        # Time loop
        t = 0.0
        obs_idx = 0
        
        while t <= config.t_end and obs_idx < len(obs_times):
            target_time = obs_times[obs_idx]
            
            # Advance to target time
            while t < target_time:
                dt_step = min(config.dt, target_time - t)
                solver.solve_timestep(dt_step)
                t += dt_step
            
            # Extract values
            for obs in self.observations:
                values_at_t = []
                
                for loc in obs.locations:
                    if obs.variable == 'water_table':
                        # Water table = surface elevation - pressure head
                        h = solver.h_n.at(loc)
                        water_table = 5.0 - h  # Assuming surface at y=5.0
                        values_at_t.append(water_table)
                    
                    elif obs.variable == 'pressure_head':
                        h = solver.h_n.at(loc)
                        values_at_t.append(h)
                    
                    elif obs.variable == 'saturation':
                        # Get saturation field
                        s = solver.get_saturation()
                        sat_value = s.at(loc)
                        values_at_t.append(sat_value)
                
                results[obs.variable].append(values_at_t)
            
            obs_idx += 1
        
        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results


# ============================================================
# 3. MAIN OPTIMIZATION SCRIPT
# ============================================================

def run_parameter_optimization(use_synthetic_data: bool = True):
    """
    Main optimization routine
    
    Parameters:
    -----------
    use_synthetic_data : bool
        If True, use synthetic observations
        If False, load from CSV file
    """
    
    print("="*70)
    print("PARAMETER OPTIMIZATION FOR RICHARDS EQUATION")
    print("="*70)
    
    # ----------------------------------------
    # Step 1: Load or create observations
    # ----------------------------------------
    if use_synthetic_data:
        print("\nUsing synthetic observation data...")
        observations = create_synthetic_observations()
    else:
        print("\nLoading observation data from CSV...")
        observations = load_observation_data("measured_data.csv")
    
    print(f"Loaded {len(observations)} observation dataset(s)")
    for obs in observations:
        print(f"  - {obs.variable}: {len(obs.times)} times, {len(obs.locations)} locations")
    
    # ----------------------------------------
    # Step 2: Define parameters to optimize
    # ----------------------------------------
    print("\nDefining parameters to optimize...")
    
    # Option A: Optimize Till only
    parameters = VanGenuchtenParams.create_optimization_params("Till")
    
    # Option B: Optimize both materials
    # parameters = (
    #     VanGenuchtenParams.create_optimization_params("Till") +
    #     VanGenuchtenParams.create_optimization_params("Terreau")
    # )
    
    print(f"Optimizing {len(parameters)} parameters:")
    for param in parameters:
        print(f"  - {param.name}: [{param.lower_bound:.2e}, {param.upper_bound:.2e}]")
    
    # ----------------------------------------
    # Step 3: Define rain scenario (simplified)
    # ----------------------------------------
    # Use a simplified scenario for calibration
    rain_zones = [
        RainZone(x_min=0.0, x_max=8.0, multiplier=1.0, name="grass"),
        RainZone(x_min=9.0, x_max=11.0, multiplier=6.0, name="green_infrastructure"),
    ]
    
    rain_event = RainScenario.single_event(
        start_hours=5.0,
        end_hours=10.0,
        intensity_mm_hr=20.0,
        zones=rain_zones
    )
    
    # ----------------------------------------
    # Step 4: Create forward model
    # ----------------------------------------
    print("\nSetting up forward model...")
    forward_model = CustomForwardModel(
        observations=observations,
        rain_scenario=rain_event,
        coarse=True  # Use coarse mesh for speed
    )
    
    # ----------------------------------------
    # Step 5: Define cost function
    # ----------------------------------------
    cost_function = L2Cost(observations)
    # Alternatives:
    # cost_function = RMSECost(observations)
    # cost_function = NashSutcliffeCost(observations)
    
    # ----------------------------------------
    # Step 6: Create optimization problem
    # ----------------------------------------
    problem = OptimizationProblem(
        parameters=parameters,
        observations=observations,
        forward_model=forward_model,
        cost_function=cost_function
    )
    
    # ----------------------------------------
    # Step 7: Choose optimizer and run
    # ----------------------------------------
    print("\n" + "="*70)
    print("STARTING OPTIMIZATION")
    print("="*70)
    
    # Option A: Differential Evolution (global, no gradients needed)
    optimizer = DifferentialEvolutionOptimizer(problem)
    results = optimizer.optimize(maxiter=20, popsize=10)
    
    # Option B: L-BFGS-B (local, faster if good initial guess)
    # optimizer = ScipyOptimizer(problem, method='L-BFGS-B')
    # results = optimizer.optimize(maxiter=50)
    
    # ----------------------------------------
    # Step 8: Display results
    # ----------------------------------------
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    print(f"\nFinal cost: {results['optimal_cost']:.6e}")
    print(f"Success: {results['success']}")
    print(f"Iterations: {results['n_iterations']}")
    
    print("\nOptimal parameters:")
    for name, value in results['optimal_parameters'].items():
        print(f"  {name:20s} = {value:12.6e}")
    
    # ----------------------------------------
    # Step 9: Visualize results
    # ----------------------------------------
    output_dir = Path("results/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot convergence
    OptimizationVisualizer.plot_convergence(
        results['history'],
        filename=output_dir / f'convergence_{timestamp}.png'
    )
    
    # Run final simulation with optimal parameters
    print("\nRunning final simulation with optimal parameters...")
    final_simulated = forward_model.run(
        results['optimal_parameters'],
        verbose=True
    )
    
    # Plot fit quality
    OptimizationVisualizer.plot_fit_quality(
        observations,
        final_simulated,
        filename=output_dir / f'fit_quality_{timestamp}.png'
    )
    
    # ----------------------------------------
    # Step 10: Save results
    # ----------------------------------------
    import json
    
    output_data = {
        'optimal_parameters': results['optimal_parameters'],
        'optimal_cost': float(results['optimal_cost']),
        'n_iterations': results['n_iterations'],
        'n_forward_evaluations': forward_model.n_evaluations,
        'success': results['success'],
        'timestamp': timestamp,
        'parameter_bounds': {
            param.name: {
                'lower': param.lower_bound,
                'upper': param.upper_bound,
                'optimal': results['optimal_parameters'][param.name]
            }
            for param in parameters
        }
    }
    
    output_file = output_dir / f'optimization_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # ----------------------------------------
    # Step 11: Export optimal parameters for use
    # ----------------------------------------
    # Create a Python file with optimal parameters
    params_code = "# Optimized parameters\n"
    params_code += f"# Generated: {timestamp}\n"
    params_code += f"# Cost: {results['optimal_cost']:.6e}\n\n"
    
    params_code += "OPTIMIZED_PARAMS = {\n"
    for name, value in results['optimal_parameters'].items():
        params_code += f"    '{name}': {value:.6e},\n"
    params_code += "}\n"
    
    params_file = output_dir / f'optimized_params_{timestamp}.py'
    with open(params_file, 'w') as f:
        f.write(params_code)
    
    print(f"Parameters exported to: {params_file}")
    
    print("\n" + "="*70)
    print("OPTIMIZATION FINISHED")
    print("="*70)
    
    return results


# ============================================================
# 4. QUICK TEST FUNCTIONS
# ============================================================

def quick_test_forward_model():
    """Test forward model with default parameters"""
    print("Testing forward model...")
    
    observations = create_synthetic_observations()
    
    rain_zones = [RainZone(x_min=0.0, x_max=20.0, multiplier=1.0)]
    rain_event = RainScenario.single_event(
        start_hours=5.0, end_hours=10.0, 
        intensity_mm_hr=20.0, zones=rain_zones
    )
    
    forward_model = CustomForwardModel(
        observations=observations,
        rain_scenario=rain_event,
        coarse=True
    )
    
    # Test with default parameters
    test_params = {
        'alpha_Till': 2.0,
        'n_Till': 1.5,
        'K_s_Till': 1e-5,
    }
    
    results = forward_model.run(test_params, verbose=True)
    
    print("\nResults shape:")
    for var, data in results.items():
        print(f"  {var}: {data.shape}")
    
    print("\nForward model test successful!")


def compare_optimized_vs_default():
    """Compare optimized parameters against default"""
    # Load optimized results
    import json
    from pathlib import Path
    
    result_files = sorted(Path("results/optimization").glob("optimization_results_*.json"))
    if not result_files:
        print("No optimization results found!")
        return
    
    latest_result = result_files[-1]
    print(f"Loading: {latest_result}")
    
    with open(latest_result) as f:
        opt_results = json.load(f)
    
    # Default parameters
    default_params = {
        'alpha_Till': 2.0,
        'n_Till': 1.5,
        'K_s_Till': 1e-5,
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("PARAMETER COMPARISON")
    print("="*70)
    print(f"{'Parameter':<20} {'Default':>15} {'Optimized':>15} {'Change %':>15}")
    print("-"*70)
    
    for param_name in default_params:
        default_val = default_params[param_name]
        opt_val = opt_results['optimal_parameters'][param_name]
        change_pct = ((opt_val - default_val) / default_val) * 100
        
        print(f"{param_name:<20} {default_val:>15.6e} {opt_val:>15.6e} {change_pct:>14.1f}%")
    
    print("-"*70)
    print(f"Final cost: {opt_results['optimal_cost']:.6e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test_forward_model()
        elif sys.argv[1] == "compare":
            compare_optimized_vs_default()
        else:
            print("Usage: python optimization_example_usage.py [test|compare]")
    else:
        # Run full optimization
        results = run_parameter_optimization(use_synthetic_data=True)