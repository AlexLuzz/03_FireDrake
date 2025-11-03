"""
Example: Optimize soil parameters + rain multipliers using the framework

This shows how to:
1. Define your forward model (simulation wrapper)
2. Set up parameters and bounds
3. Run optimization
4. Analyze results
"""
import numpy as np
from src import *
from firedrake import FunctionSpace
from datetime import datetime, timedelta

# ==============================================================================
# STEP 1: Define your forward model
# ==============================================================================

def your_simulation(params: dict) -> np.ndarray:
    """
    Wrapper for your Richards equation simulation
    
    This is where you integrate with your existing code:
    - Extract soil parameters and rain multipliers from params dict
    - Run your RichardsSolver with these parameters
    - Return simulated values at observation points and times
    
    Args:
        params: Dictionary like {
            'theta_r': 0.065, 'theta_s': 0.41, 'alpha': 0.075, 'n': 1.89, 'Ks': 106,
            'rain_mult_0': 1.0, 'rain_mult_1': 6.0
        }
    
    Returns:
        Simulated values [n_times, n_locations]
    """
    config = SimulationConfig(
        name="Datetime_Duration",
        start_datetime=datetime(2024, 4, 15),
        end_datetime=datetime(2024, 4, 30),
        dt_td=timedelta(hours=6)
    )
    
    rain_zones = [
        {'name': 'grass', 'x_min': 0.0, 'x_max': 8.0, 'multiplier': 1.0},
        {'name': 'green_infrastructure', 'x_min': 9.0, 'x_max': 11.0, 'multiplier': 6.0},
    ]
    
    rain_source = rainfall_scenario(
        from_date=config.start_datetime,
        to_date=config.end_datetime,
        # From CSV file (need to specify path and rain unit)
        csv_path=config.data_input_dir / "BB_METEO.csv",
        rain_unit="mm/day",
        # From Meteostat (uncomment to use)
        #meteostat_station='SOK6B',
        #meteostat_agg_hours=6,
        zones=rain_zones
    )

    domain = Domain(nx=80, ny=40, Lx=20.0, Ly=5.0)
    domain.assign("base", Material.till(
        theta_r=params['theta_r'],
        theta_s=params['theta_s'],
        alpha=params['alpha'],
        n=params['n'],
        Ks=params['Ks']
    )
    )

    #domain.add_rectangle("GI", 9.0, 11.0, 4.0, 5.0)
    #domain.assign("GI", Material.terreau_curve_RAF())


    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)

    bc_manager = BoundaryConditionManager(
        V,
        left_wt=1.2,
        right_wt=1.2
    )

    probe_manager = ProbeManager(domain.mesh)

    solver = RichardsSolver(
        domain=domain,
        V=V,
        field_map=field_map,  # NEW: field_map instead of domain
        source_scenario=rain_source,
        bc_manager=bc_manager,
        config=config
    )

    solver.run(probe_manager)
    
    sim_data = probe_manager.get_data()
    times = sim_data['times']
    simulated = np.zeros((len(times), 3))  # (n_times, 3 probes)
    
    for i, t in enumerate(times):
        simulated[i, 0] = sim_data['data']['Probe_1']['water_table'][i]
        simulated[i, 1] = sim_data['data']['Probe_2']['water_table'][i]
        simulated[i, 2] = sim_data['data']['Probe_3']['water_table'][i]

    return simulated


# ==============================================================================
# STEP 2: Set up observations, parameters, and bounds
# ==============================================================================

def main():
    """Run parameter optimization"""
    
    print("="*70)
    print("PARAMETER OPTIMIZATION EXAMPLE")
    print("="*70)

    LTC_data = load_measured_data(start_datetime=datetime(2024, 4, 15),
                                  end_datetime=datetime(2024, 4, 30),
                                  time_converter=TimeConverter(datetime(2024, 4, 15)), align=True)

    observations = ObservationData(
        times=None,
        locations=[[8.0, 1.0], [10.0, 1.0], [12.5, 1.0]],
        values=np.array([LTC_data['LTC 101'], LTC_data['LTC 102'], LTC_data['LTC 103']]).T
    )
    
    print(f"  Loaded {len(observations.values)} observations at {len(observations.locations)} locations")
    
    # Your initial parameter estimates
    base_params = {
        # Soil parameters
        'theta_r': 0.02,   # Your best guess
        'theta_s': 0.14,
        'alpha': 0.9399,
        'n': 2.3579,
        'Ks': 9e-6,
        # Rain multipliers (if optimizing these)
        #'rain_mult_0': 1.2,
        #'rain_mult_1': 5.5,
    }
    
    true_params = base_params.copy()  # For testing with synthetic data

    print("\n2. Creating parameter bounds (±20%)...")
    
    # Create tight bounds
    initial_params, bounds = create_tight_bounds(
        base_params,
        variation_pct=20.0,  # ±20%
        min_constraints={'n': 1.05}  # n must be > 1
    )
    
    # Cost estimate
    print("\n3. Estimating computational cost...")
    estimate_computational_cost(
        n_params=len(base_params),
        n_iterations=40,
        simulation_time_sec=60.0,  # Adjust to your simulation time!
        method='both'
    )
    
    # ==============================================================================
    # STEP 3: Run optimization
    # ==============================================================================
    
    print("\n4. Creating optimizer...")
    
    optimizer = ParameterOptimizer(
        forward_model=your_simulation,
        observations=observations,
        bounds=bounds,
        initial_params=initial_params
    )
    
    # Optional: Check parameter correlations first
    print("\n5. Checking parameter correlations...")
    correlation = compute_parameter_correlation(optimizer, n_samples=20, plot=True)
    
    # Run optimization
    print("\n6. Running optimization...")
    best_params = optimizer.optimize(
        n_iterations=20,
        learning_rate=0.03,  # Lower for tight bounds
        optimizer_type='adam',
        use_finite_diff=True,
        verbose=True
    )
    
    # ==============================================================================
    # STEP 4: Analyze results
    # ==============================================================================
    
    print("\n7. Results:")
    print_parameter_comparison(
        optimized=best_params,
        initial=initial_params,
        true=true_params  # Remove this for real data
    )
    
    # Plot results
    print("\n8. Generating plots...")
    plot_optimization_results(
        optimizer,
        true_params=true_params,  # Remove this for real data
        save_path='optimization_results.png'
    )
    
    print("\n✅ Optimization complete!")
    print("   Results saved to: optimization_results.png")
    
    return best_params, optimizer


if __name__ == "__main__":
    # Run main example
    best_params, optimizer = main()
    