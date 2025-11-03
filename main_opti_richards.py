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
    - Extract soil parameters (and rain multipliers) from params dict
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
        start_datetime=generic_param['start_datetime'],
        end_datetime=generic_param['end_datetime'],
        dt_td=timedelta(hours=6)
    )
    
    rain_zones = [
        {'name': 'grass', 'x_min': 0.0, 'x_max': 8.0, 'multiplier':params['rain_mult_0']},
        {'name': 'green_infrastructure', 'x_min': 9.0, 'x_max': 11.0, 'multiplier': params['rain_mult_1']},
    ]
    
    rain_source = rainfall_scenario(
        from_date=config.start_datetime,
        to_date=config.end_datetime,
        # From CSV file (need to specify path and rain unit)
        #csv_path=config.data_input_dir / "BB_METEO.csv",
        #rain_unit="mm/day",
        # From Meteostat (uncomment to use)
        meteostat_station='SOK6B',
        meteostat_agg_hours=6,
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

    bc_manager = BoundaryConditionManager(V, params['wt_left'], params['wt_right'],
                                    left_trend=(generic_param['end_datetime'], 
                                                params['wt_left']-params['wt_trend']),
                                    right_trend=(generic_param['end_datetime'], 
                                                 params['wt_right']-params['wt_trend']),
                                    time_converter=config.time_converter)

    probe_manager = ProbeManager(domain.mesh,
                                 probe_positions = [[8.0, 1.0], [10.0, 1.0], [12.5, 1.0]]
    )

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

def main(generic_param):
    """Run parameter optimization"""

    LTC_data = load_measured_data(start_datetime=generic_param['start_datetime'],
                                  end_datetime=generic_param['end_datetime']+timedelta(hours=6),
                                  time_converter=TimeConverter(generic_param['start_datetime']),
                                  align=True,
                                  align_freq=generic_param['timestep_hours'])

    observations = ObservationData(
        times=LTC_data['times'],  # days to hours
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
        'rain_mult_0': 1.0,
        'rain_mult_1': 6.0,
        # Water table initial condition (if optimizing)
        'wt_left': 0.6,
        'wt_right': 1.5,
        'wt_trend': 0.25 # linear trend over simulation period
    }
        
    # Create tight bounds
    initial_params, bounds = create_tight_bounds(
        base_params,
        variation_pct=20.0,  # ±20%
    )
    """
    # Cost estimate
    print("\n3. Estimating computational cost...")
    estimate_computational_cost(
        n_params=len(base_params),
        n_iterations=40,
        simulation_time_sec=60.0,  # Adjust to your simulation time!
        method='both'
    )
    """
    # ==============================================================================
    # STEP 3: Run optimization
    # ==============================================================================
        
    optimizer = ParameterOptimizer(
        forward_model=your_simulation,
        observations=observations,
        bounds=bounds,
        initial_params=initial_params
    )
    
    # Optional: Check parameter correlations first
    print("\n5. Checking parameter correlations...")
    #correlation = compute_parameter_correlation(optimizer, n_samples=20, plot=True)
    
    # Run optimization
    print("\n6. Running optimization...")
    best_params = optimizer.optimize(
        n_iterations=5,
        learning_rate=0.3,  # Lower for tight bounds
        epsilon_relative=1e-2,
        adaptive_lr=True,
        gradient_clip=10.0,
        optimizer_type='sgd',
        use_finite_diff=True,
        verbose=False
    )
    
    # ==============================================================================
    # STEP 4: Analyze results
    # ==============================================================================
    
    print("\n7. Results:")
    print_parameter_comparison(
        optimized=best_params,
        initial=initial_params,
        #true=true_params  # Remove this for real data
    )
    
    save_path = f'optimized_parameters_{datetime.now().strftime("%m%d_%H%M")}.png'

    # Plot results
    plot_optimization_results(
        optimizer,
        #true_params=true_params,  # Remove this for real data
        save_path=save_path
    )
    print(f"   Results saved to: {save_path}")

    return best_params, optimizer


if __name__ == "__main__":
    # Run main example
    generic_param = {
        'start_datetime': datetime(2024, 4, 15),
        'end_datetime': datetime(2024, 6, 30),
        'timestep_hours': 6,
    }

    best_params, optimizer = main(generic_param)
