"""
Example: Optimize soil parameters using Firedrake adjoint

This demonstrates the complete workflow:
1. Define simulation with Constant parameters
2. Run simulation once with annotation
3. Setup adjoint optimization
4. Optimize and visualize results
"""
import numpy as np
from src import *
from firedrake import *
from firedrake.adjoint import *
from datetime import datetime, timedelta
from pyadjoint import get_working_tape, continue_annotation
from typing import Dict

def your_simulation(param_constants: Dict[str, Function], domain, V) -> ProbeManager:
    """
    Richards simulation with loss computed during solve (stays on tape!)
    
    Args:
        param_constants: Dictionary of scalar Functions (R space) for parameters
                        NOTE: Ks_till and Ks_terreau are stored in ln() space!
                              Use exp() to convert to physical values.
        domain: Domain object with mesh
        V: Function space for pressure field
    """
    config = SimulationConfig(
        name="Adjoint_Optimization",
        start_datetime=generic_param['start_datetime'],
        end_datetime=generic_param['end_datetime'],
        dt_td=timedelta(hours=generic_param['timestep_hours'])
    )
    
    rain_zones = [
        {'name': 'grass', 'x_min': 0.0, 'x_max': 8.0, 'multiplier': param_constants['rain_mult_0']},
        {'name': 'green_infrastructure', 'x_min': 9.0, 'x_max': 11.0, 'multiplier': param_constants['rain_mult_1']},
    ]
    
    rain_source = rainfall_scenario(
        from_date=config.start_datetime,
        to_date=config.end_datetime,
        meteostat_station='SOK6B',
        meteostat_agg_hours=generic_param['timestep_hours'],
        zones=rain_zones
    )

    domain.add_rectangle("GI", 9.0, 11.0, 4.0, 5.0)

    domain.assign("base", Material.till(
        theta_r=param_constants['theta_r_till'],
        theta_s=param_constants['theta_s_till'],
        alpha=param_constants['alpha_till'],
        n=param_constants['n_till'],
        Ks=param_constants['Ks_till']
    ))

    domain.assign("GI", Material.terreau(
        theta_r=param_constants['theta_r_terreau'],
        theta_s=param_constants['theta_s_terreau'],
        alpha=param_constants['alpha_terreau'],
        n=param_constants['n_terreau'],
        Ks=param_constants['Ks_terreau']
    ))

    field_map = MaterialField(domain, V)

    bc_manager = BoundaryConditionManager(V, 
                                          left_wt=param_constants['wt_left'], 
                                          right_wt=param_constants['wt_right'],
                                          #left_wt=0.8, 
                                          #right_wt=1.5,
                                    time_converter=config.time_converter)
    
    probe_manager = ProbeManager(domain.mesh)

    solver = RichardsSolver(
        V=V,
        field_map=field_map,
        source_scenario=rain_source,
        bc_manager=bc_manager,
        config=config,
        verbose=True
    )

    solver.run(probe_manager)

    return probe_manager

def main(generic_param):
    """
    Complete adjoint-based optimization workflow
    """
    print("="*70)
    print("ADJOINT-BASED PARAMETER OPTIMIZATION")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # STEP 1: Load observation data (3 probes)
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading observations...")
    
    LTC_data = load_measured_data(
        start_datetime=generic_param['start_datetime'],
        end_datetime=generic_param['end_datetime'], # + timedelta(hours=6)
        time_converter=TimeConverter(generic_param['start_datetime']),
        align=True,
        align_freq=generic_param['timestep_hours']
    )

    continue_annotation()

    observations = ObservationData(
        times=LTC_data['times'],
        locations=[[8.0, 1.0], [10.0, 1.0], [12.5, 1.0]],
        values=np.array([
            LTC_data['LTC 101'],  # Probe 1
            LTC_data['LTC 102'],  # Probe 2  
            LTC_data['LTC 103']   # Probe 3
        ]).T  # Shape: [n_times, 3]
    )
    
    print(f"   ✓ Loaded {len(observations.values)} time steps")
    print(f"   ✓ Time range: {observations.times[0]:.6f} to {observations.times[-1]:.6f} days")
    print(f"   ✓ Time range: {observations.times[0]*24:.2f} to {observations.times[-1]*24:.2f} hours")
    print(f"   ✓ Monitoring {len(observations.locations)} probe locations")
    
    # -------------------------------------------------------------------------
    # STEP 2: Define parameters to optimize (15 parameters!)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Setting up parameters...")
    
    initial_params = {
        # Till soil parameters (5)
        'theta_r_till': 0.02,
        'theta_s_till': 0.14,
        'alpha_till': 0.9399,
        'n_till': 2.3579,
        'Ks_till': 9e-6,  # Natural log for UFL exp()
        
        # Terreau soil parameters (5)
        'theta_r_terreau': 0.02,
        'theta_s_terreau': 0.43,
        'alpha_terreau': 1.1670,
        'n_terreau': 2.1052,
        'Ks_terreau': 4e-5,  # Natural log for UFL exp()
        
        # Rain multipliers (2)
        'rain_mult_0': 1.0,
        'rain_mult_1': 6.0,

        # Water table conditions (3)
        'wt_left': 0.8,
        'wt_right': 1.5,
    }
    
    realistic_bounds = {
    # Till (clayey soil) - tight bounds based on literature
    'theta_r_till': (0.01, 0.05),      # Residual moisture
    'theta_s_till': (0.10, 0.20),      # Saturated (clay is 0.1-0.2)
    'Ks_till':  (5e-7, 5e-5), 
    
    # Terreau (organic soil) - wider bounds
    'theta_r_terreau': (0.01, 0.10),   # Can retain more
    'theta_s_terreau': (0.35, 0.55),   # High porosity         
    'Ks_terreau': (5e-6, 5e-4),  # Natural log bounds
    
    # Rain multipliers - based on your system
    'rain_mult_0': (0.5, 2.0),         # ±100% adjustment
    'rain_mult_1': (2.0, 10.0),        # GI can have higher mult
    }

    print(f"   ✓ Total parameters to optimize: {len(initial_params)}")
    
    # -------------------------------------------------------------------------
    # STEP 3: Create domain and parameter controls
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Creating parameter controls...")
    domain = Domain(nx=60, ny=30, Lx=20.0, Ly=5.0)    
    controls_dict, param_functions = create_parameter_controls(initial_params, domain.mesh)
    print(f"   ✓ Created {len(controls_dict)} scalar Function controls (R space)")

    # -------------------------------------------------------------------------
    # STEP 4: Run forward simulation WITH ANNOTATION
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Running forward simulation with adjoint annotation...")
    print("   (This may take a few minutes...)")
    
    # Create function space for pressure
    V = FunctionSpace(domain.mesh, "CG", 1, name="pressure")

    # CRITICAL: Enable annotation and pass Functions (not dict of floats!)
    probe_manager = your_simulation(param_functions, domain, V)
    
    n_blocks = len(get_working_tape().get_blocks())
    print(f"   ✓ Adjoint tape recorded {n_blocks} operations")
    
    # -------------------------------------------------------------------------
    # STEP 5: Setup optimization
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Setting up optimization problem...")

    bounds = create_tight_bounds(initial_params, variation_pct=30.0,
                                 custom_bounds=realistic_bounds)
    
    optimizer = AdjointOptimizer(observations, bounds)
    optimizer.setup_optimization(probe_manager, controls_dict, param_functions, initial_params)
    
    # -------------------------------------------------------------------------
    # STEP 6: Run optimization!
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Running optimization...")
    print("   Method: L-BFGS-B (bound-constrained quasi-Newton)")
    print("   Note: First gradient computation may take 2-5 minutes")
    print("         (compiling adjoint solve)")
    print()
    
    optimized_params = optimizer.optimize(
        method='L-BFGS-B',
        maxiter=4,     
        gtol=1e-4,
        verbose=True
    )
    
    # -------------------------------------------------------------------------
    # STEP 7: Analyze results
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Analyzing results...")
    
    # Print parameter comparison
    print_parameter_comparison(
        optimized=optimized_params,
        initial=initial_params,
        bounds=bounds.bounds
    )
    
    # Plot optimization results
    save_path = f'optimization_results_{datetime.now().strftime("%m%d_%H%M")}.png'
    plot_optimization_results(
        optimizer=optimizer,
        forward_model_func=your_simulation,
        observations=observations,
        initial_params=initial_params,
        optimized_params=optimized_params,
        mesh=domain.mesh,
        domain=domain,
        V=V,
        save_path=save_path
    )
    
    # Residual analysis
    residual_path = f'residual_analysis_{datetime.now().strftime("%m%d_%H%M")}.png'
    plot_residual_analysis(
        optimizer=optimizer,
        forward_model_func=your_simulation,
        observations=observations,
        optimized_params=optimized_params,
        mesh=domain.mesh,
        domain=domain,
        V=V,
        save_path=residual_path
    )
    
    # -------------------------------------------------------------------------
    # STEP 8: Validation
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Validating optimized parameters...")
    validate_optimized_parameters(
        forward_model_func=your_simulation,
        observations=observations,
        optimized_params=optimized_params,
        mesh=domain.mesh,
        domain=domain,
        V=V,
        n_runs=1
    )
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Results saved to:")
    print(f"  - {save_path}")
    print(f"  - {residual_path}")
    
    return optimized_params, optimizer


if __name__ == "__main__":
    # Simulation period
    generic_param = {
        'start_datetime': datetime(2024, 4, 15),
        'end_datetime': datetime(2024, 5, 30),
        'timestep_hours': 6,
    }
    
    # Run optimization
    best_params, optimizer = main(generic_param)
    
    print("\n✓ All done! Check the generated plots for detailed results.")