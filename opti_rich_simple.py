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
from pyadjoint import get_working_tape, continue_annotation, AdjFloat
from typing import Dict

def your_simulation(param_constants: Dict[str, Constant]) -> ProbeManager:
    """
    Richards simulation with loss computed during solve (stays on tape!)
    """
    config = SimulationConfig(
        name="Adjoint_Optimization",
        start_datetime=generic_param['start_datetime'],
        end_datetime=generic_param['end_datetime'],
        dt_td=timedelta(hours=6)
    )
    
    rain_zones = [
        {'name': 'grass', 'x_min': 0.0, 'x_max': 8.0, 'multiplier': 1.0},
        {'name': 'green_infrastructure', 'x_min': 9.0, 'x_max': 11.0, 'multiplier': 6.0},
    ]
    
    rain_source = rainfall_scenario(
        from_date=config.start_datetime,
        to_date=config.end_datetime,
        meteostat_station='SOK6B',
        meteostat_agg_hours=6,
        zones=rain_zones
    )

    domain = Domain(nx=60, ny=30, Lx=20.0, Ly=5.0)

    domain.assign("base", Material.till(
        theta_r=0.02,
        theta_s=0.14,
        alpha=0.94,
        n=2.3579,
        Ks=param_constants['Ks_till']
    ))
    field_map = MaterialField(domain)

    bc_manager = BoundaryConditionManager(
        field_map.V, left_wt=1.2, right_wt=1.5
    )
    
    probe_manager = ProbeManager(domain.mesh)

    solver = RichardsSolver(
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
    print(f"   ✓ Monitoring {len(observations.locations)} probe locations")
    
    # -------------------------------------------------------------------------
    # STEP 2: Define parameters to optimize (15 parameters!)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Setting up parameters...")
    
    initial_params = {
        # Till soil parameters (5)
        #'theta_r_till': 0.02,
        #'theta_s_till': 0.14,
        #'alpha_till': 0.9399,
        #'n_till': 2.3579,
        'Ks_till': 9e-6,
    }
    
    print(f"   ✓ Total parameters to optimize: {len(initial_params)}")
    
    # -------------------------------------------------------------------------
    # STEP 3: Create parameter controls (Firedrake Constants)
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Creating parameter controls...")
    controls_dict, param_constants = create_parameter_controls(initial_params)
    print(f"   ✓ Created {len(controls_dict)} Firedrake Controls")
    
    # -------------------------------------------------------------------------
    # STEP 4: Run forward simulation WITH ANNOTATION
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Running forward simulation with adjoint annotation...")
    print("   (This may take a few minutes...)")
    
    # CRITICAL: Enable annotation to record operations on the tape!
    get_working_tape().progress_bar = ProgressBar()
    probe_manager = your_simulation(param_constants)  # Returns ProbeManager!
    
    n_blocks = len(get_working_tape().get_blocks())
    print(f"   ✓ Adjoint tape recorded {n_blocks} operations")
    
    # -------------------------------------------------------------------------
    # STEP 5: Setup optimization
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Setting up optimization problem...")
    
    bounds = create_tight_bounds(initial_params, variation_pct=20.0)
    optimizer = AdjointOptimizer(observations, bounds)
    optimizer.setup_optimization(probe_manager, controls_dict, initial_params)
    
    # -------------------------------------------------------------------------
    # STEP 6: Run optimization!
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Running optimization...")
    print("   Method: L-BFGS-B (bound-constrained quasi-Newton)")
    print("   Note: First gradient computation may take 2-5 minutes")
    print("         (compiling adjoint solve)")
    print()
    
    optimized_params = optimizer.optimize(
        #method='L-BFGS-B',
        method='SLSQP',
        maxiter=3,      # Start with 10, increase to 50 for production
        gtol=1e-3,
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
        save_path=save_path
    )
    
    # Residual analysis
    residual_path = f'residual_analysis_{datetime.now().strftime("%m%d_%H%M")}.png'
    plot_residual_analysis(
        optimizer=optimizer,
        forward_model_func=your_simulation,
        observations=observations,
        optimized_params=optimized_params,
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
        n_runs=3
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
        'end_datetime': datetime(2024, 5, 15),
        'timestep_hours': 6,
    }
    
    # Run optimization
    best_params, optimizer = main(generic_param)
    
    print("\n✓ All done! Check the generated plots for detailed results.")