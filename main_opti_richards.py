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
from firedrake import FunctionSpace, Constant
from datetime import datetime, timedelta
from pyadjoint import get_working_tape, continue_annotation, AdjFloat
from typing import Dict

def your_simulation(param_constants: Dict[str, Constant], observations: ObservationData) -> AdjFloat:
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
        {'name': 'grass', 'x_min': 0.0, 'x_max': 8.0, 'multiplier': param_constants['rain_mult_0']},
        {'name': 'green_infrastructure', 'x_min': 9.0, 'x_max': 11.0, 'multiplier': param_constants['rain_mult_1']},
    ]
    
    rain_source = rainfall_scenario(
        from_date=config.start_datetime,
        to_date=config.end_datetime,
        meteostat_station='SOK6B',
        meteostat_agg_hours=6,
        zones=rain_zones
    )

    domain = Domain(nx=60, ny=30, Lx=20.0, Ly=5.0)
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

    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)

    bc_manager = BoundaryConditionManager(
        V, param_constants['wt_left'], param_constants['wt_right'],
        left_trend=(generic_param['end_datetime'], param_constants['wt_left'] - param_constants['wt_trend']),
        right_trend=(generic_param['end_datetime'], param_constants['wt_right'] - param_constants['wt_trend']),
        time_converter=config.time_converter
    )

    probe_locations = [(8.0, 1.0), (10.0, 1.0), (12.5, 1.0)]

    solver = RichardsSolver(
        domain=domain, V=V, field_map=field_map,
        source_scenario=rain_source,
        bc_manager=bc_manager,
        config=config,
        verbose=False
    )

    # Initialize loss accumulator (Firedrake Constant - stays on tape!)
    loss = Constant(0.0)
    n_obs = 0
    cutoff = int(0.2 * config.num_steps)  # Skip first 20%
    
    # Run solver with custom timestep callback to compute loss
    u = solver.u  # The solution Function
    
    for step in range(config.num_steps):
        solver.step()  # Advance one timestep
        
        # Only accumulate loss after cutoff
        if step >= cutoff:
            # Evaluate solution at probe locations
            for probe_idx, (x, y) in enumerate(probe_locations):
                # Evaluate Function at point (stays on tape!)
                sim_val = u.at(x, y)  # or u((x, y))
                
                # Get observation (as Constant to keep on tape)
                obs_val = Constant(observations.values[step, probe_idx])
                weight = Constant(observations.weights[step, probe_idx])
                
                # Compute residual (Firedrake operations!)
                residual = (sim_val - obs_val) * weight
                loss = loss + residual * residual
                n_obs += 1
    
    # Normalize
    loss = loss / n_obs
    
    return loss

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
        'theta_r_till': 0.02,
        'theta_s_till': 0.14,
        'alpha_till': 0.9399,
        'n_till': 2.3579,
        'Ks_till': 9e-6,
        
        # Terreau soil parameters (5)
        'theta_r_terreau': 0.02,
        'theta_s_terreau': 0.43,
        'alpha_terreau': 1.1670,
        'n_terreau': 2.1052,
        'Ks_terreau': 4e-5,
        
        # Rain multipliers (2)
        'rain_mult_0': 1.0,
        'rain_mult_1': 6.0,
        
        # Water table conditions (3)
        'wt_left': 0.8,
        'wt_right': 1.5,
        'wt_trend': 0.25
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
    continue_annotation()
    loss = your_simulation(param_constants, observations)  # Returns AdjFloat!
    
    n_blocks = len(get_working_tape().get_blocks())
    print(f"   ✓ Initial loss: {float(loss):.6e}")
    print(f"   ✓ Adjoint tape recorded {n_blocks} operations")
    
    # -------------------------------------------------------------------------
    # STEP 5: Setup optimization
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Setting up optimization problem...")
    
    bounds = create_tight_bounds(initial_params, variation_pct=20.0)
    optimizer = AdjointOptimizer(observations, bounds)
    optimizer.setup_optimization(loss, controls_dict, initial_params)
    
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
        maxiter=10,      # Start with 10, increase to 50 for production
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
        'end_datetime': datetime(2024, 7, 15),
        'timestep_hours': 6,
    }
    
    # Run optimization
    best_params, optimizer = main(generic_param)
    
    print("\n✓ All done! Check the generated plots for detailed results.")