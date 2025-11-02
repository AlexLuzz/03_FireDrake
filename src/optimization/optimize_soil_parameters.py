"""
Integration Example: Optimize soil parameters to fit observed water levels
Wraps your existing Richards simulation in the optimization framework
"""

from datetime import datetime, timedelta
from firedrake import FunctionSpace
from src import *
from parameter_optimizer import (
    RichardsParameterOptimizer, 
    ObservationData, 
    ParameterBounds,
    load_observations_from_csv
)
import numpy as np


def build_simulation_with_params(
    params: dict,
    material_zone: str = "base"
) -> ProbeManager:
    """
    Build and run Richards simulation with given parameters
    This wraps your existing simulation code
    
    Args:
        params: Dictionary with van Genuchten parameters
        material_zone: Which material zone to update with optimized parameters
    
    Returns:
        ProbeManager with simulation results
    """
    
    # Same configuration as your existing code
    config = SimulationConfig(
        name="Parameter_Optimization",
        start_datetime=datetime(2024, 4, 15),
        end_datetime=datetime(2024, 6, 30),
        dt_td=timedelta(hours=1)
    )
    
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

    # Create material with optimized parameters
    optimized_material = Material(
        theta_r=params['theta_r'],
        theta_s=params['theta_s'],
        alpha=params['alpha'],
        n=params['n'],
        Ks=params['Ks']
    )
    
    # Assign materials
    if material_zone == "base":
        domain.assign("base", optimized_material)
        domain.assign("GI", Material.terreau())
    elif material_zone == "GI":
        domain.assign("base", Material.till())
        domain.assign("GI", optimized_material)
    else:
        domain.assign("base", Material.till())
        domain.assign("GI", Material.terreau())

    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)
    
    bc_manager = BoundaryConditionManager(V, left_wt=1.2, right_wt=1.2)
    
    # Setup probes at observation locations
    probe_manager = ProbeManager(domain.mesh)
    # Add probes at locations where you have observations
    # probe_manager.add_probe("obs1", x=5.0, y=1.0)
    # probe_manager.add_probe("obs2", x=10.0, y=1.0)
    
    snapshot_times = [0.0, 5*3600.0, config.t_end * 0.1, config.t_end * 0.3, 
                     config.t_end * 0.7, config.t_end]
    snapshot_manager = SnapshotManager(snapshot_times)
    
    solver = RichardsSolver(
        domain=domain, V=V, field_map=field_map,
        source_scenario=rain_source,
        bc_manager=bc_manager, config=config
    )
    
    # Run simulation
    solver.run(probe_manager, snapshot_manager)
    
    return probe_manager


def create_synthetic_observations(
    true_params: dict,
    observation_times: np.ndarray,
    observation_locations: list,
    noise_level: float = 0.05
) -> ObservationData:
    """
    Create synthetic observations for testing (with noise)
    In practice, you'd load real observations from CSV
    """
    # Run simulation with true parameters
    probe_manager = build_simulation_with_params(true_params, "base")
    
    # Extract water levels at observation points
    n_times = len(observation_times)
    n_locs = len(observation_locations)
    water_levels = np.zeros((n_times, n_locs))
    
    for i, t in enumerate(observation_times):
        for j, (x, y) in enumerate(observation_locations):
            # Get simulated value and add noise
            true_value = probe_manager.get_value_at(t, x, y)
            noise = np.random.normal(0, noise_level * abs(true_value))
            water_levels[i, j] = true_value + noise
    
    return ObservationData(
        times=observation_times,
        locations=observation_locations,
        water_levels=water_levels
    )


def main():
    """Main optimization script"""
    
    # Define parameter bounds (based on soil type)
    bounds = ParameterBounds(
        theta_r=(0.01, 0.10),   # Residual water content
        theta_s=(0.35, 0.50),   # Saturated water content
        alpha=(0.005, 0.05),    # Inverse air entry (1/cm)
        n=(1.2, 2.5),           # Pore size distribution
        Ks=(10.0, 200.0)        # Saturated conductivity (cm/day)
    )
    
    # True parameters (for synthetic test)
    true_params = {
        'theta_r': 0.045,
        'theta_s': 0.43,
        'alpha': 0.015,
        'n': 1.56,
        'Ks': 106.1
    }
    
    # Initial guess (perturbed from true)
    initial_params = {
        'theta_r': 0.06,
        'theta_s': 0.38,
        'alpha': 0.025,
        'n': 1.8,
        'Ks': 150.0
    }
    
    # Define observation times and locations
    simulation_duration = (datetime(2024, 6, 30) - datetime(2024, 4, 15)).total_seconds()
    observation_times = np.linspace(0, simulation_duration, 50)  # 50 observations
    observation_locations = [
        (5.0, 1.0),   # Location 1
        (10.0, 1.0),  # Location 2
    ]
    
    # Create synthetic observations (or load real data)
    print("Generating synthetic observations...")
    observations = create_synthetic_observations(
        true_params,
        observation_times,
        observation_locations,
        noise_level=0.03  # 3% noise
    )
    
    # Or load real observations from CSV:
    # observations = load_observations_from_csv(
    #     csv_path="data/water_level_observations.csv",
    #     location=(5.0, 1.0)
    # )
    
    # Create optimizer
    print("\nInitializing optimizer...")
    optimizer = RichardsParameterOptimizer(
        simulation_builder=build_simulation_with_params,
        observations=observations,
        bounds=bounds,
        initial_params=initial_params,
        material_zone="base"  # Optimize the base material
    )
    
    # Run optimization
    print("\nStarting parameter optimization...")
    print(f"True parameters: {true_params}")
    print(f"Initial parameters: {initial_params}")
    print()
    
    best_params = optimizer.optimize(
        max_iterations=30,
        learning_rate=0.05,
        optimizer_type='adam',
        use_finite_diff=True  # Use finite differences for gradients
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print("\nTrue Parameters:")
    for key, val in true_params.items():
        print(f"  {key:10s}: {val:.6f}")
    
    print("\nOptimized Parameters:")
    for key, val in best_params.items():
        print(f"  {key:10s}: {val:.6f}")
    
    print("\nParameter Errors:")
    for key in true_params.keys():
        error = abs(best_params[key] - true_params[key]) / true_params[key] * 100
        print(f"  {key:10s}: {error:.2f}%")
    
    # Plot results
    print("\nGenerating plots...")
    optimizer.plot_results(save_path="optimization_results.png")
    
    # Run final simulation with optimized parameters
    print("\nRunning final simulation with optimized parameters...")
    final_probe_manager = build_simulation_with_params(best_params, "base")
    
    # Generate full simulation report
    print("\nGenerating simulation report...")
    # You can add your existing report generation here


if __name__ == "__main__":
    main()