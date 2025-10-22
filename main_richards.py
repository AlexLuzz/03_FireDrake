from datetime import datetime
from firedrake import RectangleMesh, FunctionSpace
from setup import *
from physics import *
from solver import *
from visualization import *

def main():
    """Main simulation function"""
    
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        # You can modify config parameters here if needed
        dt=300,
        t_end=305*24*3600
    )
    # ==========================================
    # 2. DEFINE RAIN SCENARIO
    # ==========================================
    rain_zones = [
        RainZone(x_min=0.0, x_max=8.0, multiplier=1.0, name="grass"),
        RainZone(x_min=9.0, x_max=11.0, multiplier=6.0, name="green_infrastructure"),
    ]

    start_from = 10

    rain_scenario = RainScenario.from_csv(
        config.data_input_dir / "RAF_rain.csv",
        start_from=start_from,
        time_col='day',
        rain_col='rain (mm)',
        time_unit="days",
        rain_unit="mm/day",
        zones=rain_zones
    )

    # ==========================================
    # 3. CREATE MESH WITH MATERIALS
    # ==========================================
    till = SoilMaterial.from_curves(name="Till")
    terreau = SoilMaterial.from_curves(name="Terreau")

    mesh = RectangleMesh(100, 50, 20.0, 5.0)
    
    # Create domain with Till as base material
    domain = Domain.homogeneous(mesh, till)
    
    domain.add_rectangle(
        material=terreau,
        x_min=9.0, x_max=11.0,
        y_min=4.0, y_max=5.0,
        name="green_infrastructure"
    )

    V = FunctionSpace(mesh, "CG", 1)
    
    # ==========================================
    # 4. CREATE BOUNDARY CONDITION MANAGER
    # ==========================================
    bc_manager = BoundaryConditionManager(
        V,
        initial_water_table=1.2,
        #water_table_trend={'t_end': 315*86400, 'H0_end': 1.0}
    )
    
    # ==========================================
    # 5. CREATE MONITORING
    # ==========================================
    probe_names = [f"LTC {i+1} (x={x:.1f}m, y={y:.1f}m)" for i, (x, y) in enumerate(config.probes_positions)]
    probe_manager = ProbeManager(mesh, config.probes_positions, probe_names)
    
    # Define 6 snapshot times (in seconds)
    snapshot_times = [
        0.0,
        5*3600.0,   # End of rain
        config.t_end * 0.1,
        config.t_end * 0.3,
        config.t_end * 0.7,
        config.t_end 
    ]
    #snapshot_manager = SnapshotManager(snapshot_times, domain)
    snapshot_manager = None

    # ==========================================
    # 6. CREATE SOLVER
    # ==========================================
    solver = RichardsSolver(mesh, V, domain, rain_scenario, bc_manager, config)
        
    # ==========================================
    # 7. RUN SIMULATION
    # ==========================================
    solver.run(probe_manager, 
               snapshot_manager if snapshot_manager is not None else None
               )
    
    # ==========================================
    # 8. VISUALIZE RESULTS
    # ==========================================
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plotter = ResultsPlotter(config, mesh)
    plotter.plot_complete_results(
        probe_data=probe_manager.get_data(),
        snapshots=snapshot_manager.snapshots if snapshot_manager is not None else None,
        rain_scenario=rain_scenario,
        filename=config.output_dir / f'RAF_rains_{now}.png',
        comsol_data_file=config.data_input_dir / 'RAF_COMSOL_PZ_CG.csv',  # Optional: overlay measured data
        plot_residuals=True,  # Plot residuals (Measured - Simulated)
        start_from=start_from
    )

    # ==========================================
    # 10. SAVE PROBE DATA TO CSV
    # ==========================================
    save_prob_data = True  # Set to True to save probe data
    if save_prob_data:
        probe_manager.save_to_csv(config.data_output_dir / "all_RAF_rain_modeled_300s.csv", reference_elevation=5.0)

if __name__ == "__main__":
    main()