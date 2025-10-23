from datetime import datetime
from firedrake import RectangleMesh, FunctionSpace
from setup import SimulationConfig
from physics import *
from solver import *
from visualization import *
from setup import *

def main():
    """Main simulation function"""
    
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        # You can modify config parameters here if needed
        dt=3600,
        t_end=5*24*3600,
    )
    # ==========================================
    # 2. DEFINE RAIN SCENARIO
    # ==========================================
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

    # ==========================================
    # 3. CREATE DOMAIN WITH MATERIALS
    # ==========================================
    till = SoilMaterial.from_curves(name="Till")
    terreau = SoilMaterial.from_curves(name="Terreau")

    # Create domain with Till as base material - mesh created internally
    domain = Domain.homogeneous(till, Lx=20.0, Ly=5.0, nx=80, ny=40)
    
    domain.add_rectangle(
        material=terreau,
        x_min=9.0, x_max=11.0,
        y_min=4.0, y_max=5.0,
        name="green_infrastructure"
    )

    V = FunctionSpace(domain.mesh, "CG", 1)
    
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
    probe_manager = ProbeManager(domain.mesh)  # Uses default probe positions

    # Define 6 snapshot times (in seconds)
    snapshot_times = [
        0.0,
        5*3600.0,   # End of rain
        config.t_end * 0.1,
        config.t_end * 0.3,
        config.t_end * 0.7,
        config.t_end 
    ]
    snapshot_manager = SnapshotManager(snapshot_times, domain)
    
    # ==========================================
    # 6. CREATE SOLVER
    # ==========================================
    solver = RichardsSolver(domain.mesh, V, domain, rain_event, bc_manager, config)
        
    # ==========================================
    # 7. RUN SIMULATION
    # ==========================================
    solver.run(probe_manager, snapshot_manager)
    
    # ==========================================
    # 8. VISUALIZE RESULTS
    # ==========================================
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plotter = ResultsPlotter(config, domain.mesh)
    plotter.plot_complete_results(
        probe_data=probe_manager.get_data(),
        snapshots=snapshot_manager.snapshots,
        rain_scenario=rain_event,
        filename=config.output_dir / f'rain_simulation_{now}_TEST.png',
    )
    
    # ==========================================
    # 9. CREATE GIF ANIMATION (OPTIONAL)
    # ==========================================
    from visualization import GifAnimator
    plot_gif_animation = False
    if plot_gif_animation:
        animator = GifAnimator(snapshot_manager.snapshots, config, domain.mesh)
        animator.create_animation(
            field_name='saturation',
            filename='results/saturation_evolution.gif',
            fps=2,  # 2 frames per second
            dpi=100,
            cmap='Blues',
            clabel='Saturation',
            vmin=0,
            vmax=1
        )

    # ==========================================
    # 10. SAVE PROBE DATA TO CSV
    # ==========================================
    save_prob_data = False
    if save_prob_data:
        probe_manager.save_to_csv("water_table_data.csv", reference_elevation=5.0)

if __name__ == "__main__":
    main()