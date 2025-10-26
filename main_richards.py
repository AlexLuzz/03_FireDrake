from datetime import datetime
from firedrake import *
from physics import *
from solver import *
from visualization import *
from datetime import datetime, timedelta
from setup import *
from setup.source_term import rainfall_scenario

def main():
    """Main simulation function"""
    
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        name="Datetime_Duration",
        start_datetime=datetime(2024, 4, 1), # YYYY, MM, DD
        end_datetime=datetime(2024, 4, 30),
        dt_td=timedelta(hours=2)
    )
    # ==========================================
    # 2. DEFINE RAIN SCENARIO
    # ==========================================
    # Define zones with different multipliers
    rain_zones = [
        {'name': 'grass', 'x_min': 0.0, 'x_max': 8.0, 'multiplier': 1.0},
        {'name': 'green_infrastructure', 'x_min': 9.0, 'x_max': 11.0, 'multiplier': 6.0},
    ]
    
    # Create rainfall scenario from CSV
    rain_source = rainfall_scenario(
        csv_path=config.data_input_dir / "BB_METEO.csv",
        from_date=config.start_datetime,
        to_date=config.end_datetime,
        #meteostat_station='SOK6B',
        #meteostat_agg_hours=6,
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
    water_table_trend={
            'start_datetime': datetime(2024, 2, 22),  # Trend starts at COMSOL reference date
            'end_datetime': datetime(2024, 12, 31),    # 315 days later
            'H0_start': 1.2,
            'H0_end': 1.0
        }
    
    bc_manager = BoundaryConditionManager(
        V,
        initial_water_table=1.2,
        water_table_trend=water_table_trend,
        time_converter=config.time_converter
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
    solver = RichardsSolver(V, domain, rain_source, bc_manager, config)
        
    # ==========================================
    # 7. RUN SIMULATION
    # ==========================================
    solver.run(probe_manager, snapshot_manager)
    
    # ==========================================
    # 8. VISUALIZE RESULTS
    # ==========================================
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plotter = ResultsPlotter(config, domain.mesh, probe_manager, rain_source, domain,
                             #snapshot_manager
                             )
    
    # Example: Individual piezometer offsets for Chabanel-Galinée IVD
    measured_offsets = {
         "LTC 101": 0.60,  # real altitude: 32.072
         "LTC 102": 0.70,  # real altitude: 32.094
         "LTC 103": 0.35   # real altitude: 31.878
     }
    
    plotter.plot_complete_results(
        filename=config.output_dir / f'rain_simulation_{now}_TEST.png',
        measured_offset=measured_offsets
    )
    
    # ==========================================
    # 9. CREATE GIF ANIMATION (OPTIONAL)
    # ==========================================
    from visualization import GifAnimator
    plot_gif_animation = False
    if plot_gif_animation:
        animator = GifAnimator(snapshot_manager.snapshots, config)
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