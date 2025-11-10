from datetime import datetime, timedelta
from firedrake import FunctionSpace
from src import *
from firedrake import *
from pyadjoint import pause_annotation

pause_annotation()

def main():
    """Main simulation with new architecture"""
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        project_name="Test",
        user="alexi",
        start_datetime=datetime(2024, 4, 15),
        end_datetime=datetime(2024, 5, 30),
        dt_td=timedelta(hours=3)
    )
    
    # ==========================================
    # 2. RAIN SCENARIO
    # ==========================================
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

    
    # ==========================================
    # 3. GEOMETRY (Domain - pure geometry)
    # ==========================================
    domain = Domain(nx=80, ny=40, Lx=20.0, Ly=5.0)
    domain.add_rectangle("GI", 9.0, 11.0, 4.0, 5.0)
    
    # ==========================================
    # 4. MATERIALS (properties)
    # ==========================================
    domain.assign("base", Material.till())
    domain.assign("GI", Material.terreau())

    # ==========================================
    # 5. MAPPING (connect materials to domain)
    # ==========================================
    V = FunctionSpace(domain.mesh, "CG", 1, name="pressure")
    field_map = MaterialField(domain, V)
    
    # ==========================================
    # 7. BOUNDARY CONDITIONS
    # ==========================================
    bc_manager = BoundaryConditionManager(V, 
                                          left_wt=1.2, 
                                          right_wt=1.2,
                                          #left_trend=(config.end_datetime, 0.5),
                                          #right_trend=(config.end_datetime, 1.1),
                                          time_converter=config.time_converter
                                          )

    # ==========================================
    # 8. MONITORING
    # ==========================================
    probe_manager = ProbeManager(domain.mesh)
    
    snapshot_times = [
        0.0,
        5*3600.0,
        config.t_end * 0.1,
        config.t_end * 0.3,
        config.t_end * 0.7,
        config.t_end
    ]
    snapshot_manager = SnapshotManager(snapshot_times)
    
    # ==========================================
    # 9. SOLVER
    # ==========================================
    solver = RichardsSolver(
        V=V,
        field_map=field_map,
        source_scenario=rain_source,
        bc_manager=bc_manager,
        config=config
    )
    
    # ==========================================
    # 10. RUN
    # ==========================================
    solver.run(probe_manager, snapshot_manager)
    
    # ==========================================
    # 11. VISUALIZATION
    # ==========================================
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    plotter = ResultsPlotter(
        config,
        probe_manager=probe_manager, 
        rain_scenario=rain_source, 
        mesh=domain.mesh,
        snapshot_manager=snapshot_manager,
        bc_manager=bc_manager
    )
    
    # Configure what to plot
    plotting_config = {
        'time_series_fields': ['water_table'],      # Water table elevation
        'plot_comsol_comparison': True,            # Set to True to use default COMSOL file
        'plot_measured_comparison': True,          # Set to True to use default measured file
        'plot_snapshots': True,                     # Snapshot plots if data available
        'snapshot_fields': ['saturation'],         # Fields to show in snapshots
    }
    
    # Optional: Override defaults if needed
    # plotting_config.update({
    #     'comsol_data_file': 'path/to/other/comsol.csv',
    #     'comsol_ref_date': datetime(2024, 3, 1),
    #     'measured_data_file': 'path/to/other/measured.csv', 
    #     'measured_offsets': {"LTC 101": 0.5, "LTC 102": 0.6, "LTC 103": 0.4},
    # })
    
    plotter.plot_complete_results(
        filename=config.output_dir / f'rain_simulation_{now}.png',
        plotting_config=plotting_config
    )
    
if __name__ == "__main__":
    main()