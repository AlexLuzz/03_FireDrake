from datetime import datetime, timedelta
from firedrake import FunctionSpace

from solver_V2 import *
from visualization import *
from setup import *
from setup.source_term import rainfall_scenario

# New architecture imports
from physics_V2 import *

def main():
    """Main simulation with new architecture"""
    
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        name="Datetime_Duration",
        start_datetime=datetime(2024, 5, 1),
        end_datetime=datetime(2024, 5, 30),
        dt_td=timedelta(hours=6)
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
        meteostat_station='SOK6B',
        meteostat_agg_hours=6,
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
    domain.assign("base", till())
    domain.assign("GI", terreau())
    domain.validate()
    
    # ==========================================
    # 5. MAPPING (connect materials to domain)
    # ==========================================
    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)
    
    # ==========================================
    # 7. BOUNDARY CONDITIONS
    # ==========================================
    bc_manager = BoundaryConditionManager(
        V,
        initial_water_table=1.2,
        water_table_trend=None,
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
    # 9. SOLVER (now receives field_map!)
    # ==========================================
    solver = RichardsSolver(
        domain=domain,
        V=V,
        field_map=field_map,  # NEW: field_map instead of domain
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
        config, domain.mesh, 
        probe_manager=probe_manager, 
        rain_scenario=rain_source, 
        domain=domain,
        snapshot_manager=snapshot_manager
    )
    
    # Configure what to plot
    plotting_config = {
        'time_series': True,
        'plot_comsol_comparison': True,     # Uses default COMSOL file from import_results.py
        'plot_measured_comparison': True,   # Uses default measured file from import_results.py
        'plot_snapshots': False              # Will plot if snapshot_manager has data
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