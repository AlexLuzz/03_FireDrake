"""
Example: How to add report generation to existing Richards simulation
MINIMAL CHANGES - just add 3 lines at the end!
"""
from datetime import datetime, timedelta
from firedrake import FunctionSpace
from src import *

def main():
    """Main simulation with report generation"""
    
    # ==========================================
    # EXISTING CODE - NO CHANGES
    # ==========================================
    config = SimulationConfig(
        name="Datetime_Duration",
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

    domain.assign("base", Material.till())
    domain.assign("GI", Material.terreau())

    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)
    
    bc_manager = BoundaryConditionManager(V, left_wt=1.2, right_wt=1.2)
    
    probe_manager = ProbeManager(domain.mesh)
    snapshot_times = [0.0, 5*3600.0, config.t_end * 0.1, config.t_end * 0.3, config.t_end * 0.7, config.t_end]
    snapshot_manager = SnapshotManager(snapshot_times)
    
    solver = RichardsSolver(
        domain=domain, V=V, field_map=field_map,
        source_scenario=rain_source,
        bc_manager=bc_manager, config=config
    )
    
    solver.run(probe_manager, snapshot_manager)
        
    plotter = ResultsPlotter(
        config, domain.mesh,
        probe_manager=probe_manager,
        rain_scenario=rain_source,
        domain=domain,
        snapshot_manager=snapshot_manager,
        bc_manager=bc_manager
    )
    
    config.get_sim_duration()
    report = SimulationReport(output_dir=config.output_dir)
    report.print_richards_report(config, domain, plotter, 
                                boundary_conditions=bc_manager)
    
if __name__ == "__main__":
    main()
