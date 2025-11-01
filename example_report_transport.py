"""
Example: How to add report generation to existing Transport simulation
MINIMAL CHANGES - just add a few lines at the end!
"""
from datetime import datetime, timedelta
from firedrake import FunctionSpace
from src import *

def main_transport():
    """Main coupled flow-transport simulation with report"""
    
    # ==========================================
    # EXISTING CODE - NO CHANGES
    # ==========================================
    config = SimulationConfig(
        name="Transport_Chloride",
        start_datetime=datetime(2024, 5, 1),
        end_datetime=datetime(2024, 12, 30),
        dt_td=timedelta(hours=2)
    )
    
    rain_zones = [
        {'name': 'grass', 'x_min': 0.0, 'x_max': 8.0, 'multiplier': 1.0},
        {'name': 'green_infrastructure', 'x_min': 9.0, 'x_max': 11.0, 'multiplier': 6.0},
    ]
    
    rain_source = rainfall_scenario(
        from_date=config.start_datetime, to_date=config.end_datetime,
        meteostat_station='SOK6B', meteostat_agg_hours=6,
        zones=rain_zones
    )
    
    chloride_source = SourceScenario(time_unit="hours", rate_unit_conversion=1.0/3600.0)
    chloride_source.add_zone("deicing_zone", x_min=9.0, x_max=11.0, y_min=4.8, y_max=5.0, multiplier=1.0)
    chloride_source.add_event(name="deicing_zone", start=config.t_end_hours*0.05,
                             end=config.t_end_hours*0.1, rate=0.02, zones="deicing_zone")
    
    domain = Domain(nx=80, ny=40, Lx=20.0, Ly=5.0)
    domain.assign("base", Material.till())
    
    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)
    
    bc_manager = BoundaryConditionManager(V, left_wt=3, right_wt=4.5)
    
    richards = RichardsSolver(domain=domain, V=V, field_map=field_map,
                             source_scenario=rain_source, bc_manager=bc_manager, config=config)
    
    transport = TransportSolver(domain=domain, V=V, field_map=field_map,
                               pressure_solver=richards, bc_manager=None,
                               transport_source=chloride_source, config=config)
    
    chloride_probe_positions = [[10.0, 4.8], [10.0, 3.0], [10.0, 1.5], [7.0, 1.5]]
    chloride_probe_names = ["LTC 1", "LTC 2", "LTC 3", "LTC 4"]
    
    probe_manager = ProbeManager(domain.mesh, probe_positions=chloride_probe_positions,
                                names=chloride_probe_names)
    
    snapshot_times = [0.0, config.t_end * 0.1, config.t_end * 0.2,
                     config.t_end * 0.3, config.t_end * 0.5, config.t_end * 0.7, config.t_end]
    snapshot_manager = SnapshotManager(snapshot_times)
    
    transport.run(probe_manager, snapshot_manager)
    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    plotter = ResultsPlotter(config, domain.mesh, probe_manager=probe_manager,
                            rain_scenario=rain_source, domain=domain, bc_manager=bc_manager,
                            snapshot_manager=snapshot_manager)
    
    plotting_config = {
        'time_series_fields': ['concentration'],
        'plot_comsol_comparison': False,
        'plot_measured_comparison': False,
        'plot_snapshots': True,
        'snapshot_fields': ['concentration'],
        'snapshot_overlay': False
    }
    
    plotter.plot_complete_results(
        filename=config.output_dir / f'chloride_transport_{now}.png',
        plotting_config=plotting_config
    )
    
    # ==========================================
    # NEW: ADD REPORT GENERATION
    # ==========================================
    # Define contaminant properties for report
    contaminant_props = {
        'Contaminant': 'Chloride (Cl⁻)',
        'Application Rate': '0.1 kg/m²/hr',
        'Application Zone': 'Deicing zone (9-11m, surface)',
        'Application Duration': f'{config.t_end_hours*0.25:.1f} hours',
        'Dispersivity (αL)': '0.05 m',
        'Bulk Density': '1600 kg/m³'
    }
    
    report = SimulationReport(output_dir=config.output_dir)
    report.print_transport_report(
        config, domain, field_map, plotter,
        contaminant_props=contaminant_props,
        filename=config.output_dir / f'report_transport_{now}.pdf',
        include_richards=False  # Set True if you want flow results too
    )
    
if __name__ == "__main__":
    main_transport()
