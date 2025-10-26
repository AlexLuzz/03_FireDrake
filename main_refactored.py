from datetime import datetime, timedelta
from firedrake import RectangleMesh, FunctionSpace

# Old imports (your existing modules)
from physics import *
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
        end_datetime=datetime(2024, 6, 30),
        dt_td=timedelta(hours=4)
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
    Lx, Ly = 20.0, 5.0
    nx, ny = 80, 40
    
    domain = Domain(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
    domain.add_layer("base", 0.0, 5.0)
    domain.add_rectangle("green_infrastructure", 
                        x_min=9.0, x_max=11.0, 
                        y_min=4.0, y_max=5.0)
    
    # ==========================================
    # 4. MATERIALS (properties)
    # ==========================================
    till = Material.library_till()
    terreau = Material.library_terreau()
    
    # ==========================================
    # 5. MAPPING (connect materials to domain)
    # ==========================================
    field_map = MaterialField(domain)
    field_map.assign("base", till)
    field_map.assign("green_infrastructure", terreau)
    field_map.validate()
    
    # ==========================================
    # 6. FIREDRAKE MESH & FUNCTION SPACE
    # ==========================================
    mesh = RectangleMesh(nx, ny, Lx, Ly)
    V = FunctionSpace(mesh, "CG", 1)
    
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
    probe_manager = ProbeManager(mesh)
    
    snapshot_times = [
        0.0,
        5*3600.0,
        config.t_end * 0.1,
        config.t_end * 0.3,
        config.t_end * 0.7,
        config.t_end
    ]
    snapshot_manager = SnapshotManager(snapshot_times, field_map)
    
    # ==========================================
    # 9. SOLVER (now receives field_map!)
    # ==========================================
    solver = RichardsSolver(
        mesh=mesh,
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
        config, mesh, probe_manager, rain_source, field_map
    )
    
    measured_offsets = {
        "LTC 101": 0.60,
        "LTC 102": 0.70,
        "LTC 103": 0.35
    }
    
    plotter.plot_complete_results(
        filename=config.output_dir / f'rain_simulation_{now}.png',
        measured_offset=measured_offsets
    )
    
if __name__ == "__main__":
    main()