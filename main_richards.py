from datetime import datetime, timedelta
from firedrake import FunctionSpace
from src import *
from firedrake import *

def main():
    """Main simulation with new architecture"""
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        project_name="Test",
        user="alexi", # alexi or AQ96560
        start_datetime=datetime(2024, 4, 15),
        end_datetime=datetime(2024, 4, 20),
        dt_td=timedelta(hours=3)
    )
    # ==========================================
    # 2. RAIN SCENARIO
    # ==========================================
    zones = {
    'grass': {
        'bounds': (0.0, 8.0),  # (x_min, x_max)
        'data_col': 'rain',    # Key to look up in the internal DataFrame
        'factor': 1.0          # The old 'multiplier'
    },
    'green_infrastructure': {
        'bounds': (9.0, 11.0),
        'data_col': 'rain',
        'factor': 6.0          # Applied during UFL construction
    }
    }

    # 2. Initialize the Manager
    # The config already contains your start/end dates and paths[cite: 2]
    source_mgr = SourceManager(config, zones)

    # 3. Import Data (Choose one method)

    # Option A: Import from CSV 
    # IMPORTANT: The new import_csv assumes values are already in SI units (m/s)[cite: 1]
    source_mgr.import_csv(
        filepath=config.paths.RAF_METEO,
        datetime_col='Date' # Ensure this matches your CSV header
    )

    # Option B: Fetch directly from Meteostat
    # This handles the unit conversion from mm/day to m/s automatically[cite: 1]
    # source_mgr.import_weather_data(config.start_datetime, config.end_datetime)

    
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
        source_scenario=source_mgr,
        bc_manager=bc_manager,
        config=config,
        probe_manager=probe_manager,
        snapshot_manager=snapshot_manager
    )
    
    # ==========================================
    # 10. RUN
    # ==========================================
    solver.run()
    
    # ==========================================
    # 11. VISUALIZATION
    # ==========================================
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    HydrogeoSimulationReport.print(solver, "prout.pdf")
    
    print(config.get_sim_duration())

if __name__ == "__main__":
    main()