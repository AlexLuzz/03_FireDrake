"""
Coupled flow-transport simulation
Chloride infiltration from de-icing salt application
"""
from datetime import datetime, timedelta
from firedrake import FunctionSpace
from src import *

def main_transport():
    """Main coupled flow-transport simulation"""
    
    t_launch = datetime.now()

    # ==========================================
    # 1. CONFIGURATION
    # ==========================================

    config = SimulationConfig(
        name="Transport_Chloride",
        start_datetime=datetime(2024, 5, 1),
        end_datetime=datetime(2024, 5, 10),
        dt_td=timedelta(hours=6)  # Smaller timestep for smoother transport curves
    )
    
    # ==========================================
    # 2. RAIN SCENARIO (keep as is)
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
    # 3. CHLORIDE SOURCE 
    # ==========================================
    # Create chloride application scenario on the GI zone
    chloride_source = SourceScenario(
        time_unit="hours", 
        rate_unit_conversion=1.0/3600.0  # Convert kg/m²/hr to kg/m²/s
    )
    
    # Define chloride application zone (surface of GI)
    chloride_source.add_zone(
        "deicing_zone", 
        x_min=9.0, 
        x_max=11.0, 
        y_min=4.8,  # Very top surface
        y_max=5.0,
        multiplier=1.0
    )

    chloride_source.add_event(
        name="deicing_zone",
        start=config.t_end_hours*0.05,
        end=config.t_end_hours*0.3,
        rate=0.1,
        zones="deicing_zone"
    )
    
    print(f"✓ Created {len(chloride_source.events)} chloride application events")
    
    # ==========================================
    # 4. GEOMETRY
    # ==========================================
    domain = Domain(nx=80, ny=40, Lx=20.0, Ly=5.0)
    #domain.add_rectangle("GI", 9.0, 11.0, 4.0, 5.0)
    
    # ==========================================
    # 5. MATERIALS WITH TRANSPORT
    # ==========================================
    domain.assign("base", Material.till_curve_RAF(transport=True))
    #domain.assign("GI", Material.terreau_curve_RAF(transport=True))

    # ==========================================
    # 6. FIELD MAPPING
    # ==========================================
    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)
        
    # ==========================================
    # 7. BOUNDARY CONDITIONS (Flow)
    # ==========================================
    bc_manager = BoundaryConditionManager(
        V,
        left_wt=3,
        right_wt=4.5
    )
    
    # ==========================================
    # 8. RICHARDS SOLVER (Flow with Rain Source)
    # ==========================================
    richards = RichardsSolver(
        domain=domain,
        V=V,
        field_map=field_map,
        source_scenario=rain_source,  # Rain affects flow
        bc_manager=bc_manager,
        config=config
    )
    
    # ==========================================
    # 9. TRANSPORT SOLVER (with Chloride Source)
    # ==========================================
    transport = TransportSolver(
        domain=domain,
        V=V,
        field_map=field_map,
        pressure_solver=richards,
        bc_manager=None,  # No transport BCs for now
        transport_source=chloride_source,  # Chloride affects transport
        config=config,
    )
    
    # ==========================================
    # 10. MONITORING SETUP
    # ==========================================
    chloride_probe_positions = [
        [10.0, 4.8],   # Near source zone (top of GI)
        [10.0, 3.0],   # Middle of GI zone
        [10.0, 1.5],   # Lower in domain (infiltration path)
        [7.0, 1.5],    # Outside GI zone (control point)
    ]
    chloride_probe_names = [
        "LTC 1", 
        "LTC 2", 
        "LTC 3", 
        "LTC 4"
    ]
    
    probe_manager = ProbeManager(
        domain.mesh, 
        probe_positions=chloride_probe_positions,
        names=chloride_probe_names
    )
    
    # Snapshot manager for concentration field visualization
    snapshot_times = [
        0.0,
        config.t_end * 0.1,      # After first chloride application
        config.t_end * 0.2,      # Day 1
        config.t_end * 0.3,      # Day 2
        config.t_end * 0.5,      # After second application
        config.t_end * 0.7,      # After third application
        config.t_end             # Final time
    ]
    snapshot_manager = SnapshotManager(snapshot_times)
    
    print(f"✓ Monitoring setup complete")
    
    # ==========================================
    # 11. RUN TRANSPORT SIMULATION
    # ==========================================
    print("Starting transport simulation...")
    
    # Run the transport solver (it will handle the time loop internally)
    transport.run(probe_manager, snapshot_manager)
    
    print("✓ Transport simulation completed")
    
    # ==========================================
    # 12. VISUALIZATION
    # ==========================================
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create chloride-specific plotter
    plotter = ResultsPlotter(
        config, domain.mesh, 
        probe_manager=probe_manager, 
        rain_scenario=rain_source,  # Show rain events for context
        domain=domain,
        snapshot_manager=snapshot_manager
    )
    
    # Configure chloride-specific plotting (no COMSOL/measured comparisons for chloride)
    plotting_config = {
        'time_series_fields': ['concentration'],  # Changed from old API
        'plot_comsol_comparison': False,          # No chloride reference data available
        'plot_measured_comparison': False,        # No chloride reference data available
        'plot_snapshots': True,                   # Show chloride concentration plumes
        'snapshot_fields': ['concentration'],     # Changed from old API
        'snapshot_overlay': False                 # Added for new API
    }
    
    plotter.plot_complete_results(
        filename=config.output_dir / f'chloride_transport_{now}.png',
        plotting_config=plotting_config
    )
    
    print(f"✓ Chloride visualization saved")
    t_total = datetime.now() - t_launch
    print(f"Total simulation time: {t_total}")

if __name__ == "__main__":
    main_transport()