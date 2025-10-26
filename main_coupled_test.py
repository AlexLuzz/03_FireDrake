"""
Coupled flow-transport simulation
Chloride infiltration from de-icing salt application
"""
from datetime import datetime, timedelta
from firedrake import FunctionSpace

# Solver imports
from solver_V2 import *
from visualization import *
from setup import *
from setup.source_term import rainfall_scenario, SourceScenario

# Physics imports
from physics_V2 import *
from physics_V2.transport_models import chloride_transport

def main_transport():
    """Main coupled flow-transport simulation"""
    
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        name="Transport_Chloride",
        start_datetime=datetime(2024, 5, 1),
        end_datetime=datetime(2024, 5, 15),
        dt_td=timedelta(hours=12)  # Shorter timestep for transport
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
        rate_unit_conversion=1.0  # kg/m²/s (SI units)
    )
    
    # Define chloride application zone (surface of GI)
    chloride_source.add_zone(
        "deicing_zone", 
        x_min=9.0, 
        x_max=11.0, 
        y_min=4.9,  # Very top surface
        y_max=5.0,
        multiplier=1.0
    )
    
    # Add chloride application events
    # Simulate 3 de-icing salt applications over the simulation period
    applications = [
        {'start': 0.0, 'duration': 2.0, 'rate': 0.1},      # Day 1: 0.1 kg/m²/hr for 2 hours
        {'start': 120.0, 'duration': 3.0, 'rate': 0.15},   # Day 6: 0.15 kg/m²/hr for 3 hours
        {'start': 240.0, 'duration': 2.0, 'rate': 0.12},   # Day 11: 0.12 kg/m²/hr for 2 hours
    ]
    
    for i, app in enumerate(applications):
        chloride_source.add_event(
            name=f"chloride_app_{i}",
            start=app['start'],
            end=app['start'] + app['duration'],
            rate=app['rate'],
            zones="deicing_zone"
        )
    
    print(f"✓ Created {len(chloride_source.events)} chloride application events")
    
    # ==========================================
    # 4. GEOMETRY
    # ==========================================
    domain = Domain(nx=80, ny=40, Lx=20.0, Ly=5.0)
    domain.add_rectangle("GI", 9.0, 11.0, 4.0, 5.0)
    
    # ==========================================
    # 5. MATERIALS WITH TRANSPORT
    # ==========================================
    # Create materials with chloride transport properties
    till_mat = till()
    till_mat.transport = chloride_transport(
        alpha_L=0.05,      # Longitudinal dispersivity [m]
        bulk_density=1600.0  # Till bulk density [kg/m³]
    )
    
    terreau_mat = terreau()
    terreau_mat.transport = chloride_transport(
        alpha_L=0.03,      # Terreau has slightly lower dispersivity
        bulk_density=1200.0  # Terreau bulk density [kg/m³]
    )
    
    domain.assign("base", till_mat)
    domain.assign("GI", terreau_mat)
    domain.validate()
    
    print(f"✓ Materials assigned with transport properties")
    
    # ==========================================
    # 6. FIELD MAPPING
    # ==========================================
    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)
    
    if not field_map.has_transport():
        raise RuntimeError("Transport models not properly assigned!")
    
    # ==========================================
    # 7. BOUNDARY CONDITIONS (Flow)
    # ==========================================
    bc_manager = BoundaryConditionManager(
        V,
        initial_water_table=1.2,
        water_table_trend=None,
        time_converter=config.time_converter
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
        config=config
    )
    
    # Set initial concentration (clean everywhere)
    transport.set_initial_concentration(0.0)
    
    print(f"✓ Transport solver initialized")
    
    # ==========================================
    # 10. MONITORING SETUP
    # ==========================================
    # Probe manager for chloride concentration monitoring
    probe_manager = ProbeManager(domain.mesh)
    
    # Snapshot manager for concentration field visualization
    snapshot_times = [
        0.0,
        2*3600.0,      # After first chloride application
        5*3600.0,      # Day 1
        24*3600.0,     # Day 2
        120*3600.0,    # After second application
        240*3600.0,    # After third application
        config.t_end   # Final time
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
    
    # Get probe data and extract concentration for plotting
    probe_data_raw = probe_manager.get_data()
    
    # Convert nested probe data to flat structure for concentration plotting
    concentration_data = {}
    for probe_name, fields in probe_data_raw['data'].items():
        if 'concentration' in fields:
            concentration_data[probe_name] = fields['concentration']
    
    # Create plotting-compatible data structure
    plot_data = {
        'times': probe_data_raw['times'],
        'data': concentration_data
    }
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for probe_name, conc_values in concentration_data.items():
        times_days = [t / 86400.0 for t in probe_data_raw['times']]
        plt.plot(times_days, conc_values, label=probe_name)

    plt.xlabel("Time (days)")
    plt.ylabel("Chloride Concentration (mg/L)")
    plt.title("Chloride Transport Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"chloride_transport_{now}.png")
    plt.close()

if __name__ == "__main__":
    main_transport()