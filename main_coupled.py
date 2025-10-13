from firedrake import RectangleMesh, FunctionSpace, Constant, conditional, SpatialCoordinate
from config import SimulationConfig
from physics import *
from solver import *
from visualization import *

def main():
    """Main coupled flow-transport simulation function"""
    
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        dt=600,  # 10 minutes for Richards (transport will sub-step)
        t_end=150*3600,
        monitor_x_positions=[8.0, 10.0, 12.5],
    )
    
    # ==========================================
    # 2. DEFINE RAIN SCENARIO
    # ==========================================
    rain_zones = [
        RainZone(x_min=0.0, x_max=8.0, multiplier=1.0, name="grass"),
        RainZone(x_min=9.0, x_max=11.0, multiplier=6.0, name="green_infrastructure"),
    ]
    rain_scenario = RainScenario.single_event(
        start_hours=3.0,
        end_hours=5.0,
        intensity_mm_hr=20.0,
        zones=rain_zones
    )

    # ==========================================
    # 3. CREATE MESH WITH MATERIALS
    # ==========================================
    till = SoilMaterial.from_curves(name="Till")
    terreau = SoilMaterial.from_curves(name="Terreau")

    mesh = RectangleMesh(80, 40, 20.0, 5.0)
    
    # Create domain with Till as base material
    domain = Domain.homogeneous(mesh, till)
    
    domain.add_rectangle(
        material=terreau,
        x_min=9.0, x_max=11.0,
        y_min=4.0, y_max=5.0,
        name="green_infrastructure"
    )

    V = FunctionSpace(mesh, "CG", 1)
    
    # ==========================================
    # 4. CREATE RICHARDS SOLVER
    # ==========================================
    bc_manager = BoundaryConditionManager(V, config)

    richards_solver = RichardsSolver(mesh, V, domain, rain_scenario, bc_manager, config)
    
    print("✓ Richards solver created")
    
    # ==========================================
    # 5. CREATE CHLORIDE TRANSPORT SOLVER
    # ==========================================
    print("\nSetting up chloride transport...")
    
    # Porous media properties
    porous_props = {
        'porosity': till.porosity,  # Will use spatially varying if heterogeneous
        'initial_saturation': 0.5  # Start partially saturated
    }
    
    # Transport properties for chloride
    transport_props = {
        'molecular_diffusion': 2.03e-9,  # Cl- at 25°C (m²/s)
        'longitudinal_dispersivity': 0.05,  # 5 cm
        'transverse_dispersivity': 0.005,  # 5 mm
        'tortuosity': 'millington_quirk',
        'include_adsorption': False
    }
    
    transport_solver = ChlorideTransport(mesh, porous_props, transport_props)
    
    # Initial chloride concentration: source at green infrastructure
    x, y = SpatialCoordinate(mesh)
    c_init = conditional(
        (x >= 9.0) & (x <= 11.0) & (y >= 4.0),  # Green infrastructure area
        Constant(100.0),  # 100 mol/m³
        Constant(0.0)     # Clean elsewhere
    )
    transport_solver.set_initial_condition(c_init)
    
    print("✓ Chloride transport solver created")
    print(f"  - Initial concentration: 100 mol/m³ at green infrastructure")
    print(f"  - Molecular diffusion: {transport_props['molecular_diffusion']:.2e} m²/s")
    print(f"  - Dispersivity: αL={transport_props['longitudinal_dispersivity']}m, "
          f"αT={transport_props['transverse_dispersivity']}m")
    
    # ==========================================
    # 6. CREATE COUPLED SOLVER
    # ==========================================
    print("\nCreating coupled flow-transport solver...")
    coupled_solver = CoupledFlowTransport(richards_solver, transport_solver)
    
    print("✓ Coupled solver created")
    
    # ==========================================
    # 7. CREATE MONITORING
    # ==========================================
    print("\nSetting up monitoring...")
    
    probe_names = [f"LTC {i+1} (x={x:.1f}m)" for i, x in enumerate(config.monitor_x_positions)]
    probe_manager = ProbeManager(mesh, config.monitor_x_positions, probe_names)
    
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
    
    print(f"✓ Monitoring: {len(probe_names)} probes, {len(snapshot_times)} snapshots")
    
    # ==========================================
    # 8. BOUNDARY CONDITIONS FOR TRANSPORT
    # ==========================================
    # Chloride BC: constant concentration at green infrastructure surface
    transport_bc = {
        3: Constant(100.0)  # Top boundary (ds(3))
    }
    
    # ==========================================
    # 9. RUN COUPLED SIMULATION
    # ==========================================
    print("\n" + "=" * 70)
    print("STARTING COUPLED SIMULATION")
    print("=" * 70 + "\n")
    
    results = coupled_solver.run(
        t_end=config.t_end,
        dt_richards=config.dt,  # 10 minutes for Richards
        dt_transport=300.0,     # 5 minutes for transport (sub-stepping)
        transport_bc=transport_bc,
        dirichlet_bcs=None,  # No Dirichlet BC for Richards
        output_interval=3600.0,  # Save every hour
        probe_manager=probe_manager,
        snapshot_manager=snapshot_manager,
        print_diagnostics=True
    )
    
    # ==========================================
    # 10. VISUALIZE FLOW RESULTS
    # ==========================================
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    
    print("\nCreating flow visualization...")
    plotter = ResultsPlotter(config, mesh)
    plotter.plot_complete_results(
        probe_data=probe_manager.get_data(),
        snapshots=snapshot_manager.snapshots,
        rain_scenario=rain_scenario,
        filename='coupled_flow_results.png'
    )
    
    # ==========================================
    # 11. VISUALIZE TRANSPORT RESULTS
    # ==========================================
    print("\nCreating transport visualization...")
    
    # Simple concentration plot
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Chloride Transport Evolution', fontsize=16, fontweight='bold')
    
    coords = mesh.coordinates.dat.data
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    sorted_times = sorted(snapshot_manager.snapshots.keys())[:6]
    
    for idx, t in enumerate(sorted_times):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Get concentration at this time
        # Note: transport_solver.c contains the final concentration
        # For intermediate times, you'd need to save them during simulation
        c_vals = transport_solver.c.dat.data[:]
        
        # Create regular grid for interpolation
        xi = np.linspace(0, 20.0, 200)
        yi = np.linspace(0, 5.0, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate
        coord_points = np.column_stack((x_coords, y_coords))
        interp = LinearNDInterpolator(coord_points, c_vals)
        Zi = interp(Xi, Yi)
        
        # Plot
        levels = np.linspace(0, 100, 25)
        contourf = ax.contourf(Xi, Yi, Zi, levels=levels,
                              cmap='YlOrRd', extend='both')
        
        # Green infrastructure outline
        from matplotlib.patches import Rectangle
        rect = Rectangle((9, 4), 2, 1, linewidth=2, edgecolor='green', 
                        facecolor='none', linestyle='-')
        ax.add_patch(rect)
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f't = {t/3600:.1f}h')
        ax.set_aspect('equal')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 5)
        
        if idx == 0:
            cbar = plt.colorbar(contourf, ax=ax)
            cbar.set_label('Cl⁻ (mol/m³)')
    
    plt.tight_layout()
    plt.savefig('coupled_transport_results.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()