"""
Transport solver verification script using VLSPI analytical solution

This script verifies the transport solver against the analytical solution
from Sethi di Molfetta 2019 for conservative transport in a steady-state flow field.
Uses SourceScenario for chloride pulse definition and leverages existing plotting functions.
"""

import numpy as np
import sys
import os

from ufl import FunctionSpace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import *


def C_analytical(x, t, v, D, L, M):
    """
    Analytical solution for VLSPI (Sethi di Molfetta 2019)
    
    Args:
        x: spatial coordinate [m]
        t: time [s]
        v: pore water velocity [m/s]
        D: dispersion coefficient [m²/s]
        L: domain length [m]
        M: initial mass per unit width [kg/m]
    
    Returns:
        Concentration [kg/m³]
    """
    if t <= 0:
        return 0.0
    
    # VLSPI solution components
    advection_term = np.exp((v * x) / (2 * D))
    diffusion_term = np.exp(-((x - v * t)**2) / (4 * D * t))
    normalization = M / np.sqrt(4 * np.pi * D * t)
    
    return normalization * advection_term * diffusion_term


def compute_analytical_at_probes(probe_manager, times, v, D, L, M):
    """Compute analytical solution at probe locations"""
    probe_positions = probe_manager.probe_positions
    analytical_data = {}
    
    for i, (x_pos, y_pos) in enumerate(probe_positions):
        probe_name = f"Probe_{i+1}"
        analytical_data[probe_name] = []
        
        for t in times:
            c_analytical = C_analytical(x_pos, t, v, D, L, M)
            analytical_data[probe_name].append(c_analytical)
    
    return analytical_data


def compute_analytical_snapshots(snapshot_manager, domain, v, D, L, M):
    """Compute analytical solution snapshots"""
    analytical_snapshots = {}
    mesh = domain.mesh
    coords = mesh.coordinates.dat.data_ro
    
    for t in snapshot_manager.snapshot_times:
        if t == 0:
            continue  # Skip t=0 for analytical
            
        # Create Function for this snapshot
        from firedrake import Function
        analytical_func = Function(snapshot_manager.V)
        analytical_values = np.zeros(len(coords))
        
        for i, (x, y) in enumerate(coords):
            analytical_values[i] = C_analytical(x, t, v, D, L, M)
        
        analytical_func.dat.data[:] = analytical_values
        analytical_snapshots[t] = analytical_func
    
    return analytical_snapshots


def create_residual_data(sim_data, analytical_data):
    """Create residual data structure matching sim_data format"""
    residual_data = {}
    residual_data['times'] = sim_data['times']
    
    for probe_name in [k for k in sim_data.keys() if k != 'times']:
        sim_vals = np.array(sim_data[probe_name])
        ana_vals = np.array(analytical_data[probe_name])
        residual_data[probe_name] = (sim_vals - ana_vals).tolist()
    
    return residual_data


def create_residual_snapshots(sim_snapshots, analytical_snapshots):
    """Create residual snapshots"""
    residual_snapshots = {}
    
    for t in sim_snapshots.keys():
        if t in analytical_snapshots:
            from firedrake import Function
            residual_func = Function(sim_snapshots[t].function_space())
            
            # Compute residual: simulated - analytical
            sim_data = sim_snapshots[t].dat.data[:]
            ana_data = analytical_snapshots[t].dat.data[:]
            residual_func.dat.data[:] = sim_data - ana_data
            
            residual_snapshots[t] = residual_func
    
    return residual_snapshots


def main():
    """Main verification script"""
    print("=== Transport Solver VLSPI Verification ===")
    

    # Domain and discretization
    Lx, Ly = 10.0, 2.0
    nx, ny = 50, 20
    dt = 6*3600
    t_end = 24*3600*10

    domain = Domain(nx, ny, Lx, Ly)
    domain.assign("base", till(transport=True))

    
    # Configuration
    config = SimulationConfig(
        name="Transport_VLSPI_Conservative",
        dt=dt,
        t_end=t_end
    )


    # Set up chloride pulse using SourceScenario
    pulse_center = 2.0  # m
    pulse_width = 0.5   # m
    pulse_mass = 1.0    # kg/m
    
    chloride_scenario = SourceScenario(time_unit="seconds")
    
    # Create pulse zone (small area around pulse center)
    pulse_zone = chloride_scenario.add_zone(
        name="pulse", 
        x_min=pulse_center - pulse_width/2, 
        x_max=pulse_center + pulse_width/2,
        y_min=0.8, 
        y_max=1.2  # Small y-range around middle
    )
    
    # Add instantaneous pulse event at t=0
    chloride_scenario.add_event(
        name="chloride_pulse",
        start=0.0,
        end=config.dt,  # Very short duration pulse
        rate=pulse_mass / (pulse_width * config.dt),  # kg/m³/s to get total mass
        zones=[pulse_zone]
    )
    
    # Transport parameters for verification
    porosity = 0.3  # Used as theta in saturated conditions
    v_x = 1e-4      # m/s (pore water velocity)
    v_y = 0.0       # m/s
    D_molecular = 1e-9  # m²/s
    alpha_L = 0.1       # m (longitudinal dispersivity)
    alpha_T = 0.01      # m (transverse dispersivity)
    
    # Calculate effective dispersion
    v_magnitude = np.sqrt(v_x**2 + v_y**2)
    D_xx = D_molecular + alpha_L * v_magnitude  # Longitudinal
    D_yy = D_molecular + alpha_T * v_magnitude  # Transverse
    D_xy = 0.0  # No cross-dispersion for aligned flow
    
    # Set up transport solver parameters dictionary
    transport_params = {
        'porosity': porosity,
        'vx': v_x,
        'vy': v_y,
        'Dxx': D_xx,
        'Dyy': D_yy,
        'Dxy': D_xy,
        'R': 1.0,      # No retardation
        'lambda': 0.0  # No decay
    }
    
    

    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)
    bc_manager = BoundaryConditionManager(
        V,
        left_wt=Ly,
        right_wt=Ly
    )
    transport_solver = TransportSolver(domain, V, field_map, bc_manager, chloride_scenario, config)
    
    # Set up monitoring
    probe_positions = [(2.0, 1.0), (4.0, 1.0), (6.0, 1.0), (8.0, 1.0)]
    probe_manager = ProbeManager(probe_positions)
    
    snapshot_times = [0, 1800, 3600, 7200]  # 0, 0.5h, 1h, 2h
    snapshot_manager = SnapshotManager(snapshot_times)
    
    # Run transport simulation with prescribed parameters
    print("Running transport simulation...")
    sim_data = transport_solver.run(
        probe_manager=probe_manager,
        snapshot_manager=snapshot_manager,
        params=transport_params
    )
    
    # Compute analytical solution at probe locations
    times = sim_data['times']
    analytical_data = compute_analytical_at_probes(
        probe_manager, times, v_x, D_xx, Lx, pulse_mass
    )
    
    # Compute analytical snapshots
    analytical_snapshots = compute_analytical_snapshots(
        snapshot_manager, domain, v_x, D_xx, Lx, pulse_mass
    )
    
    # Create residual data
    residual_data = create_residual_data(sim_data, analytical_data)
    residual_snapshots = create_residual_snapshots(
        snapshot_manager.snapshots, analytical_snapshots
    )
    
    # Set up plotting with existing classes
    plotter = ResultsPlotter(
        domain=domain, 
        config=config,
        probe_manager=probe_manager,
        snapshot_manager=snapshot_manager
    )
    
    # Update plotter with simulation data
    plotter.sim_data = sim_data
    
    # Generate report with custom configuration
    report_config = {
        'parameters': True,
        'model_setup': False,  # Skip because no materials/BC for this test
        'time_series_plots': True,
        'snapshot_plots': False,  # Will use custom snapshot pages
        
        # Custom page: Simulated vs Analytical timeseries 
        'custom_page_comparison': {
            'type': 'timeseries_comparison',
            'sim_data': sim_data,
            'analytical_data': analytical_data,
            'title': 'VLSPI Verification: Simulated vs Analytical',
            'ylabel': 'Concentration (kg/m³)'
        },
        
        # Custom page: Residuals timeseries
        'custom_page_residuals': {
            'type': 'timeseries',
            'data': residual_data,
            'title': 'Residuals (Simulated - Analytical)',
            'ylabel': 'Residual (kg/m³)'
        },
        
        # Custom page: Simulated snapshots
        'custom_page_sim_snapshots': {
            'type': 'snapshots',
            'data': snapshot_manager.snapshots,
            'title': 'Simulated Concentration'
        },
        
        # Custom page: Analytical snapshots
        'custom_page_ana_snapshots': {
            'type': 'snapshots',
            'data': analytical_snapshots,
            'title': 'Analytical Concentration'
        },
        
        # Custom page: Residual snapshots
        'custom_page_res_snapshots': {
            'type': 'snapshots',
            'data': residual_snapshots,
            'title': 'Residual Concentration (Simulated - Analytical)'
        }
    }
    
    # Generate report using the report generator
    from src.visualization.report_generator import SimulationReport
    
    report_generator = SimulationReport()
    report_generator.print_transport_verification_report(
        config=config,
        domain=domain,
        plotter=plotter,
        report_config=report_config,
        filename="transport_vlspi_verification_report.pdf"
    )
    
    # Print error statistics
    print("\n=== Error Analysis ===")
    for probe_name in analytical_data.keys():
        sim_vals = np.array(sim_data[probe_name])
        ana_vals = np.array(analytical_data[probe_name])
        
        # Skip first time point (t=0) where both should be zero
        sim_vals = sim_vals[1:]
        ana_vals = ana_vals[1:]
        
        residuals = sim_vals - ana_vals
        rmse = np.sqrt(np.mean(residuals**2))
        max_error = np.max(np.abs(residuals))
        
        print(f"{probe_name}:")
        print(f"  RMSE: {rmse:.2e} kg/m³")
        print(f"  Max Error: {max_error:.2e} kg/m³")
    
    print("\nVerification completed successfully!")
    print(f"Report saved as: transport_vlspi_verification_report.pdf")




if __name__ == "__main__":
    main()