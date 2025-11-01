"""
Transport solver verification using VLSPI analytical solution
Verifies against Sethi di Molfetta 2019 for conservative transport in steady-state flow
"""

import numpy as np
import sys
import os
from firedrake import FunctionSpace, Function
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *

def C_analytical(x, t, v, D, M):
    """
    VLSPI analytical solution (Sethi di Molfetta 2019)
    
    Args:
        x: spatial coordinate [m]
        t: time [s]
        v: pore water velocity [m/s]
        D: dispersion coefficient [m²/s]
        M: initial mass per unit width [kg/m]
    """
    if t <= 0:
        return 0.0
    
    advection = np.exp((v * x) / (2 * D))
    diffusion = np.exp(-((x - v * t)**2) / (4 * D * t))
    normalization = M / np.sqrt(4 * np.pi * D * t)
    
    return normalization * advection * diffusion


def compute_analytical_at_probes(probe_manager, times, v, D, M):
    """Compute analytical solution at probe locations"""
    analytical = {'times': times}
    
    for i, (x, y) in enumerate(probe_manager.probe_positions):
        values = [C_analytical(x, t, v, D, M) for t in times]
        analytical[f'Probe {i+1}'] = np.array(values)
    
    return analytical


def compute_analytical_snapshots(snapshot_manager, domain, v, D, M):
    """Compute analytical solution on full mesh at snapshot times"""
    V = FunctionSpace(domain.mesh, "CG", 1)
    snapshots = {}
    coords = domain.mesh.coordinates.dat.data
    
    for t in snapshot_manager.snapshot_times:
        C_func = Function(V)
        values = np.array([C_analytical(x, t, v, D, M) for x, y in coords])
        C_func.dat.data[:] = values
        snapshots[t] = {'concentration': C_func}
    
    return snapshots


def create_residual_data(sim_data, analytical_data):
    """Compute residuals between simulation and analytical"""
    residuals = {'times': sim_data['times']}
    
    for key in sim_data.keys():
        if key != 'times':
            residuals[key] = np.array(sim_data[key]) - np.array(analytical_data[key])
    
    return residuals


def create_residual_snapshots(sim_snapshots, ana_snapshots):
    """Compute residual snapshots"""
    residuals = {}
    
    for t in sim_snapshots.keys():
        V = sim_snapshots[t]['concentration'].function_space()
        residual = Function(V)
        residual.dat.data[:] = sim_snapshots[t]['concentration'].dat.data[:] - \
                               ana_snapshots[t]['concentration'].dat.data[:]
        residuals[t] = {'concentration': residual}
    
    return residuals


def main():
    print("=== Transport Solver VLSPI Verification ===\n")
    
    # Domain setup
    Lx, Ly = 10.0, 2.0
    nx, ny = 50, 20
    dt = 2*3600
    t_end = 10 * 24 * 3600
    
    domain = Domain(nx, ny, Lx, Ly)
    domain.assign("base", Material.till(contaminant='chloride'))
    
    # Configuration with time converter
    config = SimulationConfig(
        name="Transport_VLSPI_Verification",
        dt=dt,
        t_end=t_end,
    )
    
    # Chloride pulse setup
    pulse_center = 2.0
    pulse_width = 0.5
    pulse_mass = 1.0
    
    chloride_scenario = SourceScenario(time_unit="seconds")
    pulse_zone = chloride_scenario.add_zone(
        name="pulse",
        x_min=pulse_center - pulse_width/2,
        x_max=pulse_center + pulse_width/2,
        y_min=0.8,
        y_max=1.2
    )
    
    chloride_scenario.add_event(
        name="pulse",
        start=0.0,
        end=dt,
        rate=pulse_mass / (pulse_width * dt),
        zones=[pulse_zone]
    )
    
    # Transport parameters (saturated, steady flow)
    porosity = 0.3
    v_x = 1e-4
    D_molecular = 1e-9
    alpha_L = 0.1
    
    v_mag = v_x
    D_eff = D_molecular + alpha_L * v_mag
    
    transport_params = {
        'porosity': porosity,
        'vx': v_x,
        'vy': 0.0,
        'Dxx': D_eff,
        'Dyy': D_molecular,
        'Dxy': 0.0,
        'R': 1.0,
        'lambda': 0.0
    }
    
    # Setup solver
    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)
    bc_manager = BoundaryConditionManager(V, left_wt=Ly, right_wt=Ly)
    transport_solver = TransportSolverAnalytical(domain, V, field_map, bc_manager, 
                                      chloride_scenario, config)
    
    # Monitoring
    probe_positions = [[2.0, 1.0], [4.0, 1.0], [6.0, 1.0], [8.0, 1.0]]
    probe_manager = ProbeManager(domain.mesh, probe_positions)
    
    snapshot_times = [0, t_end*0.1, t_end*0.2, t_end*0.4, t_end*0.75, t_end]
    snapshot_manager = SnapshotManager(snapshot_times)
    snapshot_manager_2 = SnapshotManager(snapshot_times)
    
    # Run simulation
    print("Running transport simulation...")
    transport_solver.run(
        probe_manager=probe_manager,
        snapshot_manager=snapshot_manager,
        params=transport_params
    )
    
    # Extract simulation data from probe manager
    probe_data = probe_manager.get_data()
    sim_data = {'times': probe_data['times']}
    for i, probe_name in enumerate(probe_data['data'].keys()):
        sim_data[f'Probe {i+1}'] = probe_data['data'][probe_name]['concentration']
    
    # Compute analytical solutions
    print("Computing analytical solutions...")
    analytical_data = compute_analytical_at_probes(
        probe_manager, sim_data['times'], v_x, D_eff, pulse_mass
    )
    
    analytical_snapshots = compute_analytical_snapshots(
        snapshot_manager_2, domain, v_x, D_eff, pulse_mass
    )
    
    # Compute residuals
    residual_data = create_residual_data(sim_data, analytical_data)
    residual_snapshots = create_residual_snapshots(
        snapshot_manager_2.snapshots, analytical_snapshots
    )
    
    plotter = ResultsPlotter(
        config=config,
        mesh=domain.mesh,
        probe_manager=probe_manager,
        snapshot_manager=snapshot_manager,
        domain=domain
    )
    
    custom_pages = {
    'comparison': {
        'type': 'timeseries_comparison',
        'sim_data': sim_data,
        'analytical_data': analytical_data,
        'title': 'VLSPI Verification: Simulated vs Analytical',
        'ylabel': 'Concentration (kg/m³)'
    },
    'residuals': {
        'type': 'timeseries',
        'data': residual_data,
        'title': 'Residuals (Simulated - Analytical)',
        'ylabel': 'Residual (kg/m³)'
    },
    'sim_snapshots': {
        'type': 'snapshots',
        'data': snapshot_manager.snapshots,
        'title': 'Simulated Concentration Field'
    },
    'ana_snapshots': {
        'type': 'snapshots',
        'data': analytical_snapshots,
        'title': 'Analytical Concentration Field'
    },
    'residual_snapshots': {
        'type': 'snapshots',
        'data': residual_snapshots,
        'title': 'Residual Field (Simulated - Analytical)'
    }
    }

    # Generate report
    report = SimulationReport(output_dir=config.output_dir)
    report.print_custom_report(
        config=config,
        domain=domain,
        plotter=plotter,
        custom_pages=custom_pages,
        filename="vlspi_verification.pdf"
    )
    
    # Error statistics
    print("\n=== Error Analysis ===")
    for probe_name in [k for k in analytical_data.keys() if k != 'times']:
        sim_vals = np.array(sim_data[probe_name][1:])
        ana_vals = np.array(analytical_data[probe_name][1:])
        
        residuals = sim_vals - ana_vals
        rmse = np.sqrt(np.mean(residuals**2))
        max_error = np.max(np.abs(residuals))
        rel_error = rmse / (np.max(ana_vals) + 1e-10) * 100
        
        print(f"{probe_name}:")
        print(f"  RMSE: {rmse:.2e} kg/m³ ({rel_error:.2f}%)")
        print(f"  Max Error: {max_error:.2e} kg/m³")
    
    print("\n✓ Verification complete!")
    print(f"✓ Report: {config.output_dir}/vlspi_verification.pdf")


if __name__ == "__main__":
    main()