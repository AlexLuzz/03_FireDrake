"""
Script to verify transport solver and evaluate errors, using analytical solutions from Sethi di Molfetta 2019.

Test 1: Vertical line source, Pulse Input (VLSPI) in 2D
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *
from firedrake import FunctionSpace, Function, interpolate, SpatialCoordinate, exp, conditional, sqrt, Constant


class NoTransportSource:
    """Dummy transport source that returns zero flux"""
    def get_flux_expression(self, t, mesh):
        return Constant(0.0)


# Analytical solution parameters (Sethi di Molfetta 2019)
m = 1.0  # mass of solute (kg)
b = 1.0  # thickness of the aquifer (m)  
n_e = 0.3  # effective porosity (-)
D_xx = 0.1  # longitudinal dispersion coefficient (m2/day)
D_yy = 0.01  # transverse dispersion coefficient (m2/day)
v_x = 1.0  # pore water velocity in x direction (m/day)
v_y = 0.0  # pore water velocity in y direction (m/day)

# Simulation parameters
Lx = 20.0  # domain length in x (m)
Ly = 5.0   # domain length in y (m)
nx = 80    # number of grid points in x
ny = 40    # number of grid points in y
t_end_hours = 2.0*24  # simulation end time (hours)
dt = 1.0   # time step (hours)

# Convert units (day -> second)
D_xx_si = D_xx / (24 * 3600)  # m²/s
D_yy_si = D_yy / (24 * 3600)  # m²/s 
v_x_si = v_x / (24 * 3600)    # m/s
v_y_si = v_y / (24 * 3600)    # m/s

def C_analytical(x, y, t_days):
    """Analytical solution for VLSPI - Pulse Input from Sethi di Molfetta 2019"""
    if t_days <= 0:
        return 0.0
    
    # Source at domain center
    x_source = Lx / 2.0
    y_source = Ly / 2.0
    
    # Displacement from source
    dx = x - x_source - v_x * t_days
    dy = y - y_source - v_y * t_days
    
    left_part = m / (4 * np.pi * b * n_e * t_days * np.sqrt(D_xx * D_yy))
    right_part = np.exp(-(dx**2) / (4 * D_xx * t_days) - (dy**2) / (4 * D_yy * t_days))
    
    return left_part * right_part


def set_initial_pulse(transport_solver, source_mass=1.0, sigma_x=0.1, sigma_y=0.1):
    """Set initial pulse condition as a Gaussian centered at domain center"""
    x, y = SpatialCoordinate(transport_solver.mesh)
    x_center = Lx / 2.0
    y_center = Ly / 2.0
    
    # Gaussian pulse initial condition
    pulse_expr = (source_mass / (2 * np.pi * sigma_x * sigma_y)) * \
                 exp(-((x - x_center)**2) / (2 * sigma_x**2) - 
                     ((y - y_center)**2) / (2 * sigma_y**2))
    
    # Set initial condition
    transport_solver.c_n.interpolate(pulse_expr)
    transport_solver.c_new.assign(transport_solver.c_n)


def compute_analytical_at_probes(probe_positions, times_hours):
    """Compute analytical solution at probe positions and times"""
    analytical_data = {}
    
    for i, (x_probe, y_probe) in enumerate(probe_positions):
        probe_name = f"Probe_{i+1}"
        analytical_data[probe_name] = []
        
        for t_hour in times_hours:
            t_days = t_hour / 24.0
            if t_days > 0:
                c_analytical = C_analytical(x_probe, y_probe, t_days)
            else:
                c_analytical = 0.0
            analytical_data[probe_name].append(c_analytical)
    
    return analytical_data


def compute_analytical_snapshots(snapshot_times, domain):
    """Compute analytical solution snapshots on the domain mesh"""
    analytical_snapshots = {}
    coords = domain.mesh.coordinates.dat.data
    
    for t_seconds in snapshot_times:
        t_days = t_seconds / (24 * 3600)
        if t_days > 0:
            # Compute analytical solution at all mesh points
            c_analytical = np.zeros(len(coords))
            for i, (x, y) in enumerate(coords):
                c_analytical[i] = C_analytical(x, y, t_days)
            
            # Create function for this snapshot
            V = FunctionSpace(domain.mesh, "CG", 1)
            c_func = Function(V, name=f"Analytical_t={t_seconds:.0f}s")
            c_func.dat.data[:] = c_analytical
            analytical_snapshots[t_seconds] = {'concentration': c_func}
        
    return analytical_snapshots


def create_residual_comparison_plot(probe_data, analytical_data, times_hours, output_dir):
    """Create residual time series plot comparing simulated vs analytical"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(probe_data['data'])))
    
    # Top plot: Time series comparison
    for i, (probe_name, sim_data) in enumerate(probe_data['data'].items()):
        sim_conc = sim_data['concentration']
        if f"Probe_{i+1}" in analytical_data:
            analytical_conc = analytical_data[f"Probe_{i+1}"]
            
            ax1.plot(times_hours, sim_conc, 'o-', color=colors[i], 
                    label=f'{probe_name} - Firedrake', linewidth=2, markersize=4)
            ax1.plot(times_hours, analytical_conc, '--', color=colors[i], 
                    label=f'{probe_name} - Analytical', linewidth=2)
    
    ax1.set_xlabel('Time (hours)', fontweight='bold')
    ax1.set_ylabel('Concentration (kg/m³)', fontweight='bold')
    ax1.set_title('VLSPI Conservative Transport: Firedrake vs Analytical', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, max(times_hours))
    
    # Bottom plot: Residuals
    for i, (probe_name, sim_data) in enumerate(probe_data['data'].items()):
        sim_conc = np.array(sim_data['concentration'])
        if f"Probe_{i+1}" in analytical_data:
            analytical_conc = np.array(analytical_data[f"Probe_{i+1}"])
            residuals = sim_conc - analytical_conc
            
            ax2.plot(times_hours, residuals, 'o-', color=colors[i], 
                    label=f'{probe_name} Residual', linewidth=2, markersize=4)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Time (hours)', fontweight='bold')
    ax2.set_ylabel('Residual (kg/m³)', fontweight='bold')
    ax2.set_title('Residuals: Firedrake - Analytical', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, max(times_hours))
    
    plt.tight_layout()
    filename = output_dir / 'VLSPI_residual_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Residual comparison saved: {filename}")
    plt.close()
    
    return fig


def create_snapshot_comparison_plots(sim_snapshots, analytical_snapshots, domain, output_dir):
    """Create snapshot comparison plots (simulated vs analytical)"""
    coords = domain.mesh.coordinates.dat.data
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    
    # Select 2 representative times for snapshot comparison
    times = sorted(list(sim_snapshots.keys()))
    selected_times = [times[len(times)//3], times[2*len(times)//3]]  # 1/3 and 2/3 through simulation
    
    for t_snapshot in selected_times:
        if t_snapshot in analytical_snapshots:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Get data
            sim_data = sim_snapshots[t_snapshot]['concentration'].dat.data
            analytical_data = analytical_snapshots[t_snapshot]['concentration'].dat.data
            residual_data = sim_data - analytical_data
            
            # Compute global color limits
            all_data = np.concatenate([sim_data, analytical_data])
            vmin = max(0, np.percentile(all_data[all_data > 0], 1))
            vmax = np.percentile(all_data, 99)
            
            # Plot simulated
            sc1 = axes[0].scatter(x_coords, y_coords, c=sim_data, s=10, cmap='Spectral_r', 
                                vmin=vmin, vmax=vmax)
            axes[0].set_title(f'Firedrake (t={t_snapshot/3600:.1f}h)', fontweight='bold')
            axes[0].set_xlabel('x (m)')
            axes[0].set_ylabel('y (m)')
            axes[0].set_aspect('equal')
            plt.colorbar(sc1, ax=axes[0], label='Concentration (kg/m³)')
            
            # Plot analytical
            sc2 = axes[1].scatter(x_coords, y_coords, c=analytical_data, s=10, cmap='Spectral_r',
                                vmin=vmin, vmax=vmax)
            axes[1].set_title(f'Analytical (t={t_snapshot/3600:.1f}h)', fontweight='bold')
            axes[1].set_xlabel('x (m)')
            axes[1].set_ylabel('y (m)')
            axes[1].set_aspect('equal')
            plt.colorbar(sc2, ax=axes[1], label='Concentration (kg/m³)')
            
            # Plot residuals
            res_max = np.max(np.abs(residual_data))
            sc3 = axes[2].scatter(x_coords, y_coords, c=residual_data, s=10, cmap='RdBu_r',
                                vmin=-res_max, vmax=res_max)
            axes[2].set_title(f'Residual (t={t_snapshot/3600:.1f}h)', fontweight='bold')
            axes[2].set_xlabel('x (m)')
            axes[2].set_ylabel('y (m)')
            axes[2].set_aspect('equal')
            plt.colorbar(sc3, ax=axes[2], label='Residual (kg/m³)')
            
            plt.tight_layout()
            filename = output_dir / f'VLSPI_snapshot_comparison_t{t_snapshot:.0f}s.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Snapshot comparison saved: {filename}")
            plt.close()


def run_firedrake_simulation():
    """Run the VLSPI conservative transport simulation"""
    
    # ==========================================
    # 1. SIMULATION CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        name="VLSPI_Conservative",
        t_end_hours=t_end_hours,
        dt=dt,
        output_dir="results/transport_verification"
    )
    
    # ==========================================
    # 2. DOMAIN SETUP
    # ==========================================
    domain = Domain.homogeneous(till_curve_RAF(), Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    V = FunctionSpace(domain.mesh, "CG", 1)
    
    # ==========================================
    # 3. MATERIAL FIELD (minimal for transport)
    # ==========================================
    field_map = MaterialField(domain, V)
    
    # ==========================================
    # 4. BOUNDARY CONDITIONS (for pressure only)
    # ==========================================
    bc_manager = BoundaryConditionManager(V, left_wt=3.0, right_wt=3.0)
    
    # ==========================================
    # 5. SOLVERS
    # ==========================================
    richards_solver = RichardsSolver(
        domain=domain, V=V, field_map=field_map,
        source_scenario=None, bc_manager=bc_manager, config=config
    )
    
    transport_solver = TransportSolver(
        domain=domain, V=V, field_map=field_map,
        pressure_solver=richards_solver,
        bc_manager=None,  # No transport BCs
        transport_source=NoTransportSource(),  # No source term
        config=config
    )
    
    # ==========================================
    # 6. INITIAL CONDITIONS
    # ==========================================
    set_initial_pulse(transport_solver, source_mass=m, sigma_x=0.2, sigma_y=0.2)
    
    # ==========================================
    # 7. MONITORING SETUP
    # ==========================================
    probe_positions = [
        [Lx/2.0, Ly/2.0],        # Center (source location)
        [Lx/2.0 + 2.0, Ly/2.0],  # 2m downstream
        [Lx/2.0 + 4.0, Ly/2.0],  # 4m downstream  
        [Lx/2.0, Ly/2.0 + 1.0],  # 1m transverse
    ]
    probe_names = ["Center", "2m_downstream", "4m_downstream", "1m_transverse"]
    
    probe_manager = ProbeManager(
        domain.mesh,
        probe_positions=probe_positions,
        names=probe_names
    )
    
    # Snapshot times for spatial comparison
    snapshot_times = np.linspace(config.dt, config.t_end, 8).tolist()
    snapshot_manager = SnapshotManager(snapshot_times)
    
    # ==========================================
    # 8. TRANSPORT PARAMETERS (prescribed)
    # ==========================================
    transport_params = {
        'vx': v_x_si,    # m/s
        'vy': v_y_si,    # m/s
        'Dxx': D_xx_si,  # m²/s
        'Dyy': D_yy_si,  # m²/s
        'Dxy': 0.0,      # m²/s (no cross-dispersion)
        'R': 1.0,        # No retardation
        'lambda': 0.0    # No decay
    }
    
    # ==========================================
    # 9. RUN SIMULATION
    # ==========================================
    print("Running VLSPI Conservative Transport Verification...")
    transport_solver.run(probe_manager, snapshot_manager, params=transport_params)
    
    return probe_manager, snapshot_manager, domain, config


def main():
    """Main verification function"""
    print("="*60)
    print("VLSPI CONSERVATIVE TRANSPORT VERIFICATION")
    print("Comparing Firedrake simulation vs Analytical solution")
    print("="*60)
    
    # Run simulation
    probe_manager, snapshot_manager, domain, config = run_firedrake_simulation()
    
    # Get simulation data
    probe_data = probe_manager.get_data()
    times_hours = probe_data['times'] / 3600.0
    sim_snapshots = snapshot_manager.snapshots
    
    # Compute analytical solutions
    print("\nComputing analytical solutions...")
    analytical_probe_data = compute_analytical_at_probes(probe_manager.probe_positions, times_hours)
    analytical_snapshots = compute_analytical_snapshots(snapshot_manager.snapshot_times, domain)
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    create_residual_comparison_plot(probe_data, analytical_probe_data, times_hours, config.output_dir)
    create_snapshot_comparison_plots(sim_snapshots, analytical_snapshots, domain, config.output_dir)
    
    # Compute error metrics
    print("\nComputing error metrics...")
    for i, (probe_name, sim_data) in enumerate(probe_data['data'].items()):
        if f"Probe_{i+1}" in analytical_probe_data:
            sim_conc = np.array(sim_data['concentration'])
            analytical_conc = np.array(analytical_probe_data[f"Probe_{i+1}"])
            
            # Skip initial zero values for meaningful error calculation
            mask = analytical_conc > 1e-10
            if mask.any():
                rmse = np.sqrt(np.mean((sim_conc[mask] - analytical_conc[mask])**2))
                max_analytical = np.max(analytical_conc[mask])
                relative_rmse = rmse / max_analytical * 100
                
                print(f"  {probe_name}: RMSE = {rmse:.2e} kg/m³ ({relative_rmse:.1f}%)")
    
    print(f"\n✓ Verification complete! Results saved to: {config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *
from firedrake import FunctionSpace


m = 10 # mass of solute (kg)
b = 1.0 # thickness of the aquifer (m)
n_e = 0.3 # effective porosity (-)
D_xx = 0.1 # longitudinal dispersion coefficient (m2/day)
D_yy = 0.01 # transverse dispersion coefficient (m2/day)
v_x = 1.0 # pore water velocity in x direction (m/day)
v_y = 0.0 # pore water velocity in y direction (m/day)

# Simulation parameters
Lx = 20.0  # domain length in x (m)
Ly = 5.0   # domain length in y (m)
nx = 100   # number of grid points in x
ny = 50    # number of grid points in y
t_end_hours = 10.0*24  # simulation end time (hours)
dt = 3    # time step (hours)


def C(x, y, t):
    """Analytical solution for VLSPI - Pulse Input from Sethi di Molfetta 2019"""
    
    left_part = m / (4 * np.pi * b * n_e * t * np.sqrt(D_xx * D_yy))
    right_part = np.exp(
                        - (x - v_x * t)**2 / (4 * D_xx * t) + 
                        - (y - v_y * t)**2 / (4 * D_yy * t)
                        )


    return left_part * right_part

def C_max(t):
    """Maximum concentration for VLSPI - Pulse Input from Sethi di Molfetta 2019"""
    return m / (4 * np.pi * b * n_e * t * np.sqrt(D_xx * D_yy) )
    

def run_analytical_simulation():
    """Run the analytical solution for VLSPI conservative transport"""
    
    
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    times = np.arange(0, t_end_hours, dt)

    concentration_profiles = []
    
    for t in times:
        C_t = C(X - Lx/2, Y - Ly/2, t)  # Centered at (Lx/2, Ly/2)
        concentration_profiles.append(C_t)
    
    return times, concentration_profiles





def run_firedrake_simulation():
    """Run the VLSPI conservative transport simulation"""
    
    config = SimulationConfig(
        name="VLSPI_Conservative",
        t_end_hours=t_end_hours,  # hours
        dt=dt,  # hours
    )
    
    domain = Domain(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
    domain.assign_all("base", till_curve_RAF())
    
    V = FunctionSpace(domain.mesh, "CG", 1)
    
    field_map = MaterialField(domain, V)
    
    bc_manager = BoundaryConditionManager(
        V,
        left_wt=3.0,
        right_wt=3.0
    )
    
    richard_solver = RichardsSolver(
        domain=domain,
        V=V,
        field_map=field_map,
        source_scenario=None,  # No flow source
        bc_manager=bc_manager,
        config=config
    )

    transport_solver = TransportSolver(
        domain=domain,
        V=V,
        field_map=field_map,
        pressure_solver=richard_solver,  # No flow coupling for this test
        bc_manager=None,  # No transport BCs for this test
        transport_source=None,  # No source term for this test
        config=config,
        debug=False
    )
    
    # ==========================================
    # 6. MONITORING SETUP
    # ==========================================
    probe_positions = [
        [10.0, 2.5],  # Center of the domain
    ]
    probe_names = [
        "Center Probe",
    ]
    
    probe_manager = ProbeManager(
        domain.mesh,
        probe_positions=probe_positions,
        names=probe_names
    )
    
    snapshot_times = np.linspace(0.1, config.t_end, 10).tolist()
    snapshot_manager = SnapshotManager(snapshot_times)
    
    # ==========================================
    # 7. RUN SIMULATION
    # ==========================================
    transport_solver.run(probe_manager, snapshot_manager)
    
    return probe_manager, snapshot_manager


if __name__ == "__main__":
    #main_VLSPI_conservative()
    print("This is a test script. Please run the tests in the test suite to verify functionality.")

    