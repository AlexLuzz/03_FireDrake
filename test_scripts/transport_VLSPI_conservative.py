"""
Script to verify transport solver and evaluate errors, using analytical solutions from Sethi di Molfetta 2019.

Test 1: Vertical line source, Pulse Input (VLSPI) in 2D
"""
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

    