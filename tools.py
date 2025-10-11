from firedrake import *
import numpy as np
from physics import *

def create_hydrostatic_bc(water_table_level, V):
    """Create a function with hydrostatic pressure profile"""
    hydro_func = Function(V)
    coords = V.mesh().coordinates.dat.data
    y_coords = coords[:, 1]
    
    for i, y in enumerate(y_coords):
        if y <= water_table_level:
            # Below water table: positive pressure (hydrostatic)
            hydro_func.dat.data[i] = water_table_level - y
        else:
            # Above water table: negative pressure (suction)
            hydro_func.dat.data[i] = -(y - water_table_level) * 2.0
    
    return hydro_func

def apply_boundary_conditions(water_table_level, V, rain_pressure=None):
    """Apply boundary conditions based on current time"""
    bcs = []
    hydrostatic_bc = create_hydrostatic_bc(water_table_level, V)
    # Left and right boundaries: Fixed water table with hydrostatic pressure
    bc_left = DirichletBC(V, hydrostatic_bc, 1)
    bcs.append(bc_left)
    
    bc_right = DirichletBC(V, hydrostatic_bc, 2)
    bcs.append(bc_right)
    
    # Bottom boundary: Apply hydrostatic pressure to avoid artifacts
    # At y=0, pressure should be water_table_level - 0 = water_table_level
    bc_bottom = DirichletBC(V, Constant(water_table_level), 3)
    bcs.append(bc_bottom)
    
    # Top boundary (y=Ly): Rain event (time-dependent)
    if rain_pressure is not None:
        # During rain: apply positive pressure head 
        bc_top = DirichletBC(V, Constant(rain_pressure), 4)
        bcs.append(bc_top)

    return bcs

def run_simple_rain_simulation(sim_params, phys_p, V, mesh, points=None):
    
    if points is not None:
        time_series = {name: [] for _, _, name in points}

    # Storage for spatial snapshots and times
    snapshots = {}
    time_vals = []
    
    print("Starting rain simulation...")
    print(f"Rain event from {sim_params['start']:.1f}h to {sim_params['end']:.1f}h")

    # Solver parameters for better stability
    solver_params = {
        'ksp_type': 'gmres',
        'pc_type': 'ilu',
        'ksp_rtol': 1e-8,
        'ksp_atol': 1e-10,
        'ksp_max_it': 1000
    }

    dt = sim_params['dt']
    t_end = sim_params['t_end']*3600
    num_steps = int(t_end / dt)
    snapshot_times = sim_params['snapshot_times']

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Solution functions
    p_n = Function(V, name="Pressure_old")  # Previous time step
    p_new = Function(V, name="Pressure")    # Current time step

    # Coefficient fields (will be updated each time step)
    Cm_n = Function(V, name="Moisture_capacity")
    kr_n = Function(V, name="Relative_permeability")

    # Set initial condition with proper hydrostatic pressure
    # At water table (y = Ly/2): pressure = 0 (saturated)
    # Below water table: pressure = (water_table - y) (positive, hydrostatic)
    # Above water table: pressure decreases with height (negative, suction)
    y_coords = V.mesh().coordinates.dat.data[:, 1]

    initial_pressure_array = np.zeros(len(y_coords))
    water_table_level = y_coords.max()/2  # Ly/2

    for i, y in enumerate(y_coords):
        if y <= water_table_level:
            # Below water table: positive hydrostatic pressure
            initial_pressure_array[i] = water_table_level - y
        else:
            # Above water table: negative pressure (unsaturated)
            # Linear decrease with height
            initial_pressure_array[i] = -(y - water_table_level) * 2.0

    p_n.dat.data[:] = initial_pressure_array

    t = 0.0
    for step in range(num_steps):
        t += dt
        
        # Update Cm and kr based on previous pressure solution
        Cm_n, kr_n = calculate_coefficients(V, p_n)

        # Update boundary conditions
        bcs = apply_boundary_conditions(t)
        
        # Solve the system (reassemble with new coefficients)
        F = Cm_n * (p - p_n) / dt * q * dx + kr_n * p['Ks'] * dot(grad(p), grad(q)) * dx
        a = lhs(F)
        L = rhs(F)
        
        solve(a == L, p_new, bcs=bcs, solver_parameters=solver_params)
        
        # Store pressure at monitoring points
        time_vals.append(t)
        for x, y, name in points:
            pressure_val = p_new.at([x, y])
            time_series[name].append(pressure_val)
        
        # Save snapshots at specific times
        if any(abs(t - st) < dt/2 for st in snapshot_times):
            saturation = calculate_saturation(p_new)
            
            # Diagnostic: check saturation range
            sat_vals_diag = saturation.dat.data[:]
            print(f"  Saved snapshot at t={t/3600:.2f}h")
            print(f"    Saturation range: [{sat_vals_diag.min():.4f}, {sat_vals_diag.max():.4f}]")
            print(f"    Pressure range: [{p_new.dat.data[:].min():.4f}, {p_new.dat.data[:].max():.4f}]")
            
            snapshots[t] = {
                'pressure': p_new.copy(deepcopy=True),
                'saturation': saturation.copy(deepcopy=True)
            }
        
        # Update solution for next time step
        p_n.assign(p_new)
        
        # Print progress every hour
        if step % int(3600/dt) == 0:
            print(f"Time: {t/3600:.1f}h / {t_end/3600:.1f}h")

    print("Simulation complete!")

    results = {
        'time': time_vals,
        'time_series': time_series,
        'snapshots': snapshots
        }

    return results  
    