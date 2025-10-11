from firedrake import *
import numpy as np
from physics import *
from tools import *
from visualization import * 

# ============================================
# 1. SIMULATION PARAMETERS
# ============================================
phys_p = {
    # van Genuchten parameters
    'alpha': 2.0,  # van Genuchten alpha (1/m)
    'n': 2.0,      # van Genudchten n
    'm': 1.0 - 1.0/2.0,  # van Genuchten m
    'theta_r': 0.05,  # Residual water content
    'theta_s': 0.40,   # Saturated water content
    'l_param': 0.5,    # Pore connectivity parameter

    # Hydraulic parameters
    'Ks': Constant(1e-6),  # Saturated hydraulic conductivity (m/s)
    'g': Constant(9.81),   # Gravity (m/s^2)

    # Numerical parameters
    'epsilon': 0.01,  # Smoothing parameter for transition at water table (m)
    'kr_min': 1e-7    # Minimum relative permeability to avoid singularities
}

sim_p = {
    # Simulation parameters
    # Time parameters
    'dt': 30.0,  # Time step (seconds)
    't_end': 13,

    # Rain event parameters
    'start': 3.0,  # Rain starts at 3 hours
    'end': 4.0,  # Rain ends at 4 hours (1 hour duration)
    'pressure': 0.2,    # Additional pressure head during rain (m)

    'snapshot_times':[0, 2, 3.5, 6, 8, 13]
}

# ============================================
# 2. DEFINE EQUATIONS
# ============================================

# Mesh parameters
Lx = 10.0
Ly = 5.0
nx = 80
ny = 40

mesh = RectangleMesh(nx, ny, Lx, Ly)
V = FunctionSpace(mesh, "CG", 1)

# ============================================
# 3. MONITORING POINTS
# ============================================
# Three points to monitor pressure at different vertical positions
points = [
    (Lx/2, 2.0, "Point at y=2m (below water table)"),
    (Lx/2, 3.0, "Point at y=3m (near water table)"),
    (Lx/2, 4.0, "Point at y=4m (above water table)")
]

# ============================================
# 4. RUN SIMULATION
# ============================================
results = run_simple_rain_simulation(sim_p, phys_p, V, mesh, points)

# ============================================
# 5. PLOT AND SAVE RESULTS
# ============================================
plot_results(results, points, V, sim_p)
