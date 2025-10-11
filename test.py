"""
Simplified Richards' Equation Model in Firedrake
Simulates groundwater flow with rain event boundary condition

Uses van Genuchten model with proper formulation:
- θ: Water content (moisture content)
- Se: Effective saturation
- Cm: Specific moisture capacity (∂θ/∂Hp)
- kr: Relative permeability

Richards equation: Cm * ∂Hp/∂t - ∇·(kr * Ks * ∇Hp) = 0

Where:
- Hp: Pressure head (m)
- Cm: Storage function describing change in θ
- kr: Increases with moisture, from nominal to 1 at saturation
- Ks: Saturated hydraulic conductivity
"""

from firedrake import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from datetime import datetime

# ============================================
# 1. MESH AND FUNCTION SPACES
# ============================================
# Create a 2D rectangular mesh (10m x 5m domain)
# Use finer mesh for better resolution
nx, ny = 60, 30  # Doubled resolution
Lx, Ly = 15.0, 5.0
mesh = RectangleMesh(nx, ny, Lx, Ly)

# Function space for pressure head
V = FunctionSpace(mesh, "CG", 1)

# ============================================
# 2. PARAMETERS
# ============================================
# van Genuchten parameters
theta_r = 0.05  # Residual water content
theta_s = 0.40  # Saturated water content
alpha = 2.0     # van Genuchten alpha (1/m)
n = 2.0         # van Genuchten n
m = 1.0 - 1.0/n # van Genuchten m
l_param = 0.5   # Pore connectivity parameter

# Hydraulic parameters
Ks = Constant(1e-6)  # Saturated hydraulic conductivity (m/s)
g = Constant(9.81)   # Gravity (m/s^2)

# Numerical parameters
epsilon = 0.01  # Smoothing parameter for transition at water table (m)
kr_min = 1e-8   # Minimum relative permeability to avoid singularities

# Time parameters
dt = 30.0  # Time step (seconds)
t_hours = 20
t_end = t_hours * 3600 
num_steps = int(t_end / dt)

# Rain event parameters
rain_start = 10800.0  # Rain starts at 3 hours
rain_end = rain_start + 7200.0    # Rain ends at 5 hours (2 hour duration)
rain_pressure = 0.2   # Additional pressure head during rain (m)

snapshot_times = [0, t_end/10, rain_start + 1800, rain_end + 7200, t_end*0.7, t_end]  # 0h, 2h, 3.5h, 4.5h, 7h, 10h

# Water table level
initial_pressure = Ly / 2.0  # Water table at average height (2.5m)

# ============================================
# 3. VAN GENUCHTEN FUNCTIONS
# ============================================

def calculate_Se(Hp):
    """Calculate effective saturation Se from pressure head with smoothing"""
    if Hp >= epsilon:
        return 1.0
    elif Hp <= -epsilon:
        return 1.0 / (1.0 + abs(alpha * Hp)**n)**m
    else:
        # Smooth transition near Hp = 0
        Se_neg = 1.0 / (1.0 + abs(alpha * (-epsilon))**n)**m
        weight = (Hp + epsilon) / (2.0 * epsilon)
        return Se_neg + (1.0 - Se_neg) * weight

def calculate_theta(Hp):
    """Calculate water content θ from pressure head"""
    Se = calculate_Se(Hp)
    return theta_r + Se * (theta_s - theta_r)

def calculate_Cm(Hp):
    """Calculate specific moisture capacity Cm with smoothing"""
    if Hp >= epsilon:
        return 1e-6  # Small value in saturated zone
    elif Hp <= -epsilon:
        Se = calculate_Se(Hp)
        return (alpha * m) / (1.0 - m) * (theta_s - theta_r) * Se**(1.0/m) * (1.0 - Se**(1.0/m))**m
    else:
        # Smooth transition
        return 1e-6

def calculate_kr(Hp):
    """Calculate relative permeability kr with bounds"""
    if Hp >= epsilon:
        return 1.0
    elif Hp <= -epsilon:
        Se = calculate_Se(Hp)
        kr_val = Se**l_param * (1.0 - (1.0 - Se**(1.0/m))**m)**2
        return max(kr_min, min(kr_val, 1.0))  # Bound between kr_min and 1.0
    else:
        # Smooth transition
        kr_neg = calculate_kr(-epsilon)
        weight = (Hp + epsilon) / (2.0 * epsilon)
        return kr_neg + (1.0 - kr_neg) * weight

def calculate_saturation(pressure_head):
    """
    Calculate saturation from pressure head using van Genuchten model
    Returns normalized saturation for plotting (0 to 1 scale)
    """
    # Create saturation field
    S_field = Function(V)
    
    # Get pressure values
    p_vals = pressure_head.dat.data[:]
    S_vals = np.zeros_like(p_vals)
    
    # Calculate saturation
    for i, Hp in enumerate(p_vals):
        theta = calculate_theta(Hp)
        # Normalize to [0, 1] range for plotting
        S_vals[i] = max(theta_r/theta_s, min(theta / theta_s, 1.0))
    
    S_field.dat.data[:] = S_vals
    return S_field

def calculate_coefficients(p_field):
    """Calculate Cm and kr fields from pressure field with smoothing"""
    Cm_field = Function(V)
    kr_field = Function(V)
    
    p_vals = p_field.dat.data[:]
    Cm_vals = np.zeros_like(p_vals)
    kr_vals = np.zeros_like(p_vals)
    
    for i, Hp in enumerate(p_vals):
        Cm_vals[i] = calculate_Cm(Hp)
        kr_vals[i] = calculate_kr(Hp)
    
    Cm_field.dat.data[:] = Cm_vals
    kr_field.dat.data[:] = kr_vals
    
    return Cm_field, kr_field

# ============================================
# 4. BOUNDARY CONDITIONS
# ============================================
# Boundary markers:
# 1: left (x=0), 2: right (x=Lx), 3: bottom (y=0), 4: top (y=Ly)

water_table_level = initial_pressure  # Ly/2

def create_spatially_varying_top_bc(t, rain_pressure_val, atmospheric_pressure):
    """
    Create a function with different pressures across the top boundary
    Rain region: rain_pressure_val
    No-rain region: atmospheric_pressure
    """
    bc_func = Function(V)
    coords = V.mesh().coordinates.dat.data
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Region of top boundary (x direction) where rain is applied
    x_min_rain = 5.0
    x_max_rain = 10.0

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        # Check if on top boundary (y ≈ Ly)
        if abs(y - Ly) < 1e-6:
            # Apply rain pressure in rain region
            if x_min_rain <= x <= x_max_rain:
                bc_func.dat.data[i] = rain_pressure_val
            else:
                # No rain region: atmospheric pressure
                bc_func.dat.data[i] = atmospheric_pressure
        # For interior nodes, the value doesn't matter (BC only constrains boundary)
    
    return bc_func

# Create hydrostatic pressure boundary condition function
def create_hydrostatic_bc():
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

hydrostatic_bc = create_hydrostatic_bc()

def apply_boundary_conditions(t):
    """Apply boundary conditions based on current time"""
    bcs = []
    
    # Left and right: Fixed water table with hydrostatic pressure
    bc_left = DirichletBC(V, hydrostatic_bc, 1)
    bcs.append(bc_left)
    
    bc_right = DirichletBC(V, hydrostatic_bc, 2)
    bcs.append(bc_right)
    
    # Bottom: Hydrostatic pressure
    bc_bottom = DirichletBC(V, Constant(water_table_level), 3)
    bcs.append(bc_bottom)
    
    # Top boundary: Spatially-varying pressure
    if rain_start <= t <= rain_end:
        # During rain: high pressure in rain region, atmospheric elsewhere
        top_bc_func = create_spatially_varying_top_bc(t, 
                                                      rain_pressure_val=0.5,  # 50cm ponding
                                                      atmospheric_pressure=-0.3)  # Slight suction
        bc_top = DirichletBC(V, top_bc_func, 4)
        bcs.append(bc_top)
    else:
        # After rain: atmospheric pressure everywhere
        top_bc_func = create_spatially_varying_top_bc(t,
                                                      rain_pressure_val=-0.3,
                                                      atmospheric_pressure=-0.3)
        bc_top = DirichletBC(V, top_bc_func, 4)
        bcs.append(bc_top)
    
    return bcs

# ============================================
# 5. VARIATIONAL FORM
# ============================================
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
x_coords = V.mesh().coordinates.dat.data[:, 0]
y_coords = V.mesh().coordinates.dat.data[:, 1]

initial_pressure_array = np.zeros(len(y_coords))
water_table_level = initial_pressure  # Ly/2

for i, y in enumerate(y_coords):
    if y <= water_table_level:
        # Below water table: positive hydrostatic pressure
        initial_pressure_array[i] = water_table_level - y
    else:
        # Above water table: negative pressure (unsaturated)
        # Linear decrease with height
        initial_pressure_array[i] = -(y - water_table_level) * 2.0

p_n.dat.data[:] = initial_pressure_array

# Initialize Cm and kr based on initial pressure
Cm_n, kr_n = calculate_coefficients(p_n)

# Weak form of Richards' equation with van Genuchten:
# Cm * ∂Hp/∂t - ∇·(kr * Ks * ∇Hp) = 0
# Time discretization: backward Euler
# Evaluate Cm and kr at previous time step (Picard iteration)

F = Cm_n * (p - p_n) / dt * q * dx + kr_n * Ks * dot(grad(p), grad(q)) * dx

# Split into bilinear and linear forms
a = lhs(F)
L = rhs(F)

# ============================================
# 6. MONITORING POINTS
# ============================================
# Three points to monitor pressure at different vertical positions
points = [
    (Lx/2, 2.0, "Point at y=2m (below water table)"),
    (Lx/2, 3.0, "Point at y=3m (near water table)"),
    (Lx/2, 4.0, "Point at y=4m (above water table)")
]

# Storage for time series
time_series = {name: [] for _, _, name in points}
time_vals = []

# Storage for spatial snapshots at key times
snapshots = {}

# ============================================
# 7. TIME STEPPING LOOP
# ============================================
print("Starting simulation...")
print(f"Rain event from {rain_start/3600:.1f}h to {rain_end/3600:.1f}h")
print(f"van Genuchten parameters: α={alpha} 1/m, n={n}, m={m:.3f}")
print(f"Water content: θr={theta_r}, θs={theta_s}")
print(f"Mesh resolution: {nx} x {ny} = {nx*ny} elements")

# Solver parameters for better stability
solver_params = {
    'ksp_type': 'gmres',
    'pc_type': 'ilu',
    'ksp_rtol': 1e-8,
    'ksp_atol': 1e-10,
    'ksp_max_it': 1000
}

t = 0.0
for step in range(num_steps):
    t += dt
    
    # Update Cm and kr based on previous pressure solution
    Cm_n, kr_n = calculate_coefficients(p_n)
    
    # Update boundary conditions
    bcs = apply_boundary_conditions(t)

    F = Cm_n * (p - p_n) / dt * q * dx \
    + kr_n * Ks * dot(grad(p), grad(q)) * dx

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

# ============================================
# 8. PLOTTING RESULTS
# ============================================
# Convert to hours for plotting
time_hours = np.array(time_vals) / 3600.0
rain_start_hr = rain_start / 3600.0
rain_end_hr = rain_end / 3600.0

# Create 3x3 grid figure
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# ============================================
# TOP ROW: All time series on one plot (spanning all 3 columns)
# ============================================
colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, green, red
ax_top = fig.add_subplot(gs[0, :])  # Span all columns in top row

for idx, (x, y, name) in enumerate(points):
    ax_top.plot(time_hours, time_series[name], color=colors[idx], 
                linewidth=2.5, label=name, marker='o', markersize=2, 
                markevery=int(len(time_hours)/50))

# Add rain event shading
ax_top.axvspan(rain_start_hr, rain_end_hr, alpha=0.2, color='skyblue', 
               label='Rain event', zorder=0)

# Add water table reference line
ax_top.axhline(y=0, color='cyan', linestyle='--', linewidth=2, 
               label='Water table (p=0)', alpha=0.7)

# Formatting
ax_top.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
ax_top.set_ylabel('Pressure head (m)', fontsize=12, fontweight='bold')
ax_top.set_title('Pressure Response to Rain Event at Different Vertical Positions', 
                 fontsize=14, fontweight='bold', pad=15)
ax_top.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax_top.legend(loc='best', fontsize=10, framealpha=0.9, edgecolor='black')

# Add statistics box
stats_lines = []
for idx, (x, y, name) in enumerate(points):
    peak = max(time_series[name])
    initial = time_series[name][0]
    stats_lines.append(f'{name.split("(")[0].strip()}: Δp = {peak-initial:.3f}m')

stats_text = '\n'.join(stats_lines)
ax_top.text(0.02, 0.98, stats_text, 
            transform=ax_top.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, 
                     edgecolor='black', linewidth=1.5))

# ============================================
# MIDDLE AND BOTTOM ROWS: Saturation snapshots
# ============================================
# Select 6 snapshots evenly distributed
sorted_times = sorted(snapshots.keys())
selected_times = sorted_times[:6] if len(sorted_times) >= 6 else sorted_times

# Pad with last snapshot if needed
while len(selected_times) < 6:
    selected_times.append(selected_times[-1])

# Get coordinate arrays for plotting
coords = V.mesh().coordinates.dat.data
x_coords = coords[:, 0]
y_coords = coords[:, 1]

# Store contour plots for shared colorbar
contour_plots = []

for idx, snap_time in enumerate(selected_times):
    row = 1 + idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    # Get saturation data
    saturation_func = snapshots[snap_time]['saturation']
    sat_values = saturation_func.dat.data[:]
    
    # Clip values to valid range
    sat_values = np.clip(sat_values, 0.0, 1.0)
    
    # Create regular grid for smooth plotting
    xi = np.linspace(0, Lx, 200)
    yi = np.linspace(0, Ly, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate saturation onto regular grid
    coord_points = np.column_stack((x_coords, y_coords))  # (N, 2) array of coordinates
    interp = LinearNDInterpolator(coord_points, sat_values)
    Zi = interp(Xi, Yi)
    
    # Plot saturation field on regular grid
    levels = np.linspace(0.0, 1.0, 25)
    contourf = ax.contourf(Xi, Yi, Zi, levels=levels, 
                           cmap='Blues', extend='neither',
                           vmin=0.0, vmax=1.0)
    
    # Store first contour plot for colorbar
    if idx == 0:
        contour_for_cbar = contourf
    
    # Add contour lines for clarity
    contour_levels = [0.3, 0.5, 0.7, 0.9]
    contour = ax.contour(Xi, Yi, Zi, levels=contour_levels, 
                        colors='gray', alpha=0.3, linewidths=0.8, linestyles='--')
    
    # Add water table line
    ax.axhline(y=water_table_level, color='cyan', linestyle='--', 
               linewidth=2, label='Initial water table', alpha=0.7)
    
    # Mark monitoring points with colors matching time series
    for pt_idx, (x, y, name) in enumerate(points):
        ax.plot(x, y, '*', color=colors[pt_idx], markersize=15, 
                markeredgecolor='black', markeredgewidth=1.0, zorder=10)
    
    # Formatting
    ax.set_xlabel('x (m)', fontsize=10, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=10, fontweight='bold')
    
    time_label = f't = {snap_time/3600:.1f}h'
    if rain_start <= snap_time <= rain_end:
        time_label += ' (RAIN)'
    ax.set_title(time_label, fontsize=11, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    # Add legend only to first plot
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

# Add single colorbar for all saturation plots
cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.56])  # [left, bottom, width, height]
cbar = fig.colorbar(contour_for_cbar, cax=cbar_ax, orientation='vertical')
cbar.set_label('Saturation', fontsize=12, fontweight='bold')

# Set ticks from 0 to 1.0, then invert axis so 0 is at top
cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_ticklabels(['0%\n(dry)', '20%', '40%', '60%', '80%', '100%\n(saturated)'])
cbar.ax.invert_yaxis()  # Invert: 0% at top, 100% at bottom

# Add overall title
fig.suptitle('Richards Equation: 10-Hour Rain Event Simulation\nPressure Response and Saturation Evolution', 
             fontsize=16, fontweight='bold', y=0.995)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
plt.savefig(f'richards_simulation_{now}.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as f'richards_simulation_{now}.png'")

