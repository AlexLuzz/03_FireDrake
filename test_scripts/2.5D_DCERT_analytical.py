#!/usr/bin/env python
"""
2.5D Geoelectrical modeling: div(sigma*grad(u)) - sigma*k^2*u = -I*delta(r-r_s)
Following Dey & Morrison (1979) for mixed boundary conditions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from firedrake import *
from firedrake.pyplot import tricontourf, triplot
from scipy.special import k0

# Parameters
source_A, source_B = (-5.25, -3.75), (5.25, -3.75)
k_val, sigma_val = 1e-2, 1.0

# Create and translate mesh to [-10,10] x [-15,0]
mesh = RectangleMesh(100, 75, 20.0, 15.0)
X = mesh.coordinates
with X.dat.vec as v:
    arr = v.array.reshape((-1, 2))
    arr[:, 0] -= 10.0
    arr[:, 1] -= 15.0

# Adaptive mesh refinement near sources
# Why: Point sources create sharp singularities requiring fine resolution
# How: Apply radial displacement inversely proportional to distance
coords = mesh.coordinates.dat.data_ro
x_old, y_old = coords[:, 0].copy(), coords[:, 1].copy()
disp_x, disp_y = np.zeros_like(x_old), np.zeros_like(y_old)

for sx, sy in [source_A, source_B]:
    rx, ry = x_old - sx, y_old - sy
    r = np.sqrt(rx**2 + ry**2) + 1e-8
    w = 1.0 / (1.0 + (r / 1.0)**3)  # Weight: strong near source, weak far away
    disp_x -= 0.8 * w * rx  # Pull nodes toward source
    disp_y -= 0.8 * w * ry

X = mesh.coordinates
with X.dat.vec as v:
    arr = v.array.reshape((-1, 2))
    arr[:, 0], arr[:, 1] = x_old + disp_x, y_old + disp_y

# Function space (P2 = quadratic elements for better accuracy)
V = FunctionSpace(mesh, "CG", 2)

# Analytical solution following Dey & Morrison (1979)
# Uses Bessel K0 function with mirror source at surface (y=0)
def analytical_solution(V, src_A, src_B, k, sigma):
    v_coords = Function(VectorFunctionSpace(mesh, "CG", V.ufl_element().degree()))
    v_coords.interpolate(SpatialCoordinate(mesh))
    coords = v_coords.dat.data
    
    def potential(src):
        # r_pos: distance to actual source
        r_pos = np.maximum(np.sqrt((coords[:, 0] - src[0])**2 + (coords[:, 1] - src[1])**2), 1e-12)
        # r_neg: distance to mirror source (reflected across y=0)
        r_neg = np.maximum(np.sqrt((coords[:, 0] - src[0])**2 + (coords[:, 1] + src[1])**2), 1e-12)
        # Potential: u = 1/(2π·σ) · [K₀(r_pos·k) + K₀(r_neg·k)]
        return (1.0 / (2.0 * np.pi * sigma)) * (k0(r_pos * k) + k0(r_neg * k))
    
    u_ana = Function(V)
    u_ana.dat.data[:] = potential(src_A) - potential(src_B)
    return u_ana

u_exact = analytical_solution(V, source_A, source_B, k_val, sigma_val)

# Numerical solution setup
u, v = TrialFunction(V), TestFunction(V)

# Weak form: ∫ σ∇u·∇v dx + ∫ σk²uv dx = ∫ I·δ(r-r_s)·v dx
# Left side becomes matrix A, right side becomes vector b
a = sigma_val * inner(grad(u), grad(v)) * dx + sigma_val * k_val**2 * u * v * dx
L = Constant(0) * v * dx  # Will add point sources manually

# Dirichlet BC on boundaries (using analytical solution)
bc = DirichletBC(V, u_exact, "on_boundary")
A, b = assemble(a, bcs=bc), assemble(L)

# Point source injection: find DOFs (degrees of freedom) nearest to sources
# DOFs are the mesh nodes where the solution is computed
# For P2 elements: vertices + edge midpoints
v_coords = Function(VectorFunctionSpace(mesh, "CG", V.ufl_element().degree()))
v_coords.interpolate(SpatialCoordinate(mesh))
coords = v_coords.dat.data

def find_nearest_dof(target, all_coords):
    return np.argmin(np.sum((all_coords - target)**2, axis=1))

idx_A, idx_B = find_nearest_dof(source_A, coords), find_nearest_dof(source_B, coords)

# Inject current: +1A at source A, -1A at source B (dipole configuration)
with b.dat.vec as b_vec:
    b_vec.setValue(idx_A, 1.0, addv=True)
    b_vec.setValue(idx_B, -1.0, addv=True)

# Solve linear system A·u = b using direct LU factorization
u_num = Function(V)
solve(A, u_num, b, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Potential field
cnt0 = tricontourf(u_num, axes=axes[0], levels=20, cmap='RdBu_r')
axes[0].set_title("Potential $u$ [V]")
plt.colorbar(cnt0, ax=axes[0], label='Potential [V]')

# Error field (log scale)
error = Function(V).interpolate(abs(u_num - u_exact))
error_log = Function(V)
error_log.dat.data[:] = np.log10(np.clip(error.dat.data, 1e-12, None))

levels_log = np.arange(-7, 0, 1)
cnt1 = tricontourf(error_log, levels=levels_log, extend='both', cmap='Reds', axes=axes[1])
triplot(mesh, axes=axes[1], interior_kw={'edgecolor': 'black', 'linewidth': 0.3, 'alpha': 0.3})
axes[1].set_title("Error $|u_{exact} - u_{num}|$ [V]")

cbar = plt.colorbar(cnt1, ax=axes[1], ticks=levels_log)
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"$10^{{{int(x)}}}$"))
cbar.set_label('Error [V]')

for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-15, 0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

plt.tight_layout()
plt.savefig('firedrake_geoelectric.png', dpi=150, bbox_inches='tight')

# Error norms
L2_error = sqrt(assemble(inner(u_num - u_exact, u_num - u_exact) * dx))
H1_error = sqrt(assemble(inner(grad(u_num - u_exact), grad(u_num - u_exact)) * dx))
print(f"L2 error: {L2_error:.6e}")
print(f"H1 error: {H1_error:.6e}")
