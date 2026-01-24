import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from scipy.special import k0
from firedrake.pyplot import tricontourf, tripcolor
import matplotlib.ticker as ticker

# 1. Configuration
source_A = (-5.25, -3.75)
source_B = (5.25, -3.75)
k_val = 1e-2
sigma_val = 1.0

mesh = RectangleMesh(100, 75, 20.0, 15.0)
mesh.coordinates.dat.data[:, 0] -= 10.0
mesh.coordinates.dat.data[:, 1] -= 15.0

# 2. Raffinement par déformation (Warping)
# On accède directement aux coordonnées pour les manipuler avec NumPy
import numpy as np
coords = mesh.coordinates.dat.data
x_old = coords[:, 0].copy()
y_old = coords[:, 1].copy()

# --- Raffinement vertical (vers la surface y=0) ---
# On utilise une fonction de puissance : plus l'exposant est élevé, 
# plus les mailles sont écrasées vers le haut.
depth_norm = np.abs(y_old) / 15.0  # de 1 (fond) à 0 (surface)
coords[:, 1] = -15.0 * (depth_norm**2.0)  # L'exposant 2.0 densifie vers 0

# --- Raffinement horizontal (vers le centre x=0) ---
# On densifie vers le centre pour mieux capturer les deux électrodes
x_norm = x_old / 10.0  # de -1 à 1
coords[:, 0] = 10.0 * np.sign(x_norm) * (np.abs(x_norm)**1.5)

V = FunctionSpace(mesh, "CG", 2)

# 2. Solution Analytique (pour les BC et la comparaison)
def get_analytical_solution(V, src_A, src_B, k, sigma):
    u_ana = Function(V)
    # Récupération des coordonnées des DOFs (nœuds de calcul)
    v_coords = Function(VectorFunctionSpace(mesh, "CG", V.ufl_element().degree()))
    v_coords.interpolate(SpatialCoordinate(mesh))
    coords = v_coords.dat.data
    
    def potential(src, coords):
        r1 = np.sqrt((coords[:, 0] - src[0])**2 + (coords[:, 1] - src[1])**2)
        r2 = np.sqrt((coords[:, 0] - src[0])**2 + (coords[:, 1] + src[1])**2) # Miroir
        r1 = np.maximum(r1, 1e-10)
        r2 = np.maximum(r2, 1e-10)
        return (1.0 / (2.0 * np.pi * sigma)) * (k0(r1 * k) + k0(r2 * k))

    u_ana.dat.data[:] = potential(src_A, coords) - potential(src_B, coords)
    return u_ana

u_exact = get_analytical_solution(V, source_A, source_B, k_val, sigma_val)

# 3. Résolution Numérique "Manuelle"
u = TrialFunction(V)
v = TestFunction(V)
a = sigma_val * inner(grad(u), grad(v)) * dx + sigma_val * k_val**2 * u * v * dx
L = Constant(0) * v * dx

# Condition de Dirichlet sur les bords (Indispensable pour comparer à l'analytique)
bc = DirichletBC(V, u_exact, "on_boundary")

# Assemblage du système
A = assemble(a, bcs=bc)
b = assemble(L)

# INJECTION MANUELLE : On trouve les indices des nœuds les plus proches
v_coords = Function(VectorFunctionSpace(mesh, "CG", V.ufl_element().degree()))
v_coords.interpolate(SpatialCoordinate(mesh))
coords = v_coords.dat.data

def find_nearest_dof(target_coord, all_coords):
    dists = np.sum((all_coords - target_coord)**2, axis=1)
    return np.argmin(dists)

idx_A = find_nearest_dof(source_A, coords)
idx_B = find_nearest_dof(source_B, coords)

# On modifie le vecteur b directement (I=1A en A, I=-1A en B)
with b.dat.vec as b_vec:
    b_vec.setValue(idx_A, 1.0, addv=True)
    b_vec.setValue(idx_B, -1.0, addv=True)

u_num = Function(V)
solve(A, u_num, b, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

# 4. Affichage
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Plot du Potentiel (Linéaire)
cnt0 = tricontourf(u_num, axes=axes[0], levels=20, cmap='RdBu_r')
axes[0].set_title("Potentiel Numérique $u$ [V]")
plt.colorbar(cnt0, ax=axes[0], label='Potentiel [V]')

# 2. Plot de l'Erreur (LogNorm en Volts)
error_field = Function(V).interpolate(abs(u_num - u_exact))
# On récupère les données et on applique un "clip"
err_data = np.clip(error_field.dat.data, 1e-10, None)
# On crée un champ avec le log10 des valeurs
error_log = Function(V)
error_log.dat.data[:] = np.log10(err_data)

# 2. Définition des niveaux log (ex: de -7 à -1)
levels_log = np.arange(-7, 0, 1) # Paliers : -7, -6, -5, -4, -3, -2, -1

# 3. On plot le champ LOG avec des niveaux LINÉAIRES
cnt1 = tricontourf(error_log, 
                  levels=levels_log, 
                  extend='both', 
                  cmap='Reds', 
                  axes=axes[1])

axes[1].set_title("Erreur $|u_{exact} - u_{num}|$ [V]")
cbar = plt.colorbar(cnt1, ax=axes[1], ticks=levels_log)
def log_formatter(x, pos):
    return f"$10^{{{int(x)}}}$"
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))
cbar.set_label('Erreur $|u_{exact} - u_{num}|$ [V]')

for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-15, 0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

plt.tight_layout()
plt.savefig('firedrake_geoelectricV2.png', dpi=150, bbox_inches='tight')