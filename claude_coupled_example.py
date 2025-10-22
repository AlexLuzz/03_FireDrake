"""
COMPLETE EXAMPLE: Montreal Green Infrastructure with ERT Calibration

Workflow:
1. Define materials (Till + Terreau)
2. Setup WaxmanSmits model for ERT
3. Define rain and chloride sources
4. Run coupled flow-transport simulation
5. Compare with ERT measurements
6. Calibrate uncertain parameters
"""

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# IMPORT YOUR REFACTORED MODULES
# ==========================================
from physics.materials import SoilMaterial
from physics.domain import Domain
from physics.geophysical_properties import WaxmanSmits, WaxmanSmitsParams, ERTCalibrator
from physics.transport_properties import TransportMaterial
from physics.source_terms import (
    HydraulicSource, ChlorideFromRain, RoadSaltSource, CompositeSource
)
from physics.rain_config import RainScenario, RainEvent, RainZone

from solvers.richards_solver import RichardsSolver
from solvers.transport_solver import ChlorideTransport, CoupledFlowTransport
from tools.boundary_conditions import BoundaryConditionManager
from tools.config import SimulationConfig

# ==========================================
# STEP 1: DEFINE GEOMETRY AND MESH
# ==========================================

# Montreal bioswale geometry: 22m x 5m
Lx, Ly = 22.0, 5.0
nx, ny = 110, 25  # 0.2m resolution

mesh = RectangleMesh(nx, ny, Lx, Ly)
V = FunctionSpace(mesh, "CG", 1)

print("=" * 60)
print("MONTREAL GREEN INFRASTRUCTURE SIMULATION")
print("=" * 60)
print(f"Domain: {Lx}m × {Ly}m")
print(f"Mesh: {nx} × {ny} elements")

# ==========================================
# STEP 2: DEFINE MATERIALS
# ==========================================

# Native Montreal till (bottom layer)
till = SoilMaterial.from_curves("Till")

# Engineered growing medium (top layer)
terreau = SoilMaterial.from_curves("Terreau")

print("\nMaterials:")
print(f"  Till: φ={till.porosity:.3f}, Ks={till.Ks:.2e} m/s")
print(f"  Terreau: φ={terreau.porosity:.3f}, Ks={terreau.Ks:.2e} m/s")

# ==========================================
# STEP 3: SETUP WAXMAN-SMITS FOR ERT
# ==========================================

# Initial parameter estimates for Montreal till
# These will be calibrated from your ERT time-series!
ws_params = WaxmanSmitsParams(
    a=1.0,      # Tortuosity (0.5-2.5)
    m=1.8,      # Cementation exponent (lower for clay-rich)
    n=2.0,      # Saturation exponent
    B=4.6e-8,   # Na+ equivalent conductance (standard)
    Qv=0.15     # CEC per pore volume (for ~15% clay)
)

ws_model = WaxmanSmits(ws_params)

print("\nWaxman-Smits parameters (initial):")
print(f"  a={ws_params.a:.2f}, m={ws_params.m:.2f}, n={ws_params.n:.2f}")
print(f"  B×Qv={ws_params.get_clay_conductivity_term():.2e} S/m")

# ==========================================
# STEP 4: CREATE DOMAIN WITH ERT
# ==========================================

# Two-layer system: till (0-2m), terreau (2-5m)
domain = Domain.horizontal_layers(
    mesh,
    layers=[
        (till, 2.0),      # 0-2m: native till
        (terreau, 3.0)    # 2-5m: engineered soil
    ]
)

# Add WaxmanSmits model for ERT predictions
domain.ws_model = ws_model

print("\nDomain configuration:")
print("  Bottom layer (0-2m): Till")
print("  Top layer (2-5m): Terreau")

# ==========================================
# STEP 5: DEFINE RAIN SCENARIO
# ==========================================

# Option A: Load from your CSV file
# rain_scenario = RainScenario.from_csv(
#     "your_rain_data.csv",
#     time_col="hours",
#     rain_col="intensity_mm_hr"
# )

# Option B: Manual winter storm scenario
rain_events = [
    # Morning rain
    RainEvent(
        start_time=0.0, end_time=2.0,
        intensity=15.0,  # mm/hr
        zones=[RainZone(0, 22, 1.0)],  # Uniform across domain
        name="morning_rain"
    ),
    # Evening storm
    RainEvent(
        start_time=18.0, end_time=20.0,
        intensity=25.0,
        zones=[RainZone(0, 22, 1.0)],
        name="evening_storm"
    ),
    # Next morning
    RainEvent(
        start_time=30.0, end_time=32.0,
        intensity=10.0,
        zones=[RainZone(0, 22, 1.0)],
        name="next_morning"
    )
]

rain_scenario = RainScenario(rain_events)
hydraulic_source = HydraulicSource(rain_scenario)

print("\nRain scenario:")
print(f"  Events: {len(rain_events)}")
for event in rain_events:
    print(f"    {event.name}: t={event.start_time}-{event.end_time}h, "
          f"I={event.intensity} mm/hr")

# ==========================================
# STEP 6: DEFINE CHLORIDE SOURCES
# ==========================================

# Background chloride in rain (atmospheric deposition)
cl_from_rain = ChlorideFromRain(
    hydraulic_source,
    rain_chloride_concentration=0.3  # mg/L (typical Montreal)
)

# Road salt applications (adjacent to bioswale)
road_salt = RoadSaltSource(
    road_x_min=9.0,
    road_x_max=13.0,  # 4m road section
    application_rate_g_m2=200.0,  # g NaCl/m² per application
    application_times_hours=[6, 18, 30],  # Morning, evening, next morning
    dissolution_time_hours=2.0
)

# Total chloride source
total_cl_source = CompositeSource([cl_from_rain, road_salt])

print("\nChloride sources:")
print(f"  Background (rain): 0.3 mg/L")
print(f"  Road salt: 200 g/m² at x=9-13m")
print(f"  Application times: 6h, 18h, 30h")

# ==========================================
# STEP 7: DEFINE TRANSPORT PROPERTIES
# ==========================================

# Chloride transport material
cl_transport = TransportMaterial.from_contaminant(
    'chloride',
    alpha_L=0.05,  # 5 cm longitudinal dispersivity
    alpha_T=0.005  # 5 mm transverse dispersivity
)

print("\nTransport properties:")
print(f"  Contaminant: {cl_transport.contaminant.name}")
print(f"  D₀: {cl_transport.contaminant.diffusion_coeff:.2e} m²/s")
print(f"  αₗ: {cl_transport.alpha_L} m")
print(f"  αₜ: {cl_transport.alpha_T} m")

# ==========================================
# STEP 8: SETUP BOUNDARY CONDITIONS
# ==========================================

# Hydraulic boundaries
bc_manager_flow = BoundaryConditionManager(mesh, V)
bc_manager_flow.set_constant_head(bottom=2.0)  # Water table at 2m elevation

# Transport boundaries
bc_manager_transport = BoundaryConditionManager(mesh, V)
bc_manager_transport.set_constant_concentration(bottom=0.0)  # No Cl at depth

# ==========================================
# STEP 9: SIMULATION CONFIGURATION
# ==========================================

config = SimulationConfig(
    Lx=Lx, Ly=Ly, nx=nx, ny=ny,
    t_end=48*3600,  # 48 hours simulation
    dt=1800,        # 30 minute timestep
    solver_parameters={
        'ksp_type': 'gmres',
        'pc_type': 'ilu',
        'ksp_rtol': 1e-8
    }
)

print("\nSimulation setup:")
print(f"  Duration: {config.t_end/3600:.0f} hours")
print(f"  Timestep: {config.dt}s ({config.dt/60:.1f} min)")
print(f"  Total steps: {config.num_steps}")

# ==========================================
# STEP 10: CREATE SOLVERS
# ==========================================

print("\nInitializing solvers...")

richards = RichardsSolver(
    mesh, V, domain, 
    hydraulic_source,  # Uses HydraulicSource
    bc_manager_flow,
    config
)

transport = ChlorideTransport(
    mesh, domain,
    cl_transport,      # Uses TransportMaterial
    total_cl_source,   # Uses SourceTerm
    bc_manager_transport
)

# Set initial conditions
transport.set_initial_condition(0.0)  # No chloride initially

# Coupled solver
coupled = CoupledFlowTransport(richards, transport)

print("  ✓ Richards solver ready")
print("  ✓ Transport solver ready")
print("  ✓ Coupled solver ready")

# ==========================================
# STEP 11: RUN SIMULATION
# ==========================================

print("\n" + "=" * 60)
print("RUNNING SIMULATION")
print("=" * 60)

# Setup monitoring (optional)
from tools.monitoring import ProbeManager, SnapshotManager

probe_manager = ProbeManager([
    ("center_surface", (Lx/2, Ly)),
    ("road_center", (11.0, Ly)),
    ("till_interface", (Lx/2, 2.0))
])

snapshot_manager = SnapshotManager(interval_hours=6.0)

# Run with ERT comparison capability
results = coupled.run(
    t_end=config.t_end,
    dt=config.dt,
    probe_manager=probe_manager,
    snapshot_manager=snapshot_manager,
    ert_comparison_interval=6*3600,  # Compare every 6 hours
    print_diagnostics=True
)

# ==========================================
# STEP 12: VALIDATE WITH YOUR ERT DATA
# ==========================================

print("\n" + "=" * 60)
print("ERT MODEL-DATA COMPARISON")
print("=" * 60)

# Predict resistivity from final model state
predicted_rho = domain.predict_resistivity_field(
    richards.p_new,
    transport.c,
    background_sigma=0.02,  # MEASURE THIS from your water samples!
    temperature=10.0        # Winter temperature (°C)
)

print("\nPredicted resistivity field:")
print(f"  Min: {predicted_rho.min():.1f} Ω·m")
print(f"  Max: {predicted_rho.max():.1f} Ω·m")
print(f"  Mean: {predicted_rho.mean():.1f} Ω·m")

# Load your actual ERT measurements
# ert_measured = np.load("your_ert_data.npy")
# 
# rmse = np.sqrt(np.mean((predicted_rho - ert_measured)**2))
# print(f"\nModel-Data RMSE: {rmse:.2f} Ω·m")

# ==========================================
# STEP 13: CALIBRATE PARAMETERS (OPTIONAL)
# ==========================================

print("\n" + "=" * 60)
print("PARAMETER CALIBRATION FROM ERT TIME-SERIES")
print("=" * 60)

# This is the KEY INNOVATION for your project!
# Use time-lapse ERT to constrain uncertain parameters

# Example: Load your TL-ERT time series
# ert_timeseries = {
#     t1: resistivity_array_t1,
#     t2: resistivity_array_t2,
#     ...
# }
# 
# # Extract model predictions at same times/locations
# model_saturation = extract_saturation_timeseries(...)
# model_chloride = extract_chloride_timeseries(...)
# 
# # Calibrate!
# calibrator = ERTCalibrator(ws_model)
# optimal_params, stats = calibrator.calibrate(
#     measured_resistivity=ert_timeseries,
#     porosity=porosity_array,
#     saturation=model_saturation,
#     chloride=model_chloride,
#     background_sigma=0.02
# )
# 
# print("\nOptimal parameters:")
# print(f"  a = {optimal_params.a:.3f}")
# print(f"  m = {optimal_params.m:.3f}")
# print(f"  n = {optimal_params.n:.3f}")
# print(f"  B×Qv = {optimal_params.get_clay_conductivity_term():.2e} S/m")
# print(f"\nCalibration RMSE: {stats['rmse']:.2f} Ω·m")

print("\n(Calibration requires your TL-ERT time-series data)")
print("See ERTCalibrator class for implementation")

# ==========================================
# STEP 14: VISUALIZE RESULTS
# ==========================================

print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

# Plot final state
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Pressure head
ax = axes[0, 0]
p_vals = richards.p_new.dat.data[:]
coords = mesh.coordinates.dat.data
scatter = ax.scatter(coords[:, 0], coords[:, 1], c=p_vals, 
                    cmap='RdYlBu', s=10)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Pressure Head (m)')
plt.colorbar(scatter, ax=ax)

# Chloride concentration
ax = axes[0, 1]
c_vals = transport.c.dat.data[:]
scatter = ax.scatter(coords[:, 0], coords[:, 1], c=c_vals,
                    cmap='YlOrRd', s=10, vmin=0, vmax=c_vals.max())
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Chloride Concentration (mg/L)')
plt.colorbar(scatter, ax=ax)

# Predicted resistivity
ax = axes[1, 0]
scatter = ax.scatter(coords[:, 0], coords[:, 1], c=predicted_rho,
                    cmap='viridis', s=10)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Predicted Resistivity (Ω·m)')
plt.colorbar(scatter, ax=ax)

# Time series at road center
ax = axes[1, 1]
probe_data = probe_manager.get_data()
times_hr = np.array(probe_data['times']) / 3600
ax.plot(times_hr, probe_data['data']['road_center'], 
        label='Pressure (m)', linewidth=2)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Pressure Head (m)')
ax.set_title('Time Series at Road Center')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('montreal_bioswale_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Results saved to 'montreal_bioswale_results.png'")

print("\n" + "=" * 60)
print("SIMULATION COMPLETE!")
print("=" * 60)

print("""
NEXT STEPS FOR YOUR PROJECT:
1. Load your actual TL-ERT time-series data
2. Run calibration to fit (a, m, n, B×Qv) parameters
3. Validate chloride transport predictions against ERT
4. Use calibrated model to evaluate green infrastructure performance
5. Test different de-icing salt management scenarios
""")