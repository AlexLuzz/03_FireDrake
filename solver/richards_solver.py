"""
Main Richards equation solver
"""
from firedrake import (
    Function, TrialFunction, TestFunction, Constant, SpatialCoordinate,
    dx, ds, lhs, rhs, solve, conditional, And, as_vector, grad, dot
)
import numpy as np

class RichardsSolver:
    """
    Richards equation solver using Firedrake
    Implements: Cm * ∂Hp/∂t - ∇·(kr * Ks * ∇Hp) = rain_flux
    """

    def __init__(self, V, domain, source_scenario, bc_manager, config):
        """
        Initialize solver
        
        Args:
            V: Function space
            domain: Domain object with material properties
            source_scenario: SourceScenario object (rain/sources)
            bc_manager: BoundaryConditionManager
            config: SimulationConfig
        """
        self.mesh = domain.mesh
        self.V = V
        self.domain = domain
        self.source_scenario = source_scenario
        self.bc_manager = bc_manager
        self.config = config
        
        # Solution functions
        self.p_n = Function(V, name="Pressure_old")
        self.p_new = Function(V, name="Pressure")
        
        # Coefficient fields
        self.Cm_n = Function(V, name="Moisture_capacity")
        self.kr_n = Function(V, name="Relative_permeability")
        
        # Ks field (spatially varying for heterogeneous domains)
        self.Ks_field = domain.compute_Ks_field(V)
        
        # Initialize
        self._set_initial_conditions()
    
    def _set_initial_conditions(self):
        """Set initial hydrostatic pressure distribution"""
        coords = self.mesh.coordinates.dat.data
        y_coords = coords[:, 1]
        water_table = self.bc_manager.H0_initial
        
        initial_pressure = np.zeros(len(y_coords))
        for i, y in enumerate(y_coords):
            # True hydrostatic equilibrium: p = z_wt - z
            initial_pressure[i] = water_table - y
        
        self.p_n.dat.data[:] = initial_pressure
        
        # Initialize coefficient fields
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Update Cm and kr based on current pressure"""
        self.Cm_n, self.kr_n = self.domain.compute_coefficient_fields(self.p_n)

    def solve_timestep(self, t: float):
        """
        Solve one time step
        
        Args:
            t: Current time (seconds)
        """
        # Update coefficients from previous solution
        self._update_coefficients()
        
        # Get boundary conditions
        bcs = self.bc_manager.get_dirichlet_bcs(t)
        
        # Get rain flux expression from SourceScenario
        # NOTE: t must be in seconds (SourceScenario expects seconds)
        rain_flux = -self.source_scenario.get_flux_expression(t, self.mesh)

        # Define variational problem
        p = TrialFunction(self.V)  # Unknown pressure head
        q = TestFunction(self.V)   # Test function

        # ============================================================================
        # WEAK FORMULATION DERIVATION FOR RICHARDS EQUATION
        # ============================================================================

        # STRONG FORM (PDE to solve):
        # -----------------------------------------------------------------------------
        # C(Hp) ∂Hp/∂t = ∇·[K(Hp)(∇Hp + ∇z)] + source_terms
        #
        # where:
        #   Hp = pressure head [L]
        #   C(Hp) = specific moisture capacity = ∂θ/∂Hp [1/L]
        #   K(Hp) = hydraulic conductivity (depends on saturation) [L/T]
        #   ∇z = gravity gradient = (0, 1) in 2D [-]
        #   source_terms = infiltration, pumping, etc. [L/T]

        # STEP 1: TIME DISCRETIZATION (Backward Euler)
        # -----------------------------------------------------------------------------
        # ∂Hp/∂t ≈ (Hp^{n+1} - Hp^n) / Δt
        #
        # This gives (using Picard linearization with C^n, K^n from previous step):
        #   C^n · (Hp^{n+1} - Hp^n)/Δt = ∇·[K^n(∇Hp^{n+1} + ∇z)] + source

        # STEP 2: WEAK FORMULATION (multiply by test function q, integrate)
        # -----------------------------------------------------------------------------
        #   ∫_Ω C^n · (Hp^{n+1} - Hp^n)/Δt · q dΩ = 
        #       ∫_Ω ∇·[K^n(∇Hp^{n+1} + ∇z)] · q dΩ + ∫_Ω source · q dΩ

        # STEP 3: INTEGRATION BY PARTS (divergence theorem)
        # -----------------------------------------------------------------------------
        # ∫_Ω ∇·(K∇Hp) · q dΩ = -∫_Ω K∇Hp · ∇q dΩ + ∫_∂Ω (K∇Hp · n) · q d∂Ω
        #
        # The boundary term (K∇Hp · n) represents flux through boundary:
        #   - For Neumann BC: specified flux
        #   - For Dirichlet BC: natural condition (handled by constraints)
        #
        # After integration by parts:
        #   ∫_Ω C^n(Hp^{n+1} - Hp^n)/Δt · q dΩ = 
        #       -∫_Ω K^n ∇Hp^{n+1} · ∇q dΩ              [pressure gradient]
        #       -∫_Ω K^n ∇z · ∇q dΩ                      [gravity]
        #       +∫_∂Ω (K^n∇Hp^{n+1} · n) · q d∂Ω        [natural BC]
        #       +∫_Ω source · q dΩ                       [sources]

        # STEP 4: RESIDUAL FORM (move everything to LHS)
        # -----------------------------------------------------------------------------
        # For sources applied as Neumann BC on boundary, replace natural BC term:
        #   ∫_∂Ω (K∇Hp · n) · q d∂Ω  →  ∫_∂Ω flux_bc · q d∂Ω
        #
        # SIGN CONVENTION:
        #   - rain_flux from SourceScenario is NEGATIVE (water entering = negative flux)
        #   - In weak form: F = LHS - RHS = 0
        #   - Boundary flux term: + ∫_∂Ω rain_flux · q d∂Ω
        #   - Since rain_flux < 0, this ADDS water to the system (reduces residual)

        # Compute K field (K = kr * Ks)
        K_field = self.kr_n * self.Ks_field

        # Gravity vector (∇z = (0, 1) assuming z increases upward)
        gravity = as_vector([0, 1])

        # WEAK FORM RESIDUAL: F(p, q) = 0
        F = (
            # TERM 1: TIME DERIVATIVE
            # ∫_Ω C^n · (Hp^{n+1} - Hp^n)/Δt · q dΩ
            self.Cm_n * (p - self.p_n) / self.config.dt * q * dx +
            
            # TERM 2: PRESSURE GRADIENT (diffusion)
            # ∫_Ω K^n ∇Hp^{n+1} · ∇q dΩ
            # (positive sign after integration by parts)
            K_field * dot(grad(p), grad(q)) * dx +
            
            # TERM 3: GRAVITY TERM
            # ∫_Ω K^n ∇z · ∇q dΩ
            # (positive sign after integration by parts)
            K_field * dot(gravity, grad(q)) * dx +
            
            # TERM 4: BOUNDARY FLUX (Neumann BC on top surface)
            # ∫_∂Ω_top rain_flux · q d∂Ω
            # rain_flux is NEGATIVE for rain entering domain
            # This term is POSITIVE in residual, so negative rain_flux reduces residual
            # → water enters system ✓
            # ds(4) = top boundary
            rain_flux * q * ds(4)
        )

        # ============================================================================
        # SOLUTION PROCEDURE
        # ============================================================================
        # Split F(p,q) = 0 into bilinear form a(p,q) and linear form L(q):
        #   a(p,q) - L(q) = 0  →  a(p,q) = L(q)
        
        a = lhs(F)  # Extract terms with both p and q → matrix A
        L = rhs(F)  # Extract terms with only q → vector b
        
        # Solve: A · p_new = b
        solve(a == L, self.p_new, bcs=bcs,
              solver_parameters=self.config.solver_parameters)

        # Update for next time step
        self.p_n.assign(self.p_new)
    
    def run(self, probe_manager=None, snapshot_manager=None):
        """
        Run full simulation
        
        Args:
            probe_manager: ProbeManager for time series (optional)
            snapshot_manager: SnapshotManager for spatial data (optional)
        """
        print("Starting simulation...")
        print(f"Duration: {self.config.t_end/3600:.1f} hours with dt={self.config.dt}s")
        
        if probe_manager is not None:
            probe_manager.record_initial(self.p_n)

        if snapshot_manager is not None:
            snapshot_manager.record_initial(self.p_n)

        t = 0.0
        for step in range(self.config.num_steps):
            t += self.config.dt
            
            # Solve time step
            self.solve_timestep(t)
            
            # Record at monitoring points
            if probe_manager is not None:
                probe_manager.record(t, self.p_new)

            # Record snapshots
            if snapshot_manager is not None:
                if snapshot_manager.should_record(t, self.config.dt):
                    snapshot_manager.record(t, self.p_new)

            # Progress bar (every 5%)
            if step % max(1, int(0.05 * self.config.num_steps)) == 0:
                progress = step / self.config.num_steps
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\rProgress: [{bar}] {progress*100:.1f}% | "
                      f"Time: {t/3600:.1f}h / {self.config.t_end/3600:.1f}h", 
                      end='', flush=True)
        
        print("\n\nSimulation complete!")
        
        # Print summary
        if probe_manager is not None:
            probe_data = probe_manager.get_data()
            print(f"  Total time steps recorded: {len(probe_data['times'])}")
            if len(probe_data['times']) > 0:
                print(f"  Time range: {probe_data['times'][0]/3600:.2f}h - "
                      f"{probe_data['times'][-1]/3600:.2f}h")
            
            for name, data in probe_data['data'].items():
                try:
                    data_array = np.array(data, dtype=float)
                    valid_data = data_array[~np.isnan(data_array)]
                    if len(valid_data) > 0:
                        print(f"  {name}: {valid_data.min():.3f}m - {valid_data.max():.3f}m "
                              f"({len(valid_data)}/{len(data_array)} valid)")
                    else:
                        print(f"  {name}: ALL NaN!")
                except (ValueError, TypeError) as e:
                    print(f"  {name}: {len(data)} data points (error: {e})")

        if snapshot_manager is not None:
            print(f"  Total snapshots recorded: {len(snapshot_manager.snapshots)}")
            for snap_t in sorted(snapshot_manager.snapshots.keys()):
                print(f"    t = {snap_t/3600:.2f}h")
    
    def compute_total_water_content(self):
        """
        Compute total water in domain (for mass balance checks)
        
        Returns:
            Total water content (m³)
        """
        coords = self.mesh.coordinates.dat.data
        p_vals = self.p_new.dat.data[:]
        
        total_theta = 0.0
        for i, (x, y) in enumerate(coords):
            Hp = p_vals[i]
            material = self.domain.get_material_at_point(x, y)
            theta = material.water_content(Hp)
            total_theta += theta
        
        # Approximate volume per node
        dx = self.config.dx
        dy = self.config.dy
        volume_per_node = dx * dy
        
        return total_theta * volume_per_node
    
    def print_diagnostics(self, t: float):
        """Print diagnostic information at current time"""
        p_vals = self.p_new.dat.data[:]
        print(f"\n  Time: {t/3600:.2f}h")
        print(f"  Pressure range: [{p_vals.min():.4f}, {p_vals.max():.4f}] m")
        print(f"  Mean pressure: {p_vals.mean():.4f} m")
        
        # Count saturated nodes
        saturated = np.sum(p_vals > 0)
        total = len(p_vals)
        print(f"  Saturated nodes: {saturated}/{total} ({100*saturated/total:.1f}%)")
        
        # Total water content
        total_water = self.compute_total_water_content()
        print(f"  Total water: {total_water:.3f} m³")