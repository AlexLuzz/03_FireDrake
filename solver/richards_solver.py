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

    def __init__(self, V, domain, rain_scenario, bc_manager, config):
        """
        Initialize solver
        
        Args:
            mesh: Firedrake mesh
            V: Function space
            domain: Domain object with material properties
            rain_scenario: RainScenario object
            bc_manager: BoundaryConditionManager
            config: SimulationConfig
        """
        self.mesh = domain.mesh
        self.V = V
        self.domain = domain
        self.rain_scenario = rain_scenario
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
    
    def get_rain_flux_expression(self, t: float):
        """
        Get spatially-varying rain flux expression for current time
        Rain is applied as Neumann BC (flux) on top boundary
        
        Args:
            t: Current time (seconds)
        
        Returns:
            UFL expression for rain flux (m/s, negative = into domain)
        """        
        t_hours = t / 3600.0
        coords = SpatialCoordinate(self.mesh)
        x = coords[0]
        
        # Build piecewise expression for all zones
        flux_expr = Constant(0.0)
        
        for event in self.rain_scenario.events:
            if event.is_active(t_hours):
                # Event is active - check each zone
                for zone in event.zones:
                    # Calculate flux for this zone
                    zone_flux = event.intensity * zone.multiplier / 3600000.0  # mm/hr to m/s
                    
                    # Add contribution if x is in zone
                    flux_expr = conditional(
                        And(x >= zone.x_min, x <= zone.x_max),
                        Constant(-zone_flux),  # Negative = into domain
                        flux_expr  # Keep previous value if not in zone
                    )
        
        return flux_expr

    def solve_timestep(self, t: float):
        """
        Solve one time step
        
        Args:
            t: Current time (seconds)
            bcs: Dirichlet boundary conditions (optional)
        """
        # Update coefficients from previous solution
        self._update_coefficients()
        
        # Get boundary conditions
        bcs = self.bc_manager.get_dirichlet_bcs(t)
        
        # Get rain flux expression
        rain_flux = self.get_rain_flux_expression(t)
        
        # Define variational problem
        p = TrialFunction(self.V)  # Unknown pressure head (to solve for)
        q = TestFunction(self.V)   # Test function (mathematical trick for FEM)

        # ============================================================================
        # WEAK FORMULATION DERIVATION FOR RICHARDS EQUATION
        # ============================================================================

        # STRONG FORM (PDE to solve):
        # -----------------------------------------------------------------------------
        # C(Hp) ∂Hp/∂t = ∇·[K(Hp)(∇Hp + ∇z)] + source terms
        #
        # where:
        #   Hp = pressure head [L]
        #   C(Hp) = specific moisture capacity = ∂θ/∂Hp [1/L]
        #   K(Hp) = hydraulic conductivity (depends on saturation) [L/T]
        #   ∇z = gravity gradient (vertical direction) [-]
        #   θ = volumetric water content [-] (used in C(Hp) and K(Hp))

        # STEP 1: TIME DISCRETIZATION (Backward Euler)
        # -----------------------------------------------------------------------------
        # Discretize time derivative using implicit scheme:
        #   ∂Hp/∂t ≈ (Hp^{n+1} - Hp^n) / Δt
        #
        # This gives us:
        #   C^n · (Hp^{n+1} - Hp^n)/Δt = ∇·[K^n(∇Hp^{n+1} + ∇z)]
        #
        # Note: We use values from previous time step (n) for C and K to linearize
        # the problem, making it easier to solve (Picard iteration)

        # STEP 2: WEAK FORMULATION (Galerkin Method)
        # -----------------------------------------------------------------------------
        # Multiply both sides by test function q and integrate over domain Ω:
        #
        #   ∫_Ω C^n · (Hp^{n+1} - Hp^n)/Δt · q dΩ = ∫_Ω ∇·[K^n(∇Hp^{n+1} + ∇z)] · q dΩ
        #
        # STEP 3: INTEGRATION BY PARTS (Green's first identity)
        # -----------------------------------------------------------------------------
        # For the right-hand side, apply divergence theorem:
        #   ∫_Ω ∇·(K∇Hp) · q dΩ = -∫_Ω K∇Hp · ∇q dΩ + ∫_∂Ω K∇Hp · n · q d∂Ω
        #
        # The boundary integral (∫_∂Ω) contains flux boundary conditions
        # The volume integral (∫_Ω) is what we keep in the weak form
        #
        # After integration by parts:
        #   ∫_Ω C^n(Hp^{n+1} - Hp^n)/Δt · q dΩ = 
        #       -∫_Ω K^n∇Hp^{n+1} · ∇q dΩ          [diffusion term]
        #       -∫_Ω K^n∇z · ∇q dΩ                  [gravity term]
        #       +∫_∂Ω flux_bc · q d∂Ω               [boundary fluxes]

        # STEP 4: REARRANGE TO RESIDUAL FORM (F = 0)
        # -----------------------------------------------------------------------------
        # Move everything to left side to get residual form F(Hp^{n+1}) = 0:

        # Compute K field (K = kr * Ks)
        # Linearization: use K from previous time step (known values)
        K_field = self.kr_n * self.Ks_field

        # Gravity vector (vertical direction, pointing down in our convention)
        # In 2D: ∇z = (∂z/∂x, ∂z/∂y) = (0, 1) assuming z increases upward
        gravity = as_vector([0, 1])  # ∇z = (0, 1)

        # WEAK FORM RESIDUAL: F(p, q) = 0
        # Each term corresponds to part of the weak formulation above
        F = (
            # TERM 1: TIME DERIVATIVE
            # ∫_Ω C^n · (Hp^{n+1} - Hp^n)/Δt · q dΩ
            # This represents the change in water content over time
            self.Cm_n * (p - self.p_n) / self.config.dt * q * dx +
            
            # TERM 2: DIFFUSION (from -∫_Ω K∇Hp · ∇q dΩ)
            # After integration by parts, we get positive sign here
            # This represents water flow driven by pressure gradients
            # dot(grad(p), grad(q)) = ∇p · ∇q
            K_field * dot(grad(p), grad(q)) * dx +
            
            # TERM 3: GRAVITY (from -∫_Ω K∇z · ∇q dΩ)
            # After integration by parts, we get positive sign here
            # This represents water flow driven by gravity
            # dot(gravity, grad(q)) = ∇z · ∇q = (0,1) · (∂q/∂x, ∂q/∂y) = ∂q/∂y
            K_field * dot(gravity, grad(q)) * dx +
            
            # TERM 4: BOUNDARY FLUX (from ∫_∂Ω flux · q d∂Ω)
            # This is the infiltration/rainfall boundary condition on top surface
            # Positive rain_flux means water entering the domain
            # ds(4) integrates over boundary marker 4 (top surface)
            rain_flux * q * ds(4)
        )

        # SOLVING THE SYSTEM:
        # We seek p (Hp^{n+1}) such that F(p, q) = 0 for all test functions q
        # FEniCS will solve this using Newton's method or linear solver depending
        # on whether the problem is nonlinear

        # ============================================================================
        # WHY THIS FORMULATION?
        # ============================================================================
        # 1. Integration by parts reduces continuity requirements on the solution
        #    (C^0 continuity instead of C^1)
        # 2. Natural boundary conditions (Neumann) automatically included via 
        #    boundary integrals
        # 3. Symmetric weak form (if K constant) leads to symmetric matrix system
        # 4. Galerkin method ensures optimal approximation in H^1 norm

        # Split residual form F=0 into linear system a=L
        # lhs: extracts terms with unknown p → matrix A
        # rhs: extracts known terms (p_n, sources) → vector b
        a = lhs(F)  # Bilinear form: a(p,q) - contains both p and q
        L = rhs(F)  # Linear form: L(q) - contains only q (negated from F)

        # Solve linear system A·p_new = b
        # The nonlinear problem is linearized using known values (Cm_n, K_field)
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
        
        probe_manager.record_initial(self.p_n)

        if snapshot_manager is not None:
            snapshot_manager.record_initial(self.p_n)

        t = 0.0
        for step in range(self.config.num_steps):
            t += self.config.dt
            
            # Solve time step
            self.solve_timestep(t)
            
            # Record at monitoring points
            probe_manager.record(t, self.p_new)

            # During simulation
            if snapshot_manager is not None:
                if snapshot_manager.should_record(t, self.config.dt):
                    snapshot_manager.record(t, self.p_new)  # Main snapshots

            # Print progress bar every 5% advancement
            if step % int(0.05 * self.config.num_steps) == 0:
                progress = step / self.config.num_steps
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\rProgress: [{bar}] {progress*100:.1f}% | Time: {t/3600:.1f}h / {self.config.t_end/3600:.1f}h", end='', flush=True)
        
        print("\n\nSimulation complete!")
        
        # Print probe data summary
        probe_data = probe_manager.get_data()
        print(f"  Total time steps recorded: {len(probe_data['times'])}")
        print(f"  Time range: {probe_data['times'][0]/3600:.2f}h - {probe_data['times'][-1]/3600:.2f}h")
        for name, data in probe_data['data'].items():
            # Convert to numpy array, handling potential non-numeric types
            try:
                data_array = np.array(data, dtype=float)
                valid_data = data_array[~np.isnan(data_array)]
                if len(valid_data) > 0:
                    print(f"  {name}: {valid_data.min():.3f}m - {valid_data.max():.3f}m ({len(valid_data)}/{len(data_array)} valid)")
                else:
                    print(f"  {name}: ALL NaN!")
            except (ValueError, TypeError) as e:
                print(f"  {name}: {len(data)} data points (error: {e})")

        if snapshot_manager is not None:
            print(f"  Total snapshots recorded: {len(snapshot_manager.snapshots)}")
            for t in sorted(snapshot_manager.snapshots.keys()):
                print(f"    t = {t/3600:.2f}h")
