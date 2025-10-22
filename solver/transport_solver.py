"""
Chloride Transport in Porous Media - REFACTORED
Uses TransportMaterial and SourceTerm interfaces

GOVERNING EQUATION:
∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S

Where:
- θ = water content = porosity × saturation (from Richards)
- c = chloride concentration (mg/L or mol/m³)
- v = Darcy velocity (m/s) - from Richards
- D = hydrodynamic dispersion tensor (m²/s) - from TransportMaterial
- S = source/sink term (mg/L/s) - from SourceTerm
"""
from firedrake import *
import numpy as np

class ChlorideTransport:
    """
    Transport of dissolved chloride in soil water
    Refactored to use modular TransportMaterial and SourceTerm classes
    """

    def __init__(self, mesh, domain, transport_material, 
                 source_term, bc_manager):
        """
        Initialize chloride transport model
        
        Parameters:
        -----------
        mesh : firedrake.Mesh
            Computational mesh
        
        domain : Domain
            Domain object with material properties (for porosity)
        
        transport_material : TransportMaterial
            Transport properties (diffusion, dispersivity, etc.)
            from transport_properties.py
        
        source_term : SourceTerm
            Source/sink term from source_terms.py
        
        bc_manager : BoundaryConditionManager
            Boundary conditions for concentration
        
        Example:
        --------
        from physics.transport_properties import TransportMaterial
        from physics.source_terms import CompositeSource
        
        # Define transport properties
        cl_transport = TransportMaterial.from_contaminant(
            'chloride',
            alpha_L=0.05,
            alpha_T=0.005
        )
        
        # Define sources
        total_source = CompositeSource([cl_rain, road_salt])
        
        # Create solver
        transport = ChlorideTransport(
            mesh, domain, cl_transport, total_source, bc_manager
        )
        """
        self.mesh = mesh
        self.domain = domain
        self.transport_material = transport_material
        self.source_term = source_term
        self.bc_manager = bc_manager

        # Function spaces
        self.V = FunctionSpace(mesh, "CG", 1)  # Scalar concentration
        self.V_vec = VectorFunctionSpace(mesh, "CG", 1)  # Vector velocity
        
        # === STATE VARIABLES ===
        self.c = Function(self.V, name="Chloride_Concentration")
        self.c_n = Function(self.V)  # Previous timestep
        
        # === FIELDS FROM RICHARDS SOLVER ===
        self.velocity = Function(self.V_vec, name="Darcy_Velocity")
        self.saturation = Function(self.V, name="Saturation")
        self.porosity_field = Function(self.V, name="Porosity")
        
        # Initialize porosity field from domain
        self._initialize_porosity_field()
        
        # Time step (updated during solve)
        self.dt = Constant(3600.0)  # Default 1 hour
        
    def _initialize_porosity_field(self):
        """Initialize spatially-varying porosity from domain materials"""
        coords = self.mesh.coordinates.dat.data
        porosity_vals = np.zeros(len(coords))
        
        for i, (x, y) in enumerate(coords):
            material = self.domain.get_material_at_point(x, y)
            porosity_vals[i] = material.porosity
        
        self.porosity_field.dat.data[:] = porosity_vals
    
    def set_velocity(self, velocity):
        """Update Darcy velocity from Richards solver"""
        self.velocity.assign(velocity)
        
    def set_saturation(self, saturation):
        """Update saturation from Richards solver"""
        self.saturation.assign(saturation)
    
    def set_initial_condition(self, c_init):
        """
        Set initial chloride concentration
        
        Parameters:
        -----------
        c_init : float, Function, or Expression
            Initial concentration (mg/L or mol/m³)
        """
        if isinstance(c_init, (int, float)):
            self.c.assign(Constant(c_init))
        else:
            self.c.assign(c_init)
        self.c_n.assign(self.c)
    
    def effective_diffusivity(self):
        """
        Calculate effective diffusion coefficient using TransportMaterial
        
        D_eff = D₀ × τ(θ) where τ depends on tortuosity model
        
        Returns: D_eff (m²/s) as UFL expression
        """
        # Use TransportMaterial's method with current porosity and saturation
        # For UFL expression, we compute it directly here
        n = self.porosity_field
        s = self.saturation
        theta = n * s
        
        D0 = self.transport_material.contaminant.diffusion_coeff
        
        # Tortuosity from transport material
        if self.transport_material.tortuosity_model == 'millington_quirk':
            tau = theta**(10.0/3.0) / (n**2)
        elif self.transport_material.tortuosity_model == 'bruggeman':
            tau = theta**1.5
        else:  # simple
            tau = theta / n
        
        return D0 * tau
    
    def dispersion_tensor(self):
        """
        Calculate hydrodynamic dispersion tensor
        Combines molecular diffusion + mechanical dispersion
        
        D_ij = (αₜ|v| + D_eff)δ_ij + (αₗ - αₜ)(vᵢvⱼ/|v|)
        
        Returns: 2×2 tensor D
        """
        v = self.velocity
        v_mag = sqrt(dot(v, v) + 1e-12)
        
        # Effective diffusion
        D_eff = self.effective_diffusivity()
        
        # Mechanical dispersion from TransportMaterial
        alpha_L = self.transport_material.alpha_L
        alpha_T = self.transport_material.alpha_T
        
        D_mech_L = alpha_L * v_mag
        D_mech_T = alpha_T * v_mag
        
        # Normalized velocity
        v_norm = v / v_mag
        vx, vy = v_norm[0], v_norm[1]
        
        # Build tensor
        D_tensor = as_matrix([
            [D_eff + D_mech_T + (D_mech_L - D_mech_T) * vx**2,
             (D_mech_L - D_mech_T) * vx * vy],
            [(D_mech_L - D_mech_T) * vx * vy,
             D_eff + D_mech_T + (D_mech_L - D_mech_T) * vy**2]
        ])
        
        return D_tensor
    
    def build_variational_form(self, t: float):
        """
        Build weak form for transport equation
        
        Parameters:
        -----------
        t : float
            Current time (seconds) - needed for source term evaluation
        
        Returns: UFL form F
        """
        # Trial and test functions
        c = self.c
        c_n = self.c_n
        w = TestFunction(self.V)
        
        # Water content
        theta = self.porosity_field * self.saturation
        
        # === TERM 1: ACCUMULATION ===
        time_term = theta * (c - c_n) / self.dt * w * dx
        
        # === TERM 2: ADVECTION ===
        v = self.velocity
        convection_term = -c * dot(v, grad(w)) * dx
        
        # === TERM 3: DISPERSION ===
        D = self.dispersion_tensor()
        diffusion_term = inner(D * grad(c), grad(w)) * dx
        
        # === TERM 4: SOURCE/SINK ===
        # Get source expression from SourceTerm
        S = self.source_term.get_ufl_expression(t, self.mesh)
        source_term_form = -S * w * dx
        
        # === COMPLETE WEAK FORM ===
        F = time_term + convection_term + diffusion_term + source_term_form
        
        return F
    
    def solve_timestep(self, t: float, dt: float = None, 
                       solver_parameters: dict = None):
        """
        Solve transport equation for one timestep
        
        Parameters:
        -----------
        t : float
            Current time (seconds)
        dt : float
            Timestep size (seconds)
        solver_parameters : dict
            Solver options
        
        Returns: concentration field c
        """
        if dt is not None:
            self.dt.assign(dt)
        
        # Build weak form with current time (for source evaluation)
        F = self.build_variational_form(t)
        
        # Get boundary conditions
        bcs = self.bc_manager.get_dirichlet_bcs()
        
        # Default solver settings
        if solver_parameters is None:
            solver_parameters = {
                'ksp_type': 'gmres',
                'pc_type': 'ilu',
                'ksp_rtol': 1e-8
            }
        
        # Solve
        solve(F == 0, self.c, bcs=bcs, solver_parameters=solver_parameters)
        
        # Update for next timestep
        self.c_n.assign(self.c)
        
        return self.c
    
    def get_concentration_stats(self):
        """Get diagnostic statistics on concentration field"""
        c_vals = self.c.dat.data[:]
        return {
            'min': c_vals.min(),
            'max': c_vals.max(),
            'mean': c_vals.mean(),
            'std': c_vals.std()
        }


# ==========================================
# COUPLED SOLVER
# ==========================================

class CoupledFlowTransport:
    """
    Couples Richards equation with chloride transport
    
    Workflow each timestep:
    1. Solve Richards → pressure field
    2. Compute Darcy velocity: v = -K∇H
    3. Update saturation from pressure
    4. Solve transport with velocity and saturation
    5. (Optional) Compare with ERT measurements
    """
    
    def __init__(self, richards_solver, transport_solver):
        """
        Parameters:
        -----------
        richards_solver : RichardsSolver
            Flow solver instance
        transport_solver : ChlorideTransport
            Transport solver instance
        """
        self.richards = richards_solver
        self.transport = transport_solver
        self.mesh = richards_solver.mesh
        self.V = richards_solver.V
        
        # Vector space for velocity
        self.V_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.velocity = Function(self.V_vec, name="Darcy_Velocity")
        
        # For ERT comparison (optional)
        self.ert_predictions = []
        self.ert_times = []
    
    def compute_darcy_velocity(self):
        """
        Compute Darcy velocity from Richards results
        
        v = -K∇H where H = Hp + z
        """
        # Get pressure head
        Hp = self.richards.p_new
        
        # Get hydraulic conductivity
        K = self.richards.kr_n * self.richards.Ks_field
        
        # Get elevation coordinate
        coords = SpatialCoordinate(self.mesh)
        z = coords[1]  # y-coordinate is elevation in 2D
        
        # Total hydraulic head
        H = Hp + z
        
        # Project velocity: v = -K∇H
        v = TrialFunction(self.V_vec)
        w = TestFunction(self.V_vec)
        
        a = inner(v, w) * dx
        L = inner(-K * grad(H), w) * dx
        
        solve(a == L, self.velocity)
        
        return self.velocity
    
    def run(self, t_end, dt, 
            probe_manager=None, 
            snapshot_manager=None,
            ert_comparison_interval=None,
            ert_measured_data=None,
            print_diagnostics=False):
        """
        Run coupled simulation
        
        Parameters:
        -----------
        t_end : float
            Total simulation time (seconds)
        dt : float
            Timestep size (seconds)
        probe_manager : ProbeManager (optional)
            For time-series at specific points
        snapshot_manager : SnapshotManager (optional)
            For spatial snapshots at intervals
        ert_comparison_interval : float (optional)
            How often to compare with ERT data (seconds)
        ert_measured_data : dict (optional)
            Measured ERT data: {time: resistivity_array}
        print_diagnostics : bool
            Print detailed diagnostics each timestep
        
        Returns:
        --------
        results : dict
            Final fields and diagnostics
        """
        print("=" * 60)
        print("COUPLED FLOW-TRANSPORT SIMULATION")
        print("=" * 60)
        print(f"Duration: {t_end/3600:.1f} hours")
        print(f"Timestep: {dt}s ({dt/3600:.3f} hours)")
        print(f"Total steps: {int(t_end/dt)}")
        
        t = 0.0
        step = 0
        n_steps = int(t_end / dt)
        
        # Record initial conditions
        if probe_manager is not None:
            probe_manager.record(t, self.richards.p_new)
            if hasattr(probe_manager, 'record_scalar'):
                probe_manager.record_scalar(t, self.transport.c, 'concentration')
        
        if snapshot_manager is not None:
            snapshot_manager.record(t, self.richards.p_new)
            if hasattr(snapshot_manager, 'record_field'):
                snapshot_manager.record_field(t, self.transport.c, 'concentration')
        
        print("\nStarting time integration...")
        
        while t < t_end:
            t += dt
            step += 1
            
            # === STEP 1: SOLVE RICHARDS ===
            self.richards.solve_timestep(t)
            
            # === STEP 2: COMPUTE VELOCITY ===
            velocity = self.compute_darcy_velocity()
            self.transport.set_velocity(velocity)
            
            # === STEP 3: UPDATE SATURATION ===
            saturation = self.richards.domain.compute_saturation_field(
                self.richards.p_new
            )
            self.transport.set_saturation(saturation)
            
            # === STEP 4: SOLVE TRANSPORT ===
            self.transport.solve_timestep(t, dt)
            
            # === STEP 5: ERT COMPARISON (optional) ===
            if ert_comparison_interval is not None:
                if step % int(ert_comparison_interval / dt) == 0:
                    self._compare_with_ert(t, ert_measured_data)
            
            # === RECORD DATA ===
            if probe_manager is not None:
                probe_manager.record(t, self.richards.p_new)
                if hasattr(probe_manager, 'record_scalar'):
                    probe_manager.record_scalar(t, self.transport.c, 'concentration')
            
            if snapshot_manager is not None:
                if snapshot_manager.should_record(t, dt):
                    snapshot_manager.record(t, self.richards.p_new)
                    if hasattr(snapshot_manager, 'record_field'):
                        snapshot_manager.record_field(t, self.transport.c, 'concentration')
                        snapshot_manager.record_field(t, self.velocity, 'velocity')
            
            # === PROGRESS BAR ===
            if step % max(1, int(0.05 * n_steps)) == 0:
                progress = step / n_steps
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r[{bar}] {progress*100:.1f}% | t={t/3600:.2f}h", 
                      end='', flush=True)
            
            # === DIAGNOSTICS ===
            if print_diagnostics and step % max(1, int(0.1 * n_steps)) == 0:
                print()
                self._print_diagnostics(t)
        
        print("\n\nSimulation complete!")
        
        # === FINAL RESULTS ===
        results = {
            'pressure': self.richards.p_new,
            'concentration': self.transport.c,
            'velocity': self.velocity,
            'saturation': saturation,
            'times': np.linspace(0, t_end, step+1)
        }
        
        # Add ERT comparison if available
        if self.ert_predictions:
            results['ert_predictions'] = self.ert_predictions
            results['ert_times'] = self.ert_times
        
        self._print_summary()
        
        return results
    
    def _compare_with_ert(self, t, ert_measured_data):
        """Compare model prediction with ERT measurements"""
        if not hasattr(self.richards.domain, 'predict_resistivity_field'):
            return
        
        # Predict resistivity from model state
        predicted_rho = self.richards.domain.predict_resistivity_field(
            self.richards.p_new,
            self.transport.c
        )
        
        self.ert_predictions.append(predicted_rho)
        self.ert_times.append(t)
        
        # Compare with measurements if available
        if ert_measured_data is not None and t in ert_measured_data:
            measured_rho = ert_measured_data[t]
            rmse = np.sqrt(np.mean((predicted_rho - measured_rho)**2))
            print(f"\n  ERT comparison at t={t/3600:.2f}h: RMSE={rmse:.2f} Ω·m")
    
    def _print_diagnostics(self, t):
        """Print detailed diagnostics"""
        print(f"\n=== Diagnostics at t={t/3600:.2f}h ===")
        
        # Pressure/saturation
        p_vals = self.richards.p_new.dat.data[:]
        print(f"Pressure: min={p_vals.min():.3f}m, max={p_vals.max():.3f}m")
        
        saturated = np.sum(p_vals > 0)
        print(f"Saturated nodes: {saturated}/{len(p_vals)} ({100*saturated/len(p_vals):.1f}%)")
        
        # Concentration
        c_stats = self.transport.get_concentration_stats()
        print(f"Chloride: min={c_stats['min']:.2f}, max={c_stats['max']:.2f}, "
              f"mean={c_stats['mean']:.2f} mg/L")
        
        # Velocity
        v_vals = self.velocity.dat.data[:]
        v_mag = np.sqrt(v_vals[:, 0]**2 + v_vals[:, 1]**2)
        print(f"Velocity: max={v_mag.max():.2e} m/s")
    
    def _print_summary(self):
        """Print final summary statistics"""
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        
        # Final concentration distribution
        c_stats = self.transport.get_concentration_stats()
        print(f"\nFinal chloride concentration:")
        print(f"  Min: {c_stats['min']:.2f} mg/L")
        print(f"  Max: {c_stats['max']:.2f} mg/L")
        print(f"  Mean: {c_stats['mean']:.2f} mg/L")
        print(f"  Std: {c_stats['std']:.2f} mg/L")
        
        # Saturation info
        p_vals = self.richards.p_new.dat.data[:]
        saturated_pct = 100 * np.sum(p_vals > 0) / len(p_vals)
        print(f"\nFinal saturation state:")
        print(f"  Saturated zone: {saturated_pct:.1f}% of domain")
        
        print("\n" + "=" * 60)


