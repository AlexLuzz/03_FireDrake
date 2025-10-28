"""
Advection-Dispersion transport solver
Solves: ∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S
"""
from firedrake import (
    Function, TrialFunction, TestFunction, dx, ds, lhs, rhs, solve, min_value,
    as_matrix, sqrt, grad, dot, conditional, project, VectorFunctionSpace, Constant, as_vector, 
)
import numpy as np

class TransportSolver:
    def __init__(self, domain, V, field_map, pressure_solver, 
                 bc_manager, transport_source, config, debug=False):
        """
        Parameters:
        -----------
        domain : Domain
            Computational domain
        V : FunctionSpace
            Function space for concentration
        field_map : MaterialField
            Field mapper (must have transport models)
        pressure_solver : RichardsSolver
            Richards solver (provides pressure field)
        bc_manager : BCManager
            Boundary condition manager
        transport_source : TransportSource
            Source/sink terms for contaminant
        config : Config
            Simulation configuration
        debug : bool
            Enable debug output
        """
        if not field_map.has_transport():
            raise ValueError("MaterialField must have transport models assigned")
        
        self.mesh = domain.mesh
        self.V = V
        self.V_vec = VectorFunctionSpace(self.mesh, "CG", 1)  # For velocity
        self.field_map = field_map
        self.pressure_solver = pressure_solver
        self.bc_manager = bc_manager
        self.transport_source = transport_source
        self.config = config
        self.debug = debug
        self.dx = dx(domain=self.mesh)

        # Concentration fields
        self.c_n = Function(V, name="Concentration_old")
        self.c_new = Function(V, name="Concentration")
        
        # Velocity field (computed from pressure)
        self.velocity = Function(self.V_vec, name="Darcy_velocity")
        
        self._set_initial_conditions()
    
    def _set_initial_conditions(self):
        """Set initial concentration (default: zero everywhere)"""
        self.c_n.dat.data[:] = 0.0
    
    def set_initial_concentration(self, concentration_func):
        """
        Set custom initial concentration
        
        Parameters:
        -----------
        concentration_func : callable or float
            - If callable: function(x, y) -> concentration
            - If float: uniform concentration
        """
        if callable(concentration_func):
            coords = self.mesh.coordinates.dat.data_ro
            for i, (x, y) in enumerate(coords):
                self.c_n.dat.data[i] = concentration_func(x, y)
        else:
            self.c_n.dat.data[:] = float(concentration_func)
    
    def compute_darcy_velocity(self):
        """
        Compute Darcy velocity from pressure field
        v = -K∇h where h is hydraulic head (pressure in meters)
        Since Richards solver works with head, we don't add gravity here
        """
        pressure = self.pressure_solver.p_n
        
        # Get hydraulic conductivity field
        k = self.field_map.get_K_field(pressure)
        
        # Compute v = -K∇h (head already includes elevation effects)
        vx = project(-k * grad(pressure)[0], self.V)
        vy = project(-k * (grad(pressure)[1] + 1.0), self.V)  # Add gravity in y-direction

        # Assemble into vector field
        self.velocity.dat.data[:, 0] = np.maximum(vx.dat.data_ro, 1e-14)
        self.velocity.dat.data[:, 1] = np.maximum(vy.dat.data_ro, 1e-14)

        # Assemble into vector field
        self.velocity.dat.data[:, 0] = vx.dat.data_ro
        self.velocity.dat.data[:, 1] = vy.dat.data_ro

    def construct_dispersion_tensor(self, D_0, alpha_L, alpha_T, vx, vy):
        """ 
        Construct the anisotropic dispersion tensor (2D)
        D_ij = D0*δ_ij + |v| * [α_T*δ_ij + (α_L - α_T)*(v_i*v_j / |v|^2)]

        Parameters
        ----------
        D0 : float or UFL expression
            Molecular diffusion coefficient
        alpha_L, alpha_T : float
            Longitudinal and transverse dispersivities
        vx, vy : UFL expressions
            Velocity components

        Returns
        -------
        D_tensor : 2x2 UFL matrix
        """
        v_mag = sqrt(vx**2 + vy**2 + 1e-10)  # avoid division by zero

        # Longitudinal–transverse dispersion difference
        dalpha = alpha_L - alpha_T

        # Tensor components
        D_xx = D_0 + v_mag * (alpha_T + dalpha * (vx**2) / (v_mag**2))
        D_yy = D_0 + v_mag * (alpha_T + dalpha * (vy**2) / (v_mag**2))
        #D_xy = v_mag * dalpha * (vx * vy) / (v_mag**2)

        # Assemble symmetric tensor
        D_tensor = as_matrix([[D_xx, 0],
                            [0, D_yy]])

        return D_tensor

    def solve_timestep(self, t: float):
        """
        Solve transport for one timestep
        1. Compute Darcy velocity from current pressure
        2. Solve advection-dispersion equation
        """
        # Step 1: Compute velocity field
        self.compute_darcy_velocity()
        
        # Step 2: Get transport coefficients from current pressure
        pressure = self.pressure_solver.p_n
        theta = self.field_map.get_theta_field(pressure)
        porosity = self.field_map.get_porosity_field()

        tortuosity = theta**(10.0/3.0) / (porosity**2)


        #R = self.field_map.get_retardation_field(pressure)
        
        # Get dispersion coefficients
        vx = Function(self.V)
        vy = Function(self.V)
        vx.dat.data[:] = self.velocity.dat.data_ro[:, 0]
        vy.dat.data[:] = self.velocity.dat.data_ro[:, 1]
        
        alpha_L = 1  # Longitudinal dispersivity [m]
        alpha_T = alpha_L/10  # Transverse dispersivity [m]
        Dd=2.03e-8
        D_0 = Dd * tortuosity

        # Step 3: Get source
        source_expr = self.transport_source.get_flux_expression(t, self.mesh)
        
        # Step 4: Solve transport equation
        # ∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S
        # With retardation: R·θ·∂c/∂t + ∇·(vc) = ∇·(D∇c) + S
        
        c = TrialFunction(self.V)
        q = TestFunction(self.V)
        
        # Effective storage term (accounts for retardation)
        #storage_coeff = R * theta
        
        # Time discretization parameter (theta-method)
        theta_time = 1.0  # is backward Euler (implicit, stable)
        dt = Constant(self.config.dt)

        c_mid = theta_time * c + (1.0 - theta_time) * self.c_n
        
        # (1) Time derivative term
        F = (c - self.c_n) / dt * q * dx

        # (2) Advection term (conservative form)
        # integrate ∇·(v c) by parts → -∫ c v·∇q + ∫_Γ q c v·n
        F += - dot(self.velocity, grad(q)) * c_mid * dx

        # (3) Dispersion–diffusion term
        D_tensor = self.construct_dispersion_tensor(D_0, alpha_L, alpha_T, vx, vy)
        F += dot(dot(D_tensor, grad(c_mid)), grad(q)) * dx

        # (4) Source/sink term
        F += -source_expr * q * dx

        # ✅ ADD DECAY (first-order degradation)
        # decay_rate = Constant(0.0001)  # 1/s - tune this (0.0001 = 0.36/hour half-life)
        # F += decay_rate * theta * c_adv * q * dx  # Removes mass

        a = lhs(F)
        L = rhs(F)
        
        # Solve with appropriate solver parameters for transport
        solver_params = {
            'ksp_type': 'preonly',    # Direct solve
            'pc_type': 'lu',          # LU factorization
            'pc_factor_mat_solver_type': 'mumps'  # Robust direct solver
        }

        # Solve (no boundary conditions for transport - natural no-flux boundaries)
        solve(a == L, self.c_new, bcs=[],
              solver_parameters=solver_params)

        # Enforce non-negative concentration (physical constraint)
        c_data = self.c_new.dat.data
        negative_nodes = c_data < 0
        if negative_nodes.any():
            num_negative = negative_nodes.sum()
            min_val = c_data.min()
            if self.debug:
                print(f"  WARNING: {num_negative} nodes with negative concentration (min={min_val:.3f}), setting to 0")
            c_data[negative_nodes] = 0.0

        # Update
        self.c_n.assign(self.c_new)
    
    def solve_timestep_DG(self, t: float):
        """DG version of transport solve"""
        from firedrake import (FacetNormal, jump, avg, conditional, 
                            dS, CellVolume, FacetArea)
        
        # Step 1-2: Same as before (compute velocity, get coefficients)
        self.compute_darcy_velocity()
        pressure = self.pressure_solver.p_n
        theta = self.field_map.get_theta_field(pressure)
        R = self.field_map.get_retardation_field(pressure)
        
        # Get velocity components
        vx = Function(self.V)
        vy = Function(self.V)
        vx.dat.data[:] = self.velocity.dat.data_ro[:, 0]
        vy.dat.data[:] = self.velocity.dat.data_ro[:, 1]
        
        D_L = Constant(5.0e-5)
        D_T = Constant(5.0e-6)
        D_tensor = self.construct_dispersion_tensor(D_L, D_T, vx, vy)
        
        source_expr = self.transport_source.get_flux_expression(t, self.mesh)
        
        # Trial and test functions
        c = TrialFunction(self.V)
        q = TestFunction(self.V)
        
        storage_coeff = R * theta
        theta_time = 1.0
        dt = Constant(self.config.dt)
        c_mid = theta_time * c + (1.0 - theta_time) * self.c_n
        
        # === DG FORMULATION ===
        
        # (1) Time derivative (unchanged)
        F = storage_coeff * (c - self.c_n) / dt * q * dx
        
        # (2) ADVECTION with upwind flux
        n = FacetNormal(self.mesh)
        v_vec = as_vector([vx, vy])
        
        # Volumetric term (integration by parts)
        F += dot(v_vec, grad(q)) * c_mid * dx
        
        # Upwind numerical flux on interior faces
        v_n = dot(v_vec, n)
        c_up = conditional(v_n('+') > 0, c_mid('+'), c_mid('-'))
        F += jump(q) * (v_n('+') * c_up) * dS
        
        # Boundary flux (for no-flux boundaries, often zero)
        # F += q * v_n * c_mid * ds  # Only if you have inflow/outflow
        
        # (3) DISPERSION with Interior Penalty (SIPG)
        h = CellVolume(self.mesh) / FacetArea(self.mesh)
        alpha = Constant(10.0)  # Penalty parameter - tune if unstable
        
        # Volumetric term
        F += dot(dot(D_tensor, grad(c_mid)), grad(q)) * dx
        
        # Interior penalty terms
        D_avg = avg(D_tensor)
        F += -dot(avg(dot(D_tensor, grad(c_mid))), jump(q, n)) * dS
        F += -dot(jump(c_mid, n), avg(dot(D_tensor, grad(q)))) * dS
        F += (alpha/avg(h)) * dot(jump(c_mid, n), jump(q, n)) * dS
        
        # Boundary penalty (if Dirichlet BCs needed)
        # F += -(dot(dot(D_tensor, grad(c_mid)), n) * q) * ds
        # F += -(dot(dot(D_tensor, grad(q)), n) * c_mid) * ds
        # F += (alpha/h) * c_mid * q * ds
        
        # (4) Source term (unchanged)
        F += -source_expr * q * dx
        
        # Extract bilinear and linear forms
        a = lhs(F)
        L = rhs(F)
    
        # Solver parameters (may need adjustment for DG)
        solver_params = {
            'ksp_type': 'gmres',
            'pc_type': 'bjacobi',  # Block Jacobi often better for DG
            'sub_pc_type': 'ilu',
            'ksp_rtol': 1e-6,
            'ksp_atol': 1e-8,
            'ksp_max_it': 200
        }
        
        solve(a == L, self.c_new, solver_parameters=solver_params)
    
        # Non-negativity enforcement (still needed)
        c_data = self.c_new.dat.data
        negative_nodes = c_data < 0
        if negative_nodes.any():
            c_data[negative_nodes] = 0.0
        
        self.c_n.assign(self.c_new)

    def run(self, probe_manager=None, snapshot_manager=None):
        """
        Run coupled flow-transport simulation
        At each timestep:
        1. Solve Richards equation (handled by pressure_solver)
        2. Solve transport equation
        """
        print("Starting coupled flow-transport simulation...")
        print(f"Duration: {self.config.t_end/3600:.1f} hours with dt={self.config.dt}s")
        
        t = 0.0

        if probe_manager is not None:
            probe_manager.record(t, self.c_new, "concentration")
        if snapshot_manager is not None:
            snapshot_manager.record(t, self.c_new, "concentration")
        
        for step in range(self.config.num_steps):
            t += self.config.dt
            
            # Step 1: Solve flow (Richards equation)
            self.pressure_solver.solve_timestep(t)
            
            # Step 2: Solve transport
            self.solve_timestep(t)
            
            # Recording
            if probe_manager is not None:
                probe_manager.record(t, self.c_new, "concentration")
            if snapshot_manager is not None:
                if snapshot_manager.should_record(t, self.config.dt):
                    snapshot_manager.record(t, self.c_new, "concentration")

            # Progress bar
            if step % max(1, int(0.05 * self.config.num_steps)) == 0:
                progress = step / self.config.num_steps
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\rProgress: [{bar}] {progress*100:.1f}% | "
                      f"Time: {t/3600:.1f}h / {self.config.t_end/3600:.1f}h", 
                      end='', flush=True)
        
        print("\n\nCoupled simulation complete!")