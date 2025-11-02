"""
Advection-Dispersion transport solver
Solves: ∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S
"""
from firedrake import (
    Function, TrialFunction, TestFunction, dx, ds, lhs, rhs, solve,
    as_matrix, sqrt, grad, dot, project, VectorFunctionSpace, Constant, assemble, 
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
    
    def compute_darcy_velocity(self, pressure, K):
        """
        Compute Darcy velocity from pressure field: v = -K∇(h + z)
        --------
        For Richards equation with pressure head p [m]:
        - Hydraulic head: h = p (pressure already in head units)
        - Total head including gravity: H = p + z
        - Darcy velocity: v = -K∇H = -K(∇p + ∇z)
        
        In 2D (y = vertical coordinate):
        - v_x = -K ∂p/∂x
        - v_y = -K (∂p/∂y + 1)  [the +1 is ∂z/∂y = 1]
        """
        # Compute velocity components as UFL expressions
        vx_expr = -K * grad(pressure)[0]
        vy_expr = -K * (grad(pressure)[1] + 1.0)
        
        # Project to scalar function space
        vx = project(vx_expr, self.V)
        vy = project(vy_expr, self.V)
        
        # Also update the vector field for advection term
        self.velocity.dat.data[:, 0] = np.maximum(vx.dat.data_ro, 1e-14)
        self.velocity.dat.data[:, 1] = np.maximum(vy.dat.data_ro, 1e-14)

        return vx, vy

    def assemble_dispersion_tensor(self, vx, vy, D_0,
                                  alpha_T, alpha_L,
                                  cross_dispersion=False):
        """
        D: Hydrodynamic dispersion tensor [m²/s]
        Construct the anisotropic dispersion tensor (2D)
        D_ij = D0*δ_ij + |v| * [α_T*δ_ij + (α_L - α_T)*(v_i*v_j / |v|^2)]
        """
        v_mag = sqrt(vx**2 + vy**2) + 1e-14  # Avoid division by zero

        dalpha = alpha_L - alpha_T
        D_xx = D_0 + v_mag * (alpha_T + dalpha * (vx**2) / (v_mag**2))
        D_yy = D_0 + v_mag * (alpha_T + dalpha * (vy**2) / (v_mag**2))
        if cross_dispersion:
            D_xy = v_mag * dalpha * (vx * vy) / (v_mag**2)
            D_tensor = as_matrix([[D_xx, D_xy],
                                  [D_xy, D_yy]])
        else:
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
        pressure = self.pressure_solver.p_n
        K = self.field_map.get_K_field(pressure)
        vx, vy = self.compute_darcy_velocity(pressure, K)
        
        # Step 2: Get transport coefficients from current pressure
        D_0 = self.field_map.get_D0_field(pressure)
        alpha_T = self.field_map.get_alpha_T_field()
        alpha_L = self.field_map.get_alpha_L_field()

        D_eff = self.assemble_dispersion_tensor(vx, vy, D_0, alpha_T, alpha_L)

        # Step 3: Get source
        source_expr = self.transport_source.get_flux_expression(t, self.mesh)
        
        # Step 4: Solve transport equation
        # ∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S
        # With retardation: R·θ·∂c/∂t + ∇·(vc) = ∇·(D∇c) + S
        
        c = TrialFunction(self.V)
        q = TestFunction(self.V)
        
        # Effective storage term (accounts for retardation)
        #storage_coeff = R * theta
        
        # Time discretization parameter
        dt = Constant(self.config.dt)
        
        # (1) Time derivative term
        F = (c - self.c_n) / dt * q * dx

        # (2) Advection term (conservative form)
        # integrate ∇·(v c) by parts → -∫ c v·∇q + ∫_Γ q c v·n
        F += - dot(self.velocity, grad(q)) * c * dx

        # (3) Dispersion–diffusion term
        F += dot(dot(D_eff, grad(c)), grad(q)) * dx

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
    
    def compute_mass_balance(self):
        """
        Compute total mass in the system for mass balance checking
        M = ∫ θ c dV
        """
        pressure = self.pressure_solver.p_n
        theta = self.field_map.get_theta_field(pressure)
        total_mass = assemble(theta * self.c_new * self.dx)
        return total_mass
    
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

            m_n = self.compute_mass_balance()

            # Step 2: Solve transport
            self.solve_timestep(t)

            m_new = self.compute_mass_balance()
            
            # Recording
            if probe_manager is not None:
                probe_manager.record(t, self.c_new, "concentration")
                probe_manager.record(t, data=m_new - m_n, field_name="mass_loss")

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