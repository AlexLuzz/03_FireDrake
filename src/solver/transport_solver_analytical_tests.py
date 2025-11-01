"""
Advection-Dispersion transport solver
Solves: ∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S
"""
from firedrake import (
    Function, TrialFunction, TestFunction, dx, lhs, rhs, solve,
    as_matrix, grad, dot, VectorFunctionSpace, Constant, assemble, 
)
import numpy as np
from ..tools.tools import loading_bar

class TransportSolverAnalytical:
    def __init__(self, domain, V, field_map, 
                 bc_manager, transport_source, config):
        """
        Parameters:
        -----------
        domain : Domain
            Computational domain
        V : FunctionSpace
            Function space for concentration
        field_map : MaterialField
            Field mapper (must have transport models)
        bc_manager : BCManager
            Boundary condition manager
        transport_source : TransportSource
            Source/sink terms for contaminant
        config : Config
            Simulation configuration
        """
        self.mesh = domain.mesh
        self.V = V
        self.V_vec = VectorFunctionSpace(self.mesh, "CG", 1)  # For velocity

        self.field_map = field_map
        self.bc_manager = bc_manager
        self.transport_source = transport_source
        
        self.config = config
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
    
    def compute_mass_balance(self, theta=None):
        """
        Compute total mass in the system for mass balance checking
        M = ∫ θ c dV
        """
        theta = Constant(theta)
        total_mass = assemble(theta * self.c_new * self.dx)
        return total_mass
    
    def solve_timestep_params(self, t: float, params: dict):
        """
        Solver with prescribed parameters
        
        Args:
            t: Current time
            params: Dict with:
                'porosity': float - Porosity [-] (optional, default=1.0)
                'vx': float - x-velocity [m/s]
                'vy': float - y-velocity [m/s]
                'Dxx': float - Dispersion tensor xx component [m²/s]
                'Dyy': float - Dispersion tensor yy component [m²/s]
                'Dxy': float - Dispersion tensor xy component [m²/s] (optional, default=0)
                'R': float - Retardation factor [-] (optional, default=1.0)
                'lambda': float - Decay rate [1/s] (optional, default=0.0)
        """
        # Set velocity
        vx = params['vx']
        vy = params['vy']
        self.velocity.dat.data[:, 0] = vx
        self.velocity.dat.data[:, 1] = vy
        
        # Build dispersion tensor
        Dxx = Constant(params['Dxx'])
        Dyy = Constant(params['Dyy'])
        Dxy = Constant(params.get('Dxy', 0.0))
        D_eff = as_matrix([[Dxx, Dxy], [Dxy, Dyy]])
        
        # Get optional parameters
        R = params.get('R', 1.0)
        decay = params.get('lambda', 0.0)
        
        self._solve_transport_equation(t, D_eff, R=R, decay=decay)
    
    def _solve_transport_equation(self, t: float, D_eff, R: float = 1.0, decay: float = 0.0):
        """
        Core solver logic
        
        Solves: R·∂c/∂t + ∇·(vc) = ∇·(D∇c) - λc + S
        
        Args:
            t: Current time
            D_eff: Dispersion tensor (UFL expression)
            R: Retardation factor
            decay: First-order decay rate [1/s]
        """
        source_expr = self.transport_source.get_flux_expression(t, self.mesh)
        
        c = TrialFunction(self.V)
        q = TestFunction(self.V)
        dt = Constant(self.config.dt)
        R_const = Constant(R)
        lambda_const = Constant(decay)
        
        # Weak form with retardation and decay
        # R·(c - c_n)/dt + ∇·(vc) = ∇·(D∇c) - λc + S
        F = R_const * (c - self.c_n) / dt * q * dx
        F += -dot(self.velocity, grad(q)) * c * dx
        F += dot(dot(D_eff, grad(c)), grad(q)) * dx
        F += lambda_const * c * q * dx  # Decay term
        F += -source_expr * q * dx
        
        a, L = lhs(F), rhs(F)
        
        solve(a == L, self.c_new, bcs=[],
              solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu',
                                'pc_factor_mat_solver_type': 'mumps'})
        
        # Enforce non-negativity
        c_data = self.c_new.dat.data
        if (c_data < 0).any():
            c_data[c_data < 0] = 0.0
        
        self.c_n.assign(self.c_new)
    
    def run(self, probe_manager=None, snapshot_manager=None, params=None):
        """
        Run simulation
        
        Args:
            probe_manager: ProbeManager instance
            snapshot_manager: SnapshotManager instance
            params: If provided, use solve_timestep_params instead of solve_timestep
        """
        print(f"Duration: {self.config.t_end/3600:.1f} hours with dt={self.config.dt}s")
        print(f"  vx={params['vx']:.2e} m/s, vy={params['vy']:.2e} m/s")
        print(f"  Dxx={params['Dxx']:.2e} m²/s, Dyy={params['Dyy']:.2e} m²/s")
        if params.get('R', 1.0) != 1.0:
            print(f"  Retardation R={params['R']:.2f}")
        if params.get('lambda', 0.0) != 0.0:
            print(f"  Decay λ={params['lambda']:.2e} 1/s")
        
        t = 0.0
        
        # Initial recording
        if probe_manager:
            probe_manager.record(t, self.c_new, "concentration")
        if snapshot_manager:
            snapshot_manager.record(t, self.c_new, "concentration")
        
        mass_residual_balance = []
        
        for step in range(self.config.num_steps):
            t += self.config.dt
                        
            # Solve transport
            m_n = self.compute_mass_balance(theta=params.get('porosity', 1.0)) # saturated conditions theta=porosity
            self.solve_timestep_params(t, params)

            m_new = self.compute_mass_balance()
            mass_residual_balance.append(m_new - m_n)
            
            # Recording
            if probe_manager:
                probe_manager.record(t, self.c_new, "concentration")
            if snapshot_manager:
                if snapshot_manager.should_record(t, self.config.dt):
                    snapshot_manager.record(t, self.c_new, "concentration")
            
            loading_bar(step, t, self.config)
        
        print(f"\n\nCoupled simulation complete!")
        print(f"Mass balance check: mean residual = {np.mean(mass_residual_balance):.2e}")