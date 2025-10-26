"""
Advection-Dispersion transport solver
Solves: ∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S
"""
from firedrake import (
    Function, TrialFunction, TestFunction, dx, ds, lhs, rhs, solve,
    as_vector, grad, dot, div, project, VectorFunctionSpace
)
import numpy as np

class TransportSolver:
    def __init__(self, domain, V, field_map, pressure_solver, 
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
        pressure_solver : RichardsSolver
            Richards solver (provides pressure field)
        bc_manager : BCManager
            Boundary condition manager
        transport_source : TransportSource
            Source/sink terms for contaminant
        config : Config
            Simulation configuration
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
        
        # Concentration fields
        self.c_n = Function(V, name="Concentration_old")
        self.c_new = Function(V, name="Concentration")
        
        # Velocity field (computed from pressure)
        self.velocity = Function(self.V_vec, name="Darcy_velocity")
        
        # Static fields
        self.Ks_field = field_map.get_Ks_field()
        
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
        v = -K(∇p + ∇z) where z is elevation (vertical coordinate)
        """
        pressure = self.pressure_solver.p_n
        
        # Get hydraulic conductivity field
        kr = self.field_map.get_kr_field(pressure)
        K_field = Function(self.V)
        K_field.dat.data[:] = kr.dat.data_ro * self.Ks_field.dat.data_ro
        
        # Compute v = -K(∇p + [0, 1]) in each direction
        # Note: gravity vector is [0, 1] pointing up
        vx = project(-K_field * grad(pressure)[0], self.V)
        vy = project(-K_field * (grad(pressure)[1] + 1.0), self.V)
        
        # Assemble into vector field
        self.velocity.dat.data[:, 0] = vx.dat.data_ro
        self.velocity.dat.data[:, 1] = vy.dat.data_ro
    
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
        R = self.field_map.get_retardation_field(pressure)
        
        # Get dispersion coefficients
        vx = Function(self.V)
        vy = Function(self.V)
        vx.dat.data[:] = self.velocity.dat.data_ro[:, 0]
        vy.dat.data[:] = self.velocity.dat.data_ro[:, 1]
        #D_L, D_T = self.field_map.get_dispersion_field(pressure, vx, vy)
        D_L, D_T = 0.01, 0.005  # Simplified constant values for demo

        # Step 3: Get boundary conditions and sources
        #bcs = self.bc_manager.get_dirichlet_bcs(t)
        source_term = self.transport_source.get_flux_expression(t, self.mesh)
        
        # Step 4: Solve transport equation
        # ∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S
        # With retardation: R·θ·∂c/∂t + ∇·(vc) = ∇·(D∇c) + S
        
        c = TrialFunction(self.V)
        q = TestFunction(self.V)
        
        # Effective storage term (accounts for retardation)
        storage_coeff = R * theta
        
        # Weak form
        F = (
            # Time derivative: R·θ·(c - c_n)/dt
            storage_coeff * (c - self.c_n) / self.config.dt * q * dx +
            # Advection: ∇·(vc) → -v·∇c (integration by parts)
            - dot(self.velocity, grad(q)) * c * dx +
            # Dispersion: ∇·(D∇c) → D∇c·∇q (integration by parts)
            # Using isotropic approximation: D ≈ D_L (simplification)
            D_L * dot(grad(c), grad(q)) * dx +
            # Source term (negative because of integration by parts convention)
            - source_term * q * dx
        )
        
        a = lhs(F)
        L = rhs(F)
        
        # Solve
        solve(a == L, self.c_new, bcs=[],
              solver_parameters=self.config.solver_parameters)
        
        # Update
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