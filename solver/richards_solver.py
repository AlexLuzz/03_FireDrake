"""
Main Richards equation solver
"""
from firedrake import (Function, TrialFunction, TestFunction, 
                       Constant, dx, ds, grad, dot, lhs, rhs, solve)
import numpy as np

class RichardsSolver:
    """
    Richards equation solver using Firedrake
    Implements: Cm * ∂Hp/∂t - ∇·(kr * Ks * ∇Hp) = rain_flux
    """
    
    def __init__(self, mesh, V, domain, bc_manager, config):
        """
        Initialize solver
        
        Args:
            mesh: Firedrake mesh
            V: Function space
            domain: Domain object with material properties
            bc_manager: BoundaryConditionManager
            config: SimulationConfig
        """
        self.mesh = mesh
        self.V = V
        self.domain = domain
        self.bc_manager = bc_manager
        self.config = config
        
        # Solution functions
        self.p_n = Function(V, name="Pressure_old")
        self.p_new = Function(V, name="Pressure")
        
        # Coefficient fields
        self.Cm_n = Function(V, name="Moisture_capacity")
        self.kr_n = Function(V, name="Relative_permeability")
        
        # Ks as Firedrake constant
        self.Ks = Constant(domain.default_material.Ks)
        
        # Initialize
        self._set_initial_conditions()
    
    def _set_initial_conditions(self):
        """Set initial hydrostatic pressure distribution"""
        coords = self.mesh.coordinates.dat.data
        y_coords = coords[:, 1]
        water_table = self.config.initial_water_table
        
        initial_pressure = np.zeros(len(y_coords))
        for i, y in enumerate(y_coords):
            if y <= water_table:
                # Below water table: positive hydrostatic pressure
                initial_pressure[i] = water_table - y
            else:
                # Above water table: negative pressure (suction)
                initial_pressure[i] = -(y - water_table) * 2.0
        
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
        
        # Get rain flux expression
        rain_flux = self.bc_manager.get_rain_flux_expression(t)
        
        # Define variational problem
        p = TrialFunction(self.V)
        q = TestFunction(self.V)
        
        # Weak form: Cm*(p-p_n)/dt + kr*Ks*∇p·∇q + flux*q on boundary
        F = (self.Cm_n * (p - self.p_n) / self.config.dt * q * dx +
             self.kr_n * self.Ks * dot(grad(p), grad(q)) * dx +
             rain_flux * q * ds(4))  # ds(4) is top boundary
        
        a = lhs(F)
        L = rhs(F)
        
        # Solve
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
        print(f"Domain: {self.config.Lx}m x {self.config.Ly}m")
        print(f"Mesh: {self.config.nx} x {self.config.ny} elements")
        print(f"Time: {self.config.t_end/3600:.1f} hours with dt={self.config.dt}s")
        print(f"Rain event: {self.config.rain_start/3600:.1f}h to {self.config.rain_end/3600:.1f}h")
        print(f"Material: {self.domain.default_material.name}")
        print(f"  Ks = {self.domain.default_material.Ks:.2e} m/s")
        print(f"  VG params: α={self.domain.default_material.hydraulic_model.params.alpha}, "
              f"n={self.domain.default_material.hydraulic_model.params.n}")
        print()
        
        t = 0.0
        for step in range(self.config.num_steps):
            t += self.config.dt
            
            # Solve time step
            self.solve_timestep(t)
            
            # Record at monitoring points
            if probe_manager is not None:
                probe_manager.record(t, self.p_new)
            
            # Save snapshots
            if snapshot_manager is not None:
                if snapshot_manager.should_record(t, self.config.dt):
                    snapshot_manager.record(t, self.p_new)
            
            # Print progress every hour
            if step % int(3600/self.config.dt) == 0:
                print(f"Time: {t/3600:.1f}h / {self.config.t_end/3600:.1f}h")
        
        print("\nSimulation complete!")
    
    def get_current_solution(self):
        """Get current pressure solution"""
        return self.p_new