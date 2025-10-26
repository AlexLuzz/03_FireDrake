"""
Richards equation solver using new architecture
"""
from firedrake import (
    Function, TrialFunction, TestFunction, dx, ds, lhs, rhs, solve,
    as_vector, grad, dot
)
import numpy as np

class RichardsSolver:
    def __init__(self, mesh, V, field_map, source_scenario, bc_manager, config):
        self.mesh = mesh
        self.V = V
        self.field_map = field_map
        self.source_scenario = source_scenario
        self.bc_manager = bc_manager
        self.config = config
        
        self.p_n = Function(V, name="Pressure_old")
        self.p_new = Function(V, name="Pressure")
        self.Cm_n = Function(V, name="Moisture_capacity")
        self.kr_n = Function(V, name="Relative_permeability")
        self.Ks_field = Function(V, name="Saturated_conductivity")
        
        self._set_Ks_field()
        self._set_initial_conditions()
    
    def _set_Ks_field(self):
        """Set Ks (static, once)"""
        coords = self.mesh.coordinates.dat.data
        Ks_vals = np.zeros(len(coords))
        
        for i, (x, y) in enumerate(coords):
            material = self.field_map.get_material_at_point(x, y)
            Ks_vals[i] = material.Ks
        
        self.Ks_field.dat.data[:] = Ks_vals
    
    def _set_initial_conditions(self):
        coords = self.mesh.coordinates.dat.data
        water_table = self.bc_manager.H0_initial
        self.p_n.dat.data[:] = water_table - coords[:, 1]
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Update Cm and kr - ONE method call to field_map!"""
        coords = self.mesh.coordinates.dat.data
        p_vals = self.p_n.dat.data[:]
        
        Cm_vals = np.zeros(len(coords))
        kr_vals = np.zeros(len(coords))
        
        for i, (x, y) in enumerate(coords):
            material = self.field_map.get_material_at_point(x, y)
            Cm_vals[i] = material.dtheta_dp(p_vals[i])
            kr_vals[i] = material.kr(p_vals[i])
        
        self.Cm_n.dat.data[:] = Cm_vals
        self.kr_n.dat.data[:] = kr_vals

    def solve_timestep(self, t: float):
        """Solve one timestep - pure numerical work, no material logic"""
        self._update_coefficients()
        
        bcs = self.bc_manager.get_dirichlet_bcs(t)
        rain_flux = -self.source_scenario.get_flux_expression(t, self.mesh)

        p = TrialFunction(self.V)
        q = TestFunction(self.V)

        K_field = self.kr_n * self.Ks_field
        gravity = as_vector([0, 1])

        F = (
            self.Cm_n * (p - self.p_n) / self.config.dt * q * dx +
            K_field * dot(grad(p), grad(q)) * dx +
            K_field * dot(gravity, grad(q)) * dx +
            rain_flux * q * ds(4)
        )

        a = lhs(F)
        L = rhs(F)
        
        solve(a == L, self.p_new, bcs=bcs,
              solver_parameters=self.config.solver_parameters)

        self.p_n.assign(self.p_new)
    
    def run(self, probe_manager=None, snapshot_manager=None):
        """Run full simulation"""
        print("Starting simulation...")
        print(f"Duration: {self.config.t_end/3600:.1f} hours with dt={self.config.dt}s")
        
        if probe_manager is not None:
            probe_manager.record_initial(self.p_n)
        if snapshot_manager is not None:
            snapshot_manager.record_initial(self.p_n)

        t = 0.0
        for step in range(self.config.num_steps):
            t += self.config.dt
            self.solve_timestep(t)
            
            if probe_manager is not None:
                probe_manager.record(t, self.p_new)
            if snapshot_manager is not None:
                if snapshot_manager.should_record(t, self.config.dt):
                    snapshot_manager.record(t, self.p_new)

            if step % max(1, int(0.05 * self.config.num_steps)) == 0:
                progress = step / self.config.num_steps
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\rProgress: [{bar}] {progress*100:.1f}% | "
                      f"Time: {t/3600:.1f}h / {self.config.t_end/3600:.1f}h", 
                      end='', flush=True)
        
        print("\n\nSimulation complete!")