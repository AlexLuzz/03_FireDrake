from firedrake import (
    Function, TrialFunction, TestFunction, dx, ds, lhs, rhs, solve,
    as_vector, grad, dot
)

from ..tools.tools import loading_bar, fancy_loading_bar

class RichardsSolver:
    def __init__(self, domain, V, field_map, source_scenario, bc_manager, config):
        self.mesh = domain.mesh  # Use domain's mesh
        self.V = V
        self.field_map = field_map
        self.source_scenario = source_scenario
        self.bc_manager = bc_manager
        self.config = config
        
        self.p_n = Function(V, name="Pressure_old")
        self.p_new = Function(V, name="Pressure")
        
        self._set_initial_conditions()
    
    def _set_initial_conditions(self):
        coords = self.mesh.coordinates.dat.data_ro
        water_table = self.bc_manager.left_wt_0
        self.p_n.dat.data[:] = water_table - coords[:, 1]
    
    def solve_timestep(self, t: float):
        # Update coefficients from current pressure
        Cm = self.field_map.get_Cm_field(self.p_n)
        K = self.field_map.get_K_field(self.p_n)

        bcs = self.bc_manager.get_dirichlet_bcs(t)
        rain_flux = -self.source_scenario.get_flux_expression(t, self.mesh)

        p = TrialFunction(self.V)
        q = TestFunction(self.V)

        gravity = as_vector([0, 1])

        F = (
            Cm * (p - self.p_n) / self.config.dt * q * dx +
            K * dot(grad(p), grad(q)) * dx +
            K * dot(gravity, grad(q)) * dx +
            rain_flux * q * ds(4)
        )

        a = lhs(F)
        L = rhs(F)
        
        solve(a == L, self.p_new, bcs=bcs,
              solver_parameters=self.config.solver_parameters)

        self.p_n.assign(self.p_new)
    
    def run(self, probe_manager=None, snapshot_manager=None):
        print("Starting simulation...")
        print(f"Duration: {self.config.t_end/3600:.1f} hours with dt={self.config.dt}s")
        
        # Record initial conditions (t=0)
        if probe_manager is not None:
            probe_manager.record(0.0, self.p_n, "pressure")
            probe_manager.record_water_table(0.0, self.p_n)
        if snapshot_manager is not None:
            if snapshot_manager.should_record(0.0, self.config.dt):
                Se = self.field_map.get_Se_field(self.p_n)
                snapshot_manager.record(0.0, Se, "saturation", verbose=False)

        t = 0.0
        for step in range(self.config.num_steps):
            t += self.config.dt
            self.solve_timestep(t)
            
            # Record probes (generic + specialized)
            if probe_manager is not None:
                probe_manager.record(t, self.p_new, "pressure")
                probe_manager.record_water_table(t, self.p_new)
            
            # Record snapshots
            if snapshot_manager is not None:
                if snapshot_manager.should_record(t, self.config.dt):
                    Se = self.field_map.get_Se_field(self.p_new)
                    snapshot_manager.record(t, Se, "saturation", verbose=False)

            # Loading bar
            loading_bar(step, t, self.config)
            #
        
        print("\n\nSimulation complete!")