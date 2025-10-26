from firedrake import (
    Function, TrialFunction, TestFunction, dx, ds, lhs, rhs, solve,
    as_vector, grad, dot
)

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
        
        # Get Ks once (static)
        self.Ks_field = field_map.get_Ks_field()
        
        self._set_initial_conditions()
    
    def _set_initial_conditions(self):
        coords = self.mesh.coordinates.dat.data_ro
        water_table = self.bc_manager.H0_initial
        self.p_n.dat.data[:] = water_table - coords[:, 1]
    
    def solve_timestep(self, t: float):
        # Update coefficients from current pressure
        Cm_n = self.field_map.get_Cm_field(self.p_n)
        kr_n = self.field_map.get_kr_field(self.p_n)
        
        bcs = self.bc_manager.get_dirichlet_bcs(t)
        rain_flux = -self.source_scenario.get_flux_expression(t, self.mesh)

        p = TrialFunction(self.V)
        q = TestFunction(self.V)

        K_field = Function(self.V)
        K_field.dat.data[:] = kr_n.dat.data_ro * self.Ks_field.dat.data_ro
        
        gravity = as_vector([0, 1])

        F = (
            Cm_n * (p - self.p_n) / self.config.dt * q * dx +
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
            
            # Record probes (generic + specialized)
            if probe_manager is not None:
                probe_manager.record(t, self.p_new, "pressure")
                probe_manager.record_water_table(t, self.p_new)
            
            # Record snapshots
            if snapshot_manager is not None:
                if snapshot_manager.should_record(t, self.config.dt):
                    snapshot_manager.record(t, self.p_new, "pressure")
                    
                    # Also record saturation
                    Se = self.field_map.compute_saturation_field(self.p_new)
                    snapshot_manager.record(t, Se, "saturation")

            if step % max(1, int(0.05 * self.config.num_steps)) == 0:
                progress = step / self.config.num_steps
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\rProgress: [{bar}] {progress*100:.1f}% | "
                      f"Time: {t/3600:.1f}h / {self.config.t_end/3600:.1f}h", 
                      end='', flush=True)
        
        print("\n\nSimulation complete!")