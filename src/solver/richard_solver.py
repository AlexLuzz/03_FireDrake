from firedrake import (
    Function, TrialFunction, TestFunction, dx, ds, lhs, rhs, solve,
    as_vector, grad, dot, SpatialCoordinate
)

from ..tools.tools import loading_bar

class RichardsSolver:
    def __init__(self, V, field_map, source_scenario, bc_manager, config, 
                 monitoring, verbose=True):
        
        self.field_map = field_map
        self.domain = self.field_map.domain
        self.mesh = self.domain.mesh
        self.V = V
        
        self.source_scenario = source_scenario
        self.bc_manager = bc_manager
        self.config = config
        self.monitoring = monitoring

        self.verbose = verbose
        self.max_ponding_flux = 1e-6
        self.Se_cutoff = 0.95
        
        self.p_n = Function(self.V, name="Pressure_old")
        self.p_new = Function(self.V, name="Pressure")
        
        self._set_initial_conditions()
    
    def _set_initial_conditions(self):
        """Set initial hydrostatic pressure profile"""        
        water_table = self.bc_manager.left_wt_0
        coords_ufl = SpatialCoordinate(self.mesh)
        
        pressure_expr = water_table - coords_ufl[1]
        self.p_n.interpolate(pressure_expr)
    
    def _get_solver_parameters(self):
        return {
            # Linear solver (no need for Newton since problem is linear)
            "ksp_type": "gmres",
            "ksp_gmres_restart": 30,         # GMRES restart (smaller = less memory, more iterations)
            "ksp_rtol": 1e-4,                # ← Relaxed from 1e-5 (good initial guess)
            "ksp_atol": 1e-6,                # ← Relaxed from 1e-7
            "ksp_max_it": 100,               # ← Reduced from 200 (should converge faster)
            #"ksp_monitor": None,            # Uncomment to see convergence
            
            # Preconditioner - AMG is good for elliptic problems
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "pc_hypre_boomeramg_max_levels": 15,
            "pc_hypre_boomeramg_coarsen_type": "HMIS",  # Aggressive coarsening
            "pc_hypre_boomeramg_strong_threshold": 0.5,  # ← Increased from 0.25 for faster setup
            "pc_hypre_boomeramg_interp_type": "ext+i",
            "pc_hypre_boomeramg_relax_type_all": "symmetric-SOR/Jacobi",
            "pc_hypre_boomeramg_truncfactor": 0.3,
        }
    
    def solve_timestep(self, t: float):
        # Update coefficients from current pressure
        Cm = self.field_map.get_Cm_field(self.p_n)
        K = self.field_map.get_K_field(self.p_n)
        
        bcs = self.bc_manager.get_dirichlet_bcs(t)
        rain_flux = -self.source_scenario.get_flux_expression(t, self.mesh)

        #Se = self.field_map.get_Se_field(self.p_n)
        #rain_flux = conditional(Se >= 0.95, min_value(rain_flux_base, self.max_ponding_flux),
        #    rain_flux_base)
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
              solver_parameters=self._get_solver_parameters())
        
        self.p_n.assign(self.p_new)
    
    def run(self):
        if self.verbose:
            print("Starting simulation...")
            print(f"Duration: {self.config.t_end/3600:.1f} hours with dt={self.config.dt}s")
        
        self.monitoring.record_probe(0.0, self.p_n, "water_table")
        self.monitoring.check_and_record_snapshot(0.0, self.config.dt, self.field_map.get_Se_field(self.p_n), "saturation")

        t = 0.0
        for step in range(self.config.num_steps):
            t += self.config.dt
            self.solve_timestep(t)
            
            self.monitoring.record_probe(t, self.p_new, "water_table")
            self.monitoring.check_and_record_snapshot(t, self.config.dt, self.field_map.get_Se_field(self.p_new), "saturation")

            # Loading bar
            if self.verbose:
                loading_bar(step, t, self.config)
        
        if self.verbose:
            print("\n\nSimulation complete!")