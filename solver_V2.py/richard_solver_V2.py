"""
Richards equation solver using new architecture
"""
from firedrake import (
    Function, TrialFunction, TestFunction, Constant, SpatialCoordinate,
    dx, ds, lhs, rhs, solve, conditional, And, as_vector, grad, dot,
    RectangleMesh, FunctionSpace
)
import numpy as np

class RichardsSolver:
    """
    Richards equation solver: Cm * ∂Hp/∂t - ∇·(K * ∇Hp) = sources
    """

    def __init__(self, mesh, V, field_map, source_scenario, bc_manager, config):
        """
        Args:
            mesh: Firedrake mesh
            V: Function space
            field_map: MaterialField object (links domain to materials)
            source_scenario: SourceScenario (rain/sources)
            bc_manager: BoundaryConditionManager
            config: SimulationConfig
        """
        self.mesh = mesh
        self.V = V
        self.field_map = field_map
        self.domain = field_map.domain
        self.source_scenario = source_scenario
        self.bc_manager = bc_manager
        self.config = config
        
        # Solution functions
        self.p_n = Function(V, name="Pressure_old")
        self.p_new = Function(V, name="Pressure")
        
        # Coefficient fields
        self.Cm_n = Function(V, name="Moisture_capacity")
        self.kr_n = Function(V, name="Relative_permeability")
        self.Ks_field = Function(V, name="Saturated_conductivity")
        
        # Initialize Ks (static, computed once)
        self._set_Ks_field()
        
        # Initialize pressure
        self._set_initial_conditions()
    
    def _set_Ks_field(self):
        """Set saturated conductivity field (static)"""
        coords = self.mesh.coordinates.dat.data
        Ks_vals = np.zeros(len(coords))
        
        for i, (x, y) in enumerate(coords):
            material = self._get_material_at_point(x, y)
            Ks_vals[i] = material.Ks
        
        self.Ks_field.dat.data[:] = Ks_vals
    
    def _set_initial_conditions(self):
        """Set initial hydrostatic pressure"""
        coords = self.mesh.coordinates.dat.data
        y_coords = coords[:, 1]
        water_table = self.bc_manager.H0_initial
        
        initial_pressure = water_table - y_coords
        self.p_n.dat.data[:] = initial_pressure
        
        self._update_coefficients()
    
    def _get_material_at_point(self, x: float, y: float):
        """Get material at spatial point using field_map"""
        # Find closest grid point in domain
        i = np.argmin(np.abs(self.domain.x - x))
        j = np.argmin(np.abs(self.domain.y - y))
        return self.field_map.get_material_at(j, i)  # Note: j, i for (ny, nx) indexing
    
    def _update_coefficients(self):
        """Update Cm and kr based on current pressure"""
        coords = self.mesh.coordinates.dat.data
        p_vals = self.p_n.dat.data[:]
        
        Cm_vals = np.zeros(len(coords))
        kr_vals = np.zeros(len(coords))
        
        for i, (x, y) in enumerate(coords):
            material = self._get_material_at_point(x, y)
            Hp = p_vals[i]
            
            Cm_vals[i] = material.dtheta_dp(Hp)
            kr_vals[i] = material.kr(Hp)
        
        self.Cm_n.dat.data[:] = Cm_vals
        self.kr_n.dat.data[:] = kr_vals

    def solve_timestep(self, t: float):
        """Solve one time step"""
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
        
        if probe_manager is not None:
            probe_data = probe_manager.get_data()
            print(f"  Total time steps: {len(probe_data['times'])}")
            for name, data in probe_data['data'].items():
                try:
                    data_array = np.array(data, dtype=float)
                    valid_data = data_array[~np.isnan(data_array)]
                    if len(valid_data) > 0:
                        print(f"  {name}: {valid_data.min():.3f}m - {valid_data.max():.3f}m")
                except (ValueError, TypeError):
                    pass
