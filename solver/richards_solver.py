"""
Main Richards equation solver
"""
from firedrake import *
import numpy as np

class RichardsSolver:
    """
    Richards equation solver using Firedrake
    Implements: Cm * ∂Hp/∂t - ∇·(kr * Ks * ∇Hp) = rain_flux
    """

    def __init__(self, mesh, V, domain, rain_scenario, bc_manager, config):
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
        self.rain_scenario = rain_scenario
        self.bc_manager = bc_manager
        self.config = config
        
        # Solution functions
        self.p_n = Function(V, name="Pressure_old")
        self.p_new = Function(V, name="Pressure")
        
        # Coefficient fields
        self.Cm_n = Function(V, name="Moisture_capacity")
        self.kr_n = Function(V, name="Relative_permeability")
        
        # Ks field (spatially varying for heterogeneous domains)
        self.Ks_field = domain.compute_Ks_field(V)
        
        # Initialize
        self._set_initial_conditions()
    
    def _set_initial_conditions(self):
        """Set initial hydrostatic pressure distribution"""
        coords = self.mesh.coordinates.dat.data
        y_coords = coords[:, 1]
        water_table = self.config.initial_water_table
        
        initial_pressure = np.zeros(len(y_coords))
        for i, y in enumerate(y_coords):
            # True hydrostatic equilibrium: p = z_wt - z
            initial_pressure[i] = water_table - y
        
        self.p_n.dat.data[:] = initial_pressure
        
        # Initialize coefficient fields
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Update Cm and kr based on current pressure"""
        self.Cm_n, self.kr_n = self.domain.compute_coefficient_fields(self.p_n)
    
    def get_rain_flux_expression(self, t: float):
        """
        Get spatially-varying rain flux expression for current time
        Rain is applied as Neumann BC (flux) on top boundary
        
        Args:
            t: Current time (seconds)
        
        Returns:
            UFL expression for rain flux (m/s, negative = into domain)
        """        
        t_hours = t / 3600.0
        coords = SpatialCoordinate(self.mesh)
        x = coords[0]
        
        # Build piecewise expression for all zones
        flux_expr = Constant(0.0)
        
        for event in self.rain_scenario.events:
            if event.is_active(t_hours):
                # Event is active - check each zone
                for zone in event.zones:
                    # Calculate flux for this zone
                    zone_flux = event.intensity * zone.multiplier / 3600000.0  # mm/hr to m/s
                    
                    # Add contribution if x is in zone
                    flux_expr = conditional(
                        And(x >= zone.x_min, x <= zone.x_max),
                        Constant(-zone_flux),  # Negative = into domain
                        flux_expr  # Keep previous value if not in zone
                    )
        
        return flux_expr

    def solve_timestep(self, t: float):
        """
        Solve one time step
        
        Args:
            t: Current time (seconds)
            bcs: Dirichlet boundary conditions (optional)
        """
        # Update coefficients from previous solution
        self._update_coefficients()
        
        # Get boundary conditions
        bcs = self.bc_manager.get_dirichlet_bcs()
        
        # Get rain flux expression
        rain_flux = self.get_rain_flux_expression(t)
        
        # Define variational problem
        p = TrialFunction(self.V)  # Unknown pressure head (to solve for)
        q = TestFunction(self.V)   # Test function (mathematical trick for FEM)

        # Compute K field (K = kr * Ks)
        # Linearization: use K from previous time step (known values)
        K_field = self.kr_n * self.Ks_field

        # Gravity vector (vertical direction)
        gravity = as_vector([0, 1])  # ∇z = (0, 1) since z = y

        # Weak form of Richards equation: C·∂Hp/∂t = ∇·[K(∇Hp + ∇z)]
        # After integration by parts and time discretization:
        F = (self.Cm_n * (p - self.p_n) / self.config.dt * q * dx +  # Time derivative: C(Hp-Hp_n)/Δt
            K_field * dot(grad(p), grad(q)) * dx +                   # Diffusion: K∇Hp·∇q
            K_field * dot(gravity, grad(q)) * dx +                   # Gravity: K∇z·∇q
            rain_flux * q * ds(4))  # Infiltration source on top boundary

        # Split residual form F=0 into linear system a=L
        # lhs: extracts terms with unknown p → matrix A
        # rhs: extracts known terms (p_n, sources) → vector b
        a = lhs(F)  # Bilinear form: a(p,q) - contains both p and q
        L = rhs(F)  # Linear form: L(q) - contains only q (negated from F)

        # Solve linear system A·p_new = b
        # The nonlinear problem is linearized using known values (Cm_n, K_field)
        solve(a == L, self.p_new, bcs=bcs,
            solver_parameters=self.config.solver_parameters)

        # Update for next time step
        self.p_n.assign(self.p_new)
    
    def run(self, probe_manager=None, snapshot_manager=None, print_diagnostics=False):
        """
        Run full simulation
        
        Args:
            probe_manager: ProbeManager for time series (optional)
            snapshot_manager: SnapshotManager for spatial data (optional)
            print_diagnostics: Print detailed diagnostics every hour (optional)
        """
        print("Starting simulation...")
        print(f"Domain: {self.config.Lx}m x {self.config.Ly}m")
        print(f"Mesh: {self.config.nx} x {self.config.ny} elements")
        print(f"Duration: {self.config.t_end/3600:.1f} hours with dt={self.config.dt}s")
        
        probe_manager.record_initial(self.p_n)
        snapshot_manager.record_initial(self.p_n)

        t = 0.0
        for step in range(self.config.num_steps):
            t += self.config.dt
            
            # Solve time step
            self.solve_timestep(t)
            
            # Record at monitoring points
            probe_manager.record(t, self.p_new)
            
            # Save snapshots
            if snapshot_manager.should_record(t, self.config.dt):
                snapshot_manager.record(t, self.p_new)
            
            # Print progress every hour
            if step % int(3600/self.config.dt) == 0:
                print(f"Time: {t/3600:.1f}h / {self.config.t_end/3600:.1f}h")
                
                # Detailed diagnostics if requested
                if print_diagnostics:
                    self.print_diagnostics(t)
                    print()
        
        print("\nSimulation complete!")
        
        # Print probe data summary
        probe_data = probe_manager.get_data()
        print(f"  Total time steps recorded: {len(probe_data['times'])}")
        print(f"  Time range: {probe_data['times'][0]/3600:.2f}h - {probe_data['times'][-1]/3600:.2f}h")
        for name, data in probe_data['data'].items():
            data_array = np.array(data)
            valid_data = data_array[~np.isnan(data_array)]
            if len(valid_data) > 0:
                print(f"  {name}: {valid_data.min():.3f}m - {valid_data.max():.3f}m ({len(valid_data)}/{len(data_array)} valid)")
            else:
                print(f"  {name}: ALL NaN!")

        # Print snapshot summary
        print(f"  Total snapshots recorded: {len(snapshot_manager.snapshots)}")
        for t in sorted(snapshot_manager.snapshots.keys()):
            print(f"    t = {t/3600:.2f}h")

    
    def compute_total_water_content(self):
        """
        Compute total water in domain (for mass balance checks)
        
        Returns:
            Total water content (m³)
        """
        coords = self.mesh.coordinates.dat.data
        p_vals = self.p_new.dat.data[:]
        
        total_theta = 0.0
        for i, (x, y) in enumerate(coords):
            Hp = p_vals[i]
            material = self.domain.get_material_at_point(x, y)
            theta = material.water_content(Hp)
            total_theta += theta
        
        # Approximate volume per node
        dx = self.config.dx
        dy = self.config.dy
        volume_per_node = dx * dy
        
        return total_theta * volume_per_node
    
    def print_diagnostics(self, t: float):
        """Print diagnostic information"""
        p_vals = self.p_new.dat.data[:]
        print(f"  Pressure range: [{p_vals.min():.4f}, {p_vals.max():.4f}] m")
        print(f"  Mean pressure: {p_vals.mean():.4f} m")
        
        # Count saturated nodes
        saturated = np.sum(p_vals > 0)
        total = len(p_vals)
        print(f"  Saturated nodes: {saturated}/{total} ({100*saturated/total:.1f}%)")
        
        # Total water
        total_water = self.compute_total_water_content()
        print(f"  Total water: {total_water:.3f} m³")