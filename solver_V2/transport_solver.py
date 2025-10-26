"""
Advection-Dispersion transport solver
Solves: ∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S
"""
from firedrake import (
    Function, TrialFunction, TestFunction, dx, ds, lhs, rhs, solve,
    as_matrix, sqrt, grad, dot, conditional, project, VectorFunctionSpace, Constant, SpatialCoordinate, And
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
        v = -K∇h where h is hydraulic head (pressure in meters)
        Since Richards solver works with head, we don't add gravity here
        """
        pressure = self.pressure_solver.p_n
        
        # Get hydraulic conductivity field
        kr = self.field_map.get_kr_field(pressure)
        K_field = Function(self.V)
        K_field.dat.data[:] = kr.dat.data_ro * self.Ks_field.dat.data_ro
        
        # Compute v = -K∇h (head already includes elevation effects)
        vx = project(-K_field * grad(pressure)[0], self.V)
        vy = project(-K_field * (grad(pressure)[1] + 1.0), self.V)  # Add gravity in y-direction
        
        # Assemble into vector field
        self.velocity.dat.data[:, 0] = vx.dat.data_ro
        self.velocity.dat.data[:, 1] = vy.dat.data_ro

        # Debug velocity diagnostics
        if self.debug:
            v_mag = np.sqrt(vx.dat.data_ro**2 + vy.dat.data_ro**2)
            print(f"  Velocity: min={v_mag.min():.2e} m/s, max={v_mag.max():.2e} m/s, mean={v_mag.mean():.2e} m/s")
    
    def get_source_function(self, t):
        """
        Create volumetric concentration source as a Function on the mesh
        Apply a direct concentration rate (mg/L/hr) in specified zones during active events
        """
        source_func = Function(self.V, name="chloride_source")
        source_func.assign(Constant(0.0))
        
        # Get current rate from scenario - t is in seconds, events are stored in seconds
        t_hours = t / 3600.0
        
        if self.debug:
            print(f"  DEBUG: get_source_function called at t={t:.1f}s ({t_hours:.1f}h)")
            print(f"  DEBUG: Found {len(self.transport_source.events)} events")
        
        for event in self.transport_source.events:
            if self.debug:
                print(f"  DEBUG: Event '{event.name}': start={event.start:.1f}s ({event.start/3600:.1f}h), end={event.end:.1f}s ({event.end/3600:.1f}h), rate={event.rate}")
            
            if event.start <= t <= event.end:
                if self.debug:
                    print(f"  DEBUG: Event '{event.name}' is ACTIVE!")
                    print(f"  DEBUG: Found {len(self.transport_source.zones)} zones")
                
                # Get coordinates for zone checking
                coords = SpatialCoordinate(self.mesh)
                x, y = coords
                
                # Apply rate to each zone
                for zone in self.transport_source.zones:
                    if self.debug:
                        print(f"  DEBUG: Checking zone '{zone.name}' against event zones: {event.zones}")
                    
                    # Check if this zone name matches any zone in the event
                    zone_names_in_event = [z.name if hasattr(z, 'name') else str(z) for z in event.zones]
                    if zone.name in zone_names_in_event:
                        if self.debug:
                            print(f"  DEBUG: Zone '{zone.name}': x=[{zone.x_min}, {zone.x_max}], y=[{zone.y_min}, {zone.y_max}]")
                        
                        # Create zone condition
                        zone_condition = And(And(x >= zone.x_min, x <= zone.x_max), 
                                           And(y >= zone.y_min, y <= zone.y_max))
                        
                        # Apply concentration rate (convert from arbitrary units to mg/L/hr)
                        concentration_rate = event.rate  # mg/L/hr during application
                        
                        # Convert to per-second rate for the timestep
                        rate_per_second = concentration_rate / 3600.0  # mg/L/s
                        
                        # Add to source function where zone condition is true
                        source_func = conditional(zone_condition, 
                                                Constant(rate_per_second), 
                                                source_func)
                        
                        if self.debug:
                            # Count nodes in the zone for debugging
                            test_func = Function(self.V)
                            test_func.interpolate(conditional(zone_condition, Constant(1.0), Constant(0.0)))
                            num_nodes = np.sum(test_func.dat.data_ro > 0.5)
                            print(f"  Source '{zone.name}': {event.rate} units applied as {concentration_rate} mg/L/hr at {num_nodes} nodes")
            else:
                if self.debug:
                    print(f"  DEBUG: No active source (rate=0.0)")
        
        return source_func
    
    def construct_dispersion_tensor(self, D_L, D_T, vx, vy):
        """
        Construct anisotropic dispersion tensor
        
        D_ij = (D_T * |v|) * δ_ij + (D_L - D_T) * (v_i * v_j / |v|)
        
        Returns:
        --------
        D_tensor : 2x2 UFL matrix
        """
        # Velocity magnitude (add small value to avoid division by zero)
        v_mag = sqrt(vx**2 + vy**2 + 1e-20)
        
        # Anisotropic components
        D_xx = D_T * v_mag + (D_L - D_T) * vx**2 / v_mag
        D_yy = D_T * v_mag + (D_L - D_T) * vy**2 / v_mag
        D_xy = (D_L - D_T) * vx * vy / v_mag
        
        # Handle very small velocities (use isotropic molecular diffusion)
        threshold = 1e-10  # m/s
        D_iso = D_L  # Molecular diffusion when v ≈ 0
        
        D_xx_final = conditional(v_mag > threshold, D_xx, D_iso)
        D_yy_final = conditional(v_mag > threshold, D_yy, D_iso)
        D_xy_final = conditional(v_mag > threshold, D_xy, 0.0)
        
        # Construct tensor as UFL matrix
        D_tensor = as_matrix([[D_xx_final, D_xy_final],
                              [D_xy_final, D_yy_final]])
        
        return D_tensor

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
        # Temporarily increase dispersion for testing
        D_L = Constant(0.001)  # m²/s - higher for testing
        D_T = Constant(0.0005)  # m²/s

        # Step 3: Get source
        source_func = self.get_source_function(t)
        
        # Step 4: Solve transport equation
        # ∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S
        # With retardation: R·θ·∂c/∂t + ∇·(vc) = ∇·(D∇c) + S
        
        c = TrialFunction(self.V)
        q = TestFunction(self.V)
        
        # Effective storage term (accounts for retardation)
        storage_coeff = R * theta
        
        # Time discretization parameter (theta-method)
        # theta_time = 1.0 is backward Euler (implicit, stable)
        # theta_time = 0.5 is Crank-Nicolson (second-order)
        theta_time = 1.0
        
        c_adv = theta_time * c + (1.0 - theta_time) * self.c_n
        c_disp = theta_time * c + (1.0 - theta_time) * self.c_n
        
        # Time derivative
        F = storage_coeff * (c - self.c_n) / self.config.dt * q * dx
        
        # Advection
        F += -dot(self.velocity, grad(q)) * c_adv * dx

        D_tensor = self.construct_dispersion_tensor(D_L, D_T, vx, vy)
        F += -dot(dot(D_tensor, grad(c_disp)), grad(q)) * dx

        F += -source_func * q * dx
        
        a = lhs(F)
        L = rhs(F)
        
                # Solve with appropriate solver parameters for transport
        solver_params = {
            'ksp_type': 'gmres',
            'pc_type': 'bjacobi',
            'ksp_rtol': 1e-6,
            'ksp_atol': 1e-10,
            'ksp_max_it': 1000,
        }

        # No boundary conditions for transport (natural no-flux boundaries)
        # Solve
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