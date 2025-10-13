"""
Coupled Richards Equation and Chloride Transport
Sequential coupling approach for your Richards solver
"""

from firedrake import *
import numpy as np

class CoupledFlowTransport:
    """
    Couples Richards equation (water flow) with chloride transport
    
    Sequential approach:
    1. Solve Richards equation for pressure/saturation
    2. Compute Darcy velocity from pressure field
    3. Solve transport equation with updated velocity and saturation
    """
    
    def __init__(self, richards_solver, transport_solver):
        """
        Initialize coupled solver
        
        Parameters:
        -----------
        richards_solver : RichardsSolver
            Your Richards equation solver instance
        transport_solver : ChlorideTransport
            Chloride transport solver instance
        """
        self.richards = richards_solver
        self.transport = transport_solver
        self.mesh = richards_solver.mesh
        self.V = richards_solver.V
        
        # Function space for velocity
        self.V_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        
        # Storage for velocity field
        self.velocity = Function(self.V_vec, name="Darcy_Velocity")
        
    def compute_darcy_velocity(self):
        """
        Compute Darcy velocity from Richards solver state
        v = -K * grad(h) = -K * (grad(p) + grad(z))
        where h = pressure_head + z (total hydraulic head)
        
        Returns:
        --------
        velocity : Function
            Darcy velocity (m/s)
        """
        # Get pressure head from Richards solver
        p = self.richards.p_new
        
        # Get hydraulic conductivity field (K = kr * Ks)
        kr = self.richards.kr_n
        Ks = self.richards.Ks_field
        K = kr * Ks
        
        # Elevation (z coordinate)
        coords = SpatialCoordinate(self.mesh)
        if self.mesh.geometric_dimension() == 2:
            z = coords[1]  # y is vertical
        elif self.mesh.geometric_dimension() == 3:
            z = coords[2]  # z is vertical
        else:  # 1D
            z = coords[0]
        
        # Total hydraulic head: h = p + z
        total_head = p + z
        
        # Darcy velocity: v = -K * grad(h)
        # Use projection to compute velocity
        v = TrialFunction(self.V_vec)
        w = TestFunction(self.V_vec)
        
        a = inner(v, w) * dx
        L = inner(-K * grad(total_head), w) * dx
        
        solve(a == L, self.velocity)
        
        return self.velocity
    
    def run(self, t_end, dt_richards, dt_transport=None,
            transport_bc=None, dirichlet_bcs=None,
            output_interval=None, probe_manager=None,
            snapshot_manager=None, print_diagnostics=False):
        """
        Run coupled simulation
        
        Parameters:
        -----------
        t_end : float
            End time (seconds)
        dt_richards : float
            Time step for Richards equation (seconds)
        dt_transport : float
            Time step for transport (seconds). If None, uses dt_richards
        transport_bc : dict
            Boundary conditions for transport {boundary_id: value}
        dirichlet_bcs : list
            Dirichlet BCs for Richards equation
        output_interval : float
            Output interval (seconds). If None, outputs every Richards step
        probe_manager : ProbeManager
            Optional probe manager for monitoring
        snapshot_manager : SnapshotManager
            Optional snapshot manager for spatial output
        print_diagnostics : bool
            Print detailed diagnostics
        """
        if dt_transport is None:
            dt_transport = dt_richards
        
        # Initialize counters
        t = 0.0
        step = 0
        output_count = 0
        
        # Output files for transport
        if snapshot_manager is not None:
            concentration_file = File("results/chloride_concentration.pvd")
            velocity_file = File("results/darcy_velocity.pvd")
        
        print("=" * 70)
        print("COUPLED FLOW-TRANSPORT SIMULATION")
        print("=" * 70)
        print(f"Total duration: {t_end/3600:.1f} hours")
        print(f"Richards dt: {dt_richards:.1f} s")
        print(f"Transport dt: {dt_transport:.1f} s")
        print(f"Substeps per Richards step: {int(dt_richards/dt_transport)}")
        print()
        
        # Print Richards solver info
        print("Richards Equation:")
        self.richards.domain.print_summary()
        print()
        
        # Print transport info
        print("Chloride Transport:")
        print(f"  Molecular diffusion: {self.transport.D_molecular:.2e} m²/s")
        print(f"  Longitudinal dispersivity: {self.transport.alpha_L:.3f} m")
        print(f"  Transverse dispersivity: {self.transport.alpha_T:.4f} m")
        print(f"  Tortuosity model: {self.transport.tortuosity_model}")
        if self.transport.include_adsorption:
            print(f"  Adsorption: Kd = {self.transport.K_d:.2e} m³/kg")
        print()
        
        print("Starting time integration...")
        print("-" * 70)
        
        while t < t_end:
            # =======================================
            # STEP 1: Solve Richards Equation
            # =======================================
            self.richards.solve_timestep(t + dt_richards, bcs=dirichlet_bcs)
            
            # =======================================
            # STEP 2: Compute Darcy Velocity
            # =======================================
            velocity = self.compute_darcy_velocity()
            self.transport.set_velocity(velocity)
            
            # =======================================
            # STEP 3: Update Saturation
            # =======================================
            saturation = self.richards.domain.compute_saturation_field(self.richards.p_new)
            self.transport.set_saturation(saturation)
            
            # =======================================
            # STEP 4: Solve Transport (with sub-stepping)
            # =======================================
            n_substeps = int(dt_richards / dt_transport)
            
            for substep in range(n_substeps):
                self.transport.solve_timestep(
                    dt=dt_transport,
                    use_stabilization=True,
                    bc_dict=transport_bc
                )
            
            # Update time
            t += dt_richards
            step += 1
            
            # =======================================
            # MONITORING AND OUTPUT
            # =======================================
            
            # Record probes
            if probe_manager is not None:
                probe_manager.record(t, self.richards.p_new)
            
            # Snapshots
            if snapshot_manager is not None:
                if output_interval is None or (t % output_interval < dt_richards):
                    snapshot_manager.record(t, self.richards.p_new)
                    concentration_file.write(self.transport.c, time=t)
                    velocity_file.write(self.velocity, time=t)
                    output_count += 1
            
            # Progress reporting
            if step % max(1, int(3600 / dt_richards)) == 0:
                print(f"Time: {t/3600:.2f}h / {t_end/3600:.1f}h", end="")
                
                if print_diagnostics:
                    # Pressure diagnostics
                    p_vals = self.richards.p_new.dat.data[:]
                    p_min, p_max, p_mean = p_vals.min(), p_vals.max(), p_vals.mean()
                    
                    # Concentration diagnostics
                    c_vals = self.transport.c.dat.data[:]
                    c_min, c_max, c_mean = c_vals.min(), c_vals.max(), c_vals.mean()
                    
                    # Velocity diagnostics
                    v_vals = self.velocity.dat.data[:]
                    v_mag = np.sqrt(np.sum(v_vals**2, axis=1))
                    v_max = v_mag.max()
                    
                    print()
                    print(f"  Pressure: [{p_min:.3f}, {p_max:.3f}] m (mean: {p_mean:.3f})")
                    print(f"  Chloride: [{c_min:.2e}, {c_max:.2e}] mol/m³ (mean: {c_mean:.2e})")
                    print(f"  Velocity: max |v| = {v_max:.2e} m/s")
                else:
                    print()
        
        print("-" * 70)
        print(f"Simulation complete! ({output_count} outputs saved)")
        print("=" * 70)
        
        return {
            'pressure': self.richards.p_new,
            'saturation': saturation,
            'concentration': self.transport.c,
            'velocity': self.velocity
        }
