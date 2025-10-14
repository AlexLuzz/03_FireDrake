"""
Coupled Richards Equation and Chloride Transport - SIMPLIFIED
Sequential coupling: Flow → Velocity → Transport
"""
from firedrake import *
import numpy as np

class CoupledFlowTransport:
    """
    Couples water flow (Richards) with chloride transport
    
    Sequence each timestep:
    1. Solve Richards: get pressure field p
    2. Compute velocity: v = -K(∇p + ∇z)
    3. Update saturation from p
    4. Solve transport with v and saturation
    """
    
    def __init__(self, richards_solver, transport_solver):
        """
        Parameters:
        -----------
        richards_solver : Your RichardsSolver instance
        transport_solver : ChlorideTransport instance
        """
        self.richards = richards_solver
        self.transport = transport_solver
        self.mesh = richards_solver.mesh
        self.V = richards_solver.V
        
        # Vector space for velocity
        self.V_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.velocity = Function(self.V_vec, name="Darcy_Velocity")
        
    def compute_darcy_velocity(self):
        """
        Compute Darcy velocity from Richards results
        
        EQUATION: v = -K(∇Hp + ∇z) = -K∇H
        Where H = total hydraulic head = pressure head + elevation
        
        Returns: velocity field (m/s)
        """
        # Pressure head from Richards solver
        Hp = self.richards.p_new
        
        # Hydraulic conductivity K = kr × Ks
        K = self.richards.kr_n * self.richards.Ks_field
        
        # Elevation (z coordinate = y in 2D)
        coords = SpatialCoordinate(self.mesh)
        z = coords[1] if self.mesh.geometric_dimension() == 2 else coords[2]
        
        # Total head: H = Hp + z
        H = Hp + z
        
        # Project velocity: v = -K∇H
        v = TrialFunction(self.V_vec)
        w = TestFunction(self.V_vec)
        
        a = inner(v, w) * dx
        L = inner(-K * grad(H), w) * dx
        
        solve(a == L, self.velocity)
        
        return self.velocity
    
    def run(self, t_end, dt,
            probe_manager=None, snapshot_manager=None):
        """
        Run coupled simulation
        
        Parameters:
        -----------
        t_end : float
            Total simulation time (seconds)
        dt : float
            Time step for both Richards and transport (seconds)
        transport_bc : dict
            Boundary conditions for chloride: {boundary_id: concentration}
            Example: {4: 100.0}  # Top boundary = 100 mol/m³
        richards_bcs : list
            Dirichlet BCs for Richards equation
        probe_manager : ProbeManager
            For time-series at specific points
        snapshot_manager : SnapshotManager  
            For spatial snapshots at intervals
        
        Returns: dict with final fields
        """
        t = 0.0
        step = 0
        
        # Output files
        p_file = File("results/pressure.pvd")
        c_file = File("results/concentration.pvd")
        v_file = File("results/velocity.pvd")
        
        print(f"Running coupled simulation: {t_end/3600:.1f} hours, dt={dt}s")
        
        while t < t_end:
            # 1. Solve Richards equation
            self.richards.solve_timestep(t + dt)
            
            # 2. Compute Darcy velocity from pressure
            velocity = self.compute_darcy_velocity()
            self.transport.set_velocity(velocity)
            
            # 3. Update saturation for transport
            saturation = self.richards.domain.compute_saturation_field(
                self.richards.p_new)
            self.transport.set_saturation(saturation)

            # 4. Solve chloride transport
            self.transport.solve_timestep(dt)
            
            # Update time
            t += dt
            step += 1
            
            # Record probes (time-series at specific points)
            if probe_manager is not None:
                probe_manager.record(t, self.richards.p_new)
                # Add concentration probes if probe_manager supports it
                if hasattr(probe_manager, 'record_scalar'):
                    probe_manager.record_scalar(t, self.transport.c, 'concentration')
            
            # Record snapshots (full spatial fields at intervals)
            if snapshot_manager is not None:
                snapshot_manager.record(t, self.richards.p_new)
                if hasattr(snapshot_manager, 'record_field'):
                    snapshot_manager.record_field(t, self.transport.c, 'concentration')
                    snapshot_manager.record_field(t, self.velocity, 'velocity')
            
            # Save to pvd files
            if step % max(1, int(3600/dt)) == 0:
                p_file.write(self.richards.p_new, time=t)
                c_file.write(self.transport.c, time=t)
                v_file.write(self.velocity, time=t)
                print(f"  t = {t/3600:.1f}h")
        
        print(f"Done! {step} steps")
        
        return {
            'pressure': self.richards.p_new,
            'concentration': self.transport.c,
            'velocity': self.velocity,
            'saturation': saturation
        }