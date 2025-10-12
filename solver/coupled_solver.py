"""
Coupled Richards Equation and Chloride Transport
Iterative or sequential coupling approach
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
    
    def __init__(self, mesh, richards_solver, transport_solver):
        """
        Parameters:
        -----------
        mesh : firedrake.Mesh
            Computational mesh
        richards_solver : object
            Your existing Richards equation solver
            Must have: pressure field, saturation field, hydraulic conductivity
        transport_solver : ChlorideTransport
            Chloride transport solver instance
        """
        self.mesh = mesh
        self.richards = richards_solver
        self.transport = transport_solver
        
        # Function spaces
        self.V_vec = VectorFunctionSpace(mesh, "CG", 1)
        self.V = FunctionSpace(mesh, "CG", 1)
        
    def compute_darcy_velocity(self, pressure_head, K_field):
        """
        Compute Darcy velocity from pressure head
        v = -K * grad(h)
        where h = pressure_head + z (total hydraulic head)
        
        Parameters:
        -----------
        pressure_head : Function
            Pressure head (m) from Richards equation
        K_field : Function
            Hydraulic conductivity (m/s)
        
        Returns:
        --------
        velocity : Function
            Darcy velocity (m/s)
        """
        # Total hydraulic head
        x = SpatialCoordinate(self.mesh)
        if self.mesh.geometric_dimension() == 2:
            z = x[1]  # Assuming vertical is second coordinate
        elif self.mesh.geometric_dimension() == 3:
            z = x[2]  # Assuming vertical is third coordinate
        else:  # 1D
            z = x[0]
            
        total_head = pressure_head + z
        
        # Darcy velocity: v = -K * grad(h)
        velocity = Function(self.V_vec, name="Darcy_Velocity")
        
        # Project velocity
        v = TrialFunction(self.V_vec)
        w = TestFunction(self.V_vec)
        
        a = inner(v, w) * dx
        L = inner(-K_field * grad(total_head), w) * dx
        
        solve(a == L, velocity)
        
        return velocity
    
    def couple_sequential(self, t_end, dt_richards, dt_transport, 
                         richards_bc, transport_bc,
                         output_interval=None):
        """
        Sequential coupling: solve Richards then Transport
        
        Parameters:
        -----------
        t_end : float
            End time (seconds)
        dt_richards : float
            Time step for Richards equation
        dt_transport : float
            Time step for transport (can be subdivisions of dt_richards)
        richards_bc : dict
            Boundary conditions for Richards equation
        transport_bc : dict
            Boundary conditions for transport
        output_interval : float
            Output every N seconds (None = every Richards step)
        """
        t = 0.0
        output_count = 0
        
        # Output files
        pressure_file = File("results/pressure_head.pvd")
        saturation_file = File("results/saturation.pvd")
        velocity_file = File("results/velocity.pvd")
        concentration_file = File("results/chloride_concentration.pvd")
        
        print(f"Starting coupled simulation")
        print(f"Richards dt: {dt_richards}s, Transport dt: {dt_transport}s")
        
        while t < t_end:
            print(f"\n=== Time: {t/86400:.3f} days ({t:.1f}s) ===")
            
            # Step 1: Solve Richards equation
            print("Solving Richards equation...")
            # Assuming your Richards solver has a method like:
            # self.richards.solve_timestep(dt_richards, bc_dict=richards_bc)
            pressure = self.richards.solve_timestep(dt_richards, richards_bc)
            
            # Get saturation from Richards solver
            # Assuming: saturation = self.richards.compute_saturation()
            saturation = self.richards.get_saturation()
            
            # Get hydraulic conductivity
            # Assuming: K = self.richards.get_hydraulic_conductivity()
            K = self.richards.get_hydraulic_conductivity()
            
            # Step 2: Compute Darcy velocity
            print("Computing Darcy velocity...")
            velocity = self.compute_darcy_velocity(pressure, K)
            
            # Step 3: Update transport solver with new flow field
            self.transport.set_velocity(velocity)
            self.transport.set_saturation(saturation)
            
            # Step 4: Solve transport (possibly with sub-stepping)
            print("Solving chloride transport...")
            n_substeps = int(dt_richards / dt_transport)
            
            for substep in range(n_substeps):
                self.transport.solve_timestep(
                    dt=dt_transport,
                    use_stabilization=True,
                    bc_dict=transport_bc
                )
                
                if substep < n_substeps - 1:
                    print(f"  Substep {substep+1}/{n_substeps}")
            
            t += dt_richards
            
            # Output
            if output_interval is None or (t % output_interval < dt_richards):
                print(f"Writing output {output_count}")
                pressure_file.write(pressure)
                saturation_file.write(saturation)
                velocity_file.write(velocity)
                concentration_file.write(self.transport.c)
                output_count += 1
                
                # Print diagnostics
                c_max = self.transport.c.dat.data.max()
                c_min = self.transport.c.dat.data.min()
                print(f"  Concentration range: [{c_min:.2e}, {c_max:.2e}] mol/m³")
                
        print(f"\nSimulation complete! Total time: {t/86400:.3f} days")


class SimpleRichardsAdapter:
    """
    Adapter to make your Richards solver compatible with coupling
    Modify this to match your actual Richards implementation
    """
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.V = FunctionSpace(mesh, "CG", 1)
        
        # State variables
        self.pressure_head = Function(self.V, name="Pressure_Head")
        self.saturation = Function(self.V, name="Saturation")
        self.hydraulic_conductivity = Function(self.V, name="Hydraulic_Conductivity")
        
        # Soil parameters (example: van Genuchten)
        self.theta_s = 0.45  # Saturated water content
        self.theta_r = 0.067  # Residual water content
        self.alpha = 0.0028  # van Genuchten parameter (1/cm)
        self.n = 1.41  # van Genuchten parameter
        self.K_sat = 1.19e-5  # Saturated hydraulic conductivity (m/s)
        
    def solve_timestep(self, dt, bc_dict):
        """
        Your actual Richards solver would go here
        This is just a placeholder showing the interface needed
        """
        # Your Richards equation solve
        # ...
        # Return pressure head
        return self.pressure_head
    
    def get_saturation(self):
        """
        Compute saturation from pressure head (van Genuchten)
        """
        h = self.pressure_head
        
        # van Genuchten equation
        alpha = self.alpha
        n = self.n
        m = 1 - 1/n
        
        # Effective saturation
        Se = conditional(
            h < 0,
            1.0 / (1.0 + abs(alpha * h)**n)**m,
            1.0
        )
        
        # Actual saturation
        self.saturation.interpolate(
            self.theta_r + (self.theta_s - self.theta_r) * Se
        )
        
        return self.saturation
    
    def get_hydraulic_conductivity(self):
        """
        Compute hydraulic conductivity from saturation
        """
        h = self.pressure_head
        
        # van Genuchten relative permeability
        alpha = self.alpha
        n = self.n
        m = 1 - 1/n
        
        Se = conditional(
            h < 0,
            1.0 / (1.0 + abs(alpha * h)**n)**m,
            1.0
        )
        
        K_rel = sqrt(Se) * (1 - (1 - Se**(1/m))**m)**2
        
        self.hydraulic_conductivity.interpolate(
            self.K_sat * K_rel
        )
        
        return self.hydraulic_conductivity


# Complete example with coupling
def run_coupled_simulation():
    """
    Complete example: coupled flow and chloride transport
    """
    from chloride_transport import ChlorideTransport
    
    # Create mesh (2D vertical cross-section)
    nx, ny = 50, 100
    Lx, Ly = 1.0, 2.0  # 1m wide, 2m deep
    mesh = RectangleMesh(nx, ny, Lx, Ly)
    
    # ===== Setup Richards Equation =====
    print("Setting up Richards equation solver...")
    richards = SimpleRichardsAdapter(mesh)
    # Initialize with hydrostatic conditions
    x = SpatialCoordinate(mesh)
    richards.pressure_head.interpolate(-x[1])  # Negative depth
    
    # ===== Setup Transport =====
    print("Setting up chloride transport solver...")
    porous_props = {
        'porosity': 0.35,
        'initial_saturation': 0.8
    }
    
    transport_props = {
        'molecular_diffusion': 2.03e-9,  # m²/s
        'longitudinal_dispersivity': 0.05,  # 5 cm
        'transverse_dispersivity': 0.005,  # 5 mm
        'tortuosity': 'millington_quirk'
    }
    
    transport = ChlorideTransport(mesh, porous_props, transport_props)
    transport.set_initial_condition(0.0)  # Initially clean
    
    # ===== Create Coupled Solver =====
    print("Creating coupled solver...")
    coupled = CoupledFlowTransport(mesh, richards, transport)
    
    # ===== Boundary Conditions =====
    # Richards BC (example)
    richards_bc = {
        1: 0.0,  # Top: ponding (h=0)
        # Bottom: free drainage (no BC needed)
    }
    
    # Transport BC
    transport_bc = {
        1: Constant(50.0),  # Top: 50 mol/m³ chloride influx
        # Bottom: outflow (no BC)
    }
    
    # ===== Run Simulation =====
    print("Running coupled simulation...")
    t_end = 7 * 86400  # 7 days
    dt_richards = 3600.0  # 1 hour for Richards
    dt_transport = 900.0  # 15 minutes for transport
    
    coupled.couple_sequential(
        t_end=t_end,
        dt_richards=dt_richards,
        dt_transport=dt_transport,
        richards_bc=richards_bc,
        transport_bc=transport_bc,
        output_interval=3600.0  # Output every hour
    )
    
    print("Done!")


if __name__ == "__main__":
    run_coupled_simulation()