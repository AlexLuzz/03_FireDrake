"""
Chloride Transport in Porous Media using FireDrake
Coupled with Richards Equation for unsaturated/saturated flow
Based on COMSOL Transport of Diluted Species interface
"""

from firedrake import *
import numpy as np

class ChlorideTransport:
    """
    Transport of diluted chloride species in porous media
    
    Implements:
    - Convection (from Darcy velocity)
    - Diffusion (with effective diffusivity)
    - Dispersion (mechanical mixing)
    - Adsorption (optional)
    - Source/sink terms
    """
    
    def __init__(self, mesh, porous_properties, transport_properties):
        """
        Initialize chloride transport model
        
        Parameters:
        -----------
        mesh : firedrake.Mesh
            Computational mesh
        porous_properties : dict
            Porosity, saturation, etc.
        transport_properties : dict
            Diffusion coefficient, dispersivity, etc.
        """
        self.mesh = mesh
        self.porous_props = porous_properties
        self.transport_props = transport_properties
        
        # Function spaces
        self.V = FunctionSpace(mesh, "CG", 1)  # For concentration
        self.V_vec = VectorFunctionSpace(mesh, "CG", 1)  # For velocity
        
        # Concentration (mol/m³)
        self.c = Function(self.V, name="Chloride_Concentration")
        self.c_n = Function(self.V)  # Previous time step
        
        # Darcy velocity (from Richards equation)
        self.velocity = Function(self.V_vec, name="Darcy_Velocity")
        
        # Porous media properties
        self.porosity = self.porous_props.get('porosity', 0.4)
        self.saturation = Function(self.V, name="Saturation")
        self.saturation.assign(self.porous_props.get('initial_saturation', 1.0))
        
        # Transport properties
        self.D_molecular = self.transport_props.get('molecular_diffusion', 2.03e-9)  # m²/s for Cl-
        self.alpha_L = self.transport_props.get('longitudinal_dispersivity', 0.01)  # m
        self.alpha_T = self.transport_props.get('transverse_dispersivity', 0.001)  # m
        self.tortuosity_model = self.transport_props.get('tortuosity', 'millington_quirk')
        
        # Adsorption (optional)
        self.include_adsorption = self.transport_props.get('include_adsorption', False)
        if self.include_adsorption:
            self.rho_b = self.transport_props.get('bulk_density', 1500)  # kg/m³
            self.K_d = self.transport_props.get('distribution_coefficient', 0.0)  # m³/kg
        
        # Time stepping
        self.dt = Constant(3600.0)  # Default 1 hour
        
    def set_velocity(self, velocity):
        """Update Darcy velocity from Richards equation solver"""
        self.velocity.assign(velocity)
        
    def set_saturation(self, saturation):
        """Update saturation from Richards equation solver"""
        self.saturation.assign(saturation)
        
    def effective_diffusivity(self):
        """
        Calculate effective diffusion coefficient in porous media
        Accounts for porosity and tortuosity
        """
        epsilon = self.porosity
        s = self.saturation
        theta = epsilon * s  # Volumetric water content
        
        if self.tortuosity_model == 'millington_quirk':
            # Millington-Quirk model: D_eff = D * theta^(10/3) / epsilon^2
            tau = theta**(10.0/3.0) / epsilon**2
        elif self.tortuosity_model == 'bruggeman':
            # Bruggeman model: tau = theta^1.5
            tau = theta**1.5
        else:
            # Simple tortuosity factor
            tau = theta / epsilon
            
        return self.D_molecular * tau
    
    def dispersion_tensor(self):
        """
        Calculate hydrodynamic dispersion tensor
        D_ij = (alpha_T |v| + D_eff) delta_ij + (alpha_L - alpha_T) v_i v_j / |v|
        """
        v = self.velocity
        v_mag = sqrt(dot(v, v) + 1e-12)  # Add small value to avoid division by zero
        
        D_eff = self.effective_diffusivity()
        
        # Dispersivity contribution
        D_mech_L = self.alpha_L * v_mag
        D_mech_T = self.alpha_T * v_mag
        
        # For 2D
        if self.mesh.geometric_dimension() == 2:
            v_norm = v / v_mag
            D_tensor = as_matrix([
                [D_eff + D_mech_T + (D_mech_L - D_mech_T) * v_norm[0]**2,
                 (D_mech_L - D_mech_T) * v_norm[0] * v_norm[1]],
                [(D_mech_L - D_mech_T) * v_norm[0] * v_norm[1],
                 D_eff + D_mech_T + (D_mech_L - D_mech_T) * v_norm[1]**2]
            ])
        else:  # 3D
            v_norm = v / v_mag
            D_tensor = as_matrix([
                [D_eff + D_mech_T + (D_mech_L - D_mech_T) * v_norm[0]**2,
                 (D_mech_L - D_mech_T) * v_norm[0] * v_norm[1],
                 (D_mech_L - D_mech_T) * v_norm[0] * v_norm[2]],
                [(D_mech_L - D_mech_T) * v_norm[1] * v_norm[0],
                 D_eff + D_mech_T + (D_mech_L - D_mech_T) * v_norm[1]**2,
                 (D_mech_L - D_mech_T) * v_norm[1] * v_norm[2]],
                [(D_mech_L - D_mech_T) * v_norm[2] * v_norm[0],
                 (D_mech_L - D_mech_T) * v_norm[2] * v_norm[1],
                 D_eff + D_mech_T + (D_mech_L - D_mech_T) * v_norm[2]**2]
            ])
            
        return D_tensor
    
    def retardation_factor(self):
        """
        Calculate retardation factor for adsorption
        R = 1 + (rho_b * K_d) / theta
        """
        if not self.include_adsorption:
            return 1.0
        
        theta = self.porosity * self.saturation
        R = 1.0 + (self.rho_b * self.K_d) / theta
        return R
    
    def build_variational_form(self, source_term=None):
        """
        Build variational formulation for transport equation
        
        Equation:
        R * theta * dc/dt + div(q_c) = S
        where q_c = v*c - D*grad(c)
        """
        c = self.c
        c_n = self.c_n
        v = self.velocity
        
        # Test function
        w = TestFunction(self.V)
        
        # Water content
        theta = self.porosity * self.saturation
        
        # Retardation factor
        R = self.retardation_factor()
        
        # Dispersion tensor
        D = self.dispersion_tensor()
        
        # Time derivative term (backward Euler)
        time_term = R * theta * (c - c_n) / self.dt * w * dx
        
        # Diffusion/dispersion term (integrated by parts)
        diffusion_term = inner(D * grad(c), grad(w)) * dx
        
        # Convection term (SUPG stabilization recommended)
        # Using conservative form: div(v*c)
        convection_term = -c * div(v * w) * dx
        
        # Source/sink term
        if source_term is None:
            source = Constant(0.0)
        else:
            source = source_term
        source_term_form = -source * w * dx
        
        # Complete form
        F = time_term + diffusion_term + convection_term + source_term_form
        
        return F
    
    def build_stabilized_form(self, source_term=None, stabilization='supg'):
        """
        Build stabilized variational form (SUPG or GLS)
        Important for convection-dominated problems
        """
        c = self.c
        c_n = self.c_n
        v = self.velocity
        
        # Test function
        w = TestFunction(self.V)
        
        # Standard form
        F = self.build_variational_form(source_term)
        
        if stabilization == 'supg':
            # SUPG stabilization
            h = CellSize(self.mesh)
            v_mag = sqrt(dot(v, v) + 1e-12)
            D_eff = self.effective_diffusivity()
            
            # Peclet number
            Pe = v_mag * h / (2.0 * D_eff)
            
            # SUPG parameter
            tau_supg = h / (2.0 * v_mag) * (1.0 / tanh(Pe) - 1.0 / Pe)
            
            # Residual
            theta = self.porosity * self.saturation
            R = self.retardation_factor()
            D = self.dispersion_tensor()
            
            residual = R * theta * (c - c_n) / self.dt - div(D * grad(c)) + dot(v, grad(c))
            if source_term is not None:
                residual -= source_term
            
            # SUPG term
            F += tau_supg * dot(v, grad(w)) * residual * dx
            
        return F
    
    def solve_timestep(self, dt=None, use_stabilization=True, bc_dict=None):
        """
        Solve transport equation for one time step
        
        Parameters:
        -----------
        dt : float
            Time step size (seconds)
        use_stabilization : bool
            Use SUPG stabilization for convection
        bc_dict : dict
            Boundary conditions {boundary_id: value or 'no_flux'}
        """
        if dt is not None:
            self.dt.assign(dt)
        
        # Build form
        if use_stabilization:
            F = self.build_stabilized_form()
        else:
            F = self.build_variational_form()
        
        # Boundary conditions
        bcs = []
        if bc_dict is not None:
            for bc_id, bc_value in bc_dict.items():
                if bc_value != 'no_flux':
                    bcs.append(DirichletBC(self.V, bc_value, bc_id))
        
        # Solve
        solve(F == 0, self.c, bcs=bcs,
              solver_parameters={
                  'ksp_type': 'gmres',
                  'pc_type': 'ilu',
                  'ksp_rtol': 1e-8
              })
        
        # Update previous time step
        self.c_n.assign(self.c)
        
        return self.c
    
    def set_initial_condition(self, c_init):
        """Set initial concentration"""
        if isinstance(c_init, (int, float)):
            self.c.assign(Constant(c_init))
        else:
            self.c.assign(c_init)
        self.c_n.assign(self.c)


# Example usage function
def example_chloride_transport():
    """
    Example: Chloride transport in 1D column
    """
    # Create 1D mesh
    nx = 100
    length = 1.0  # 1 meter column
    mesh = IntervalMesh(nx, length)
    
    # Porous media properties
    porous_props = {
        'porosity': 0.35,
        'initial_saturation': 1.0  # Fully saturated
    }
    
    # Transport properties for chloride
    transport_props = {
        'molecular_diffusion': 2.03e-9,  # m²/s at 25°C
        'longitudinal_dispersivity': 0.01,  # m
        'transverse_dispersivity': 0.001,  # m
        'tortuosity': 'millington_quirk',
        'include_adsorption': False
    }
    
    # Create transport solver
    transport = ChlorideTransport(mesh, porous_props, transport_props)
    
    # Initial condition: no chloride
    transport.set_initial_condition(0.0)
    
    # Set constant velocity (example: 1e-6 m/s downward)
    V_vec = VectorFunctionSpace(mesh, "CG", 1)
    velocity = Function(V_vec)
    velocity.interpolate(as_vector([1e-6]))
    transport.set_velocity(velocity)
    
    # Boundary conditions
    # Top: constant concentration (100 mol/m³)
    # Bottom: outflow (no BC needed with convection form)
    bc_dict = {1: Constant(100.0)}  # boundary_id: value
    
    # Time stepping
    t_end = 86400.0  # 1 day
    dt = 3600.0  # 1 hour
    t = 0.0
    
    # Output file
    outfile = File("chloride_concentration.pvd")
    
    while t < t_end:
        print(f"Time: {t/3600:.2f} hours")
        
        # Solve transport
        transport.solve_timestep(dt=dt, bc_dict=bc_dict)
        
        # Output
        outfile.write(transport.c)
        
        t += dt
    
    return transport


if __name__ == "__main__":
    # Run example
    transport_model = example_chloride_transport()
    print("Simulation complete!")