"""
Chloride Transport in Porous Media - SIMPLIFIED AND EXPLAINED
Coupled with Richards Equation results (pressure → velocity, saturation)

GOVERNING EQUATION:
∂(θc)/∂t + ∇·(vc) = ∇·(D∇c) + S

Where:
- θ = water content = porosity × saturation
- c = chloride concentration (mol/m³)
- v = Darcy velocity vector (m/s) - from Richards equation
- D = hydrodynamic dispersion tensor (m²/s)
- S = source/sink term (mol/m³/s)

PHYSICAL TERMS:
1. ∂(θc)/∂t        : Accumulation (concentration change in time)
2. ∇·(vc)          : ADVECTION/CONVECTION (transport by water flow)
3. ∇·(D∇c)         : DIFFUSION + DISPERSION (molecular + mechanical mixing)
4. S               : Sources/sinks (rainfall input, root uptake, etc.)
"""
from firedrake import *

class ChlorideTransport:
    """Transport of dissolved chloride in soil water"""

    def __init__(self, mesh, porous_properties, transport_properties, bc_manager):
        """
        Initialize chloride transport model
        
        Parameters:
        -----------
        mesh : firedrake.Mesh
        
        porous_properties : dict
            'porosity': float - soil porosity (0-1), typically 0.3-0.5
            'initial_saturation': float - initial water saturation (0-1)
        
        transport_properties : dict
            'molecular_diffusion': float - D₀ for Cl⁻ in water (m²/s)
                Default: 2.03e-9 m²/s (standard value for chloride)
            'longitudinal_dispersivity': float - αₗ (m)
                Spreading in flow direction, typically 0.01-0.1 m
            'transverse_dispersivity': float - αₜ (m)
                Spreading perpendicular to flow, typically αₗ/10
            'tortuosity': str - how porous media restricts diffusion
                'millington_quirk' (most accurate), 'bruggeman', or 'simple'
        """
        self.mesh = mesh
        self.porous_props = porous_properties
        self.transport_props = transport_properties
        self.bc_manager = bc_manager

        # Function spaces
        self.V = FunctionSpace(mesh, "CG", 1)  # Scalar concentration field
        self.V_vec = VectorFunctionSpace(mesh, "CG", 1)  # Vector velocity field
        
        # === STATE VARIABLES ===
        # Chloride concentration (mol/m³ of water)
        self.c = Function(self.V, name="Chloride_Concentration")
        self.c_n = Function(self.V)  # Previous time step value
        
        # Darcy velocity from Richards equation: v = -K(∇Hp + ∇z)
        self.velocity = Function(self.V_vec, name="Darcy_Velocity")
        
        # === POROUS MEDIA PROPERTIES ===
        # Porosity (volume fraction of pores)
        porosity_input = self.porous_props.get('porosity', 0.4)
        self.porosity = Constant(porosity_input)
        
        # Saturation (fraction of pores filled with water, from Richards)
        self.saturation = Function(self.V, name="Saturation")
        self.saturation.assign(self.porous_props.get('initial_saturation', 1.0))
        
        # === TRANSPORT PROPERTIES ===
        # Molecular diffusion coefficient for Cl⁻ in free water
        self.D_molecular = self.transport_props.get('molecular_diffusion', 2.03e-9)
        
        # Dispersivity: mechanical mixing due to pore-scale velocity variations
        self.alpha_L = self.transport_props.get('longitudinal_dispersivity', 0.01)  # Parallel to flow
        self.alpha_T = self.transport_props.get('transverse_dispersivity', 0.001)  # Perpendicular to flow
        
        # Tortuosity model: accounts for tortuous path through pores
        self.tortuosity_model = self.transport_props.get('tortuosity', 'millington_quirk')
        
        # Time step (will be updated from Richards solver)
        self.dt = Constant(3600.0)  # Default 1 hour
        
    def set_velocity(self, velocity):
        """
        Update Darcy velocity from Richards equation solver
        v = -K(Hp)[∇Hp + ∇z]  (computed in Richards solver)
        """
        self.velocity.assign(velocity)
        
    def set_saturation(self, saturation):
        """
        Update saturation from Richards equation solver
        S = θ/n (degree of saturation, 0-1)
        """
        self.saturation.assign(saturation)
    
    def set_initial_condition(self, c_init):
        """
        Set initial chloride concentration
        
        Parameters:
        -----------
        c_init : float or Function
            Initial concentration (mol/m³ or mg/L)
            Example: 0.0 (no chloride initially)
        """
        if isinstance(c_init, (int, float)):
            self.c.assign(Constant(c_init))
        else:
            self.c.assign(c_init)
        self.c_n.assign(self.c)
    
    # ============================================================================
    # DIFFUSION TERM: ∇·(D_eff ∇c)
    # ============================================================================
    def effective_diffusivity(self):
        """
        Calculate effective diffusion coefficient in porous media
        
        EQUATION: D_eff = D₀ × τ(θ)
        
        Where:
        - D₀ = molecular diffusion in free water (2.03e-9 m²/s for Cl⁻)
        - τ = tortuosity factor (< 1, accounts for tortuous path)
        - θ = volumetric water content = porosity × saturation
        
        TORTUOSITY MODELS:
        1. Millington-Quirk (most accurate):
           τ = θ^(10/3) / n²
        
        2. Bruggeman:
           τ = θ^1.5
        
        3. Simple:
           τ = θ/n
        
        Returns: D_eff (m²/s)
        """
        n = self.porosity  # Porosity
        s = self.saturation  # Saturation (0-1)
        theta = n * s  # Volumetric water content (m³ water / m³ soil)
        
        # Choose tortuosity model
        if self.tortuosity_model == 'millington_quirk':
            # Most commonly used in soil science
            tau = theta**(10.0/3.0) / n**2
            
        elif self.tortuosity_model == 'bruggeman':
            # Simpler, but less accurate for soils
            tau = theta**1.5
            
        else:  # 'simple'
            # Linear approximation
            tau = theta / n
            
        # Effective diffusion = molecular diffusion × tortuosity
        D_eff = self.D_molecular * tau
        
        return D_eff
    
    # ============================================================================
    # DISPERSION TERM: ∇·(D_mech ∇c)
    # ============================================================================
    def dispersion_tensor(self):
        """
        Calculate HYDRODYNAMIC DISPERSION tensor
        Combines molecular diffusion + mechanical dispersion
        
        EQUATION (Bear, 1972):
        D_ij = (αₜ|v| + D_eff)δ_ij + (αₗ - αₜ)(vᵢvⱼ/|v|)
        
        Where:
        - D_eff = effective molecular diffusion (from tortuosity)
        - αₗ = longitudinal dispersivity (spreading parallel to flow)
        - αₜ = transverse dispersivity (spreading perpendicular to flow)
        - v = Darcy velocity vector
        - |v| = velocity magnitude
        - δ_ij = Kronecker delta (identity component)
        
        PHYSICAL MEANING:
        - Isotropic part: (αₜ|v| + D_eff) in all directions
        - Anisotropic part: Extra (αₗ - αₜ) in flow direction
        
        Returns: 2×2 tensor D
        """
        v = self.velocity  # Darcy velocity vector
        v_mag = sqrt(dot(v, v) + 1e-12)  # |v|, add small value to avoid divide-by-zero
        
        # Effective diffusion (molecular transport through tortuous pores)
        D_eff = self.effective_diffusivity()
        
        # Mechanical dispersion (mixing due to velocity variations in pores)
        D_mech_L = self.alpha_L * v_mag  # Longitudinal
        D_mech_T = self.alpha_T * v_mag  # Transverse
        
        # Normalized velocity direction
        v_norm = v / v_mag  # Unit vector in flow direction
        
        # Build dispersion tensor for 2D: [vₓ, vᵧ]
        vx, vy = v_norm[0], v_norm[1]
        
        # D = [D_xx  D_xy]
        #     [D_yx  D_yy]
        D_tensor = as_matrix([
            # D_xx = D_eff + D_T + (D_L - D_T)×vₓ²
            [D_eff + D_mech_T + (D_mech_L - D_mech_T) * vx**2,
                # D_xy = (D_L - D_T)×vₓvᵧ
                (D_mech_L - D_mech_T) * vx * vy],
            
            # D_yx = (D_L - D_T)×vᵧvₓ (symmetric)
            [(D_mech_L - D_mech_T) * vx * vy,
                # D_yy = D_eff + D_T + (D_L - D_T)×vᵧ²
                D_eff + D_mech_T + (D_mech_L - D_mech_T) * vy**2]
        ])
            
        return D_tensor
    
    # ============================================================================
    # WEAK FORM (Variational Formulation)
    # ============================================================================
    def build_variational_form(self, source_term=None):
        """
        Build variational (weak) formulation for transport equation
        
        STRONG FORM (PDE):
        ∂(θc)/∂t + ∇·(vc - D∇c) = S
        
        Expanding:
        ∂(θc)/∂t + ∇·(vc) - ∇·(D∇c) = S
            ↓           ↓          ↓       ↓
        Accumulation Advection Dispersion Source
        
        WEAK FORM (multiply by test function w, integrate):
        ∫ θ(c-cₙ)/Δt × w dx  +  ∫ vc·∇w dx  +  ∫ D∇c·∇w dx  =  ∫ S×w dx
           (time term)        (convection)    (diffusion)       (source)
        
        Note: Integration by parts transforms:
        - ∫ ∇·(vc) w dx  →  -∫ vc·∇w dx  (conservative form)
        - ∫ ∇·(D∇c) w dx  →  ∫ D∇c·∇w dx  (diffusion form)
        
        Parameters:
        -----------
        source_term : Constant, Function, or None
            Source/sink term S (mol/m³/s)
            - Positive = source (chloride input)
            - Negative = sink (removal)
        
        Returns: UFL form F
        """
        # Trial function (what we're solving for)
        c = self.c  # Current concentration
        c_n = self.c_n  # Previous time step
        
        # Test function (for weak formulation)
        w = TestFunction(self.V)
        
        # === TERM 1: ACCUMULATION ===
        # ∂(θc)/∂t ≈ θ(c - cₙ)/Δt
        # Backward Euler time discretization
        theta = self.porosity * self.saturation  # Water content
        time_term = theta * (c - c_n) / self.dt * w * dx
        
        # === TERM 2: ADVECTION (CONVECTION) ===
        # ∇·(vc) in weak form: -∫ c(v·∇w) dx
        # Chloride carried by water flow
        v = self.velocity
        convection_term = -c * dot(v, grad(w)) * dx
        # Note: negative sign from integration by parts
        
        # === TERM 3: DIFFUSION + DISPERSION ===
        # ∇·(D∇c) in weak form: ∫ D∇c·∇w dx
        # Molecular diffusion + mechanical mixing
        D = self.dispersion_tensor()
        diffusion_term = inner(D * grad(c), grad(w)) * dx
        # inner() = dot product for tensors
        
        # === TERM 4: SOURCE/SINK ===
        if source_term is None:
            source = Constant(0.0)  # No sources
        else:
            source = source_term
        source_term_form = -source * w * dx
        # Negative because we move it to LHS: (...) - S = 0
        
        # === COMPLETE WEAK FORM ===
        # Find c such that F = 0 for all test functions w
        F = time_term + convection_term + diffusion_term + source_term_form
        
        return F
    
    # ============================================================================
    # SOLVER
    # ============================================================================
    def solve_timestep(self, dt=None, solver_parameters=None):
        """
        Solve transport equation for one time step
        
        Parameters:
        -----------
        dt : float
            Time step size (seconds) - should match Richards solver dt
        
        bc_dict : dict
            Boundary conditions {boundary_id: value}
            Example: {1: 0.0, 4: 100.0}  # Left=0, Top=100 mol/m³
            Use 'no_flux' for natural boundary (∂c/∂n = 0)
        
        solver_parameters : dict
            Linear solver options
        
        Returns: concentration field c
        """
        if dt is not None:
            self.dt.assign(dt)
        
        # Build weak form
        F = self.build_variational_form()

        # Boundary conditions
        bcs = self.bc_manager.get_dirichlet_bcs()

        # Solver settings
        if solver_parameters is None:
            solver_parameters = {
                'ksp_type': 'gmres',  # Iterative solver
                'pc_type': 'ilu',  # Preconditioner
                'ksp_rtol': 1e-8  # Tolerance
            }
        
        # SOLVE: Find c such that F = 0
        # This solves the linear system A·c = b
        solve(F == 0, self.c, bcs=bcs, solver_parameters=solver_parameters)
        
        # Update for next time step
        self.c_n.assign(self.c)
        
        return self.c
    
