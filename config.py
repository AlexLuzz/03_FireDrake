"""
Configuration module for Richards equation simulation
Contains all global simulation parameters
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SimulationConfig:
    """Global simulation configuration"""
    
    # Identification
    name: str = "Richards_Rain_Event"
    description: str = "Rain event simulation"
    
    # Domain geometry
    Lx: float = 20.0  # Domain length (m)
    Ly: float = 5.0   # Domain height (m)
    nx: int = 80      # Number of elements in x
    ny: int = 40      # Number of elements in y
    
    # Time discretization
    dt: float = 300.0            # Time step (seconds)
    t_end: float = 168 * 3600    # End time (seconds)
    
    # Physical parameters (not soil-specific)
    g: float = 9.81             # Gravity (m/s^2)
    initial_water_table: float = 1.5  # Initial water table height (m)
    
    # Numerical parameters
    epsilon: float = 0.01       # Smoothing parameter near water table (m)
    kr_min: float = 0.01        # Minimum relative permeability
    Ss: float = 1e-4            # Specific storage coefficient (1/m)
    
    # Monitoring points
    monitor_points: list = None  # Will be set in __post_init__
    
    # Snapshot times (in seconds)
    snapshot_times: list = None  # Will be computed in __post_init__
    
    # Output settings
    output_dir: Path = Path("./results")
    save_frequency: int = 60    # Save every N steps
    
    # Solver parameters
    solver_type: str = 'gmres'
    preconditioner: str = 'ilu'
    rtol: float = 1e-8
    atol: float = 1e-10
    max_iter: int = 100
    
    def __post_init__(self):
        """Compute derived quantities and set defaults"""
        self.num_steps = int(self.t_end / self.dt)
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        
        # Set default monitoring points if not provided
        if self.monitor_points is None:
            self.monitor_points = [
                (8, 1.0, "LTC 1"),
                (10, 1.0, "LTC 2"),
                (12.5, 1.0, "LTC 3")
            ]
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    @property
    def solver_parameters(self):
        """Return solver parameters as dict for Firedrake"""
        return {
            'ksp_type': self.solver_type,
            'pc_type': self.preconditioner,
            'ksp_rtol': self.rtol,
            'ksp_atol': self.atol,
            'ksp_max_it': self.max_iter
        }