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
    Lx: float = 15.0  # Domain length (m)
    Ly: float = 5.0   # Domain height (m)
    nx: int = 60      # Number of elements in x
    ny: int = 30      # Number of elements in y
    
    # Time discretization
    dt: float = 300.0            # Time step (seconds)
    t_end: float = 168 * 3600    # End time (seconds)
    
    # Physical parameters (not soil-specific)
    g: float = 9.81             # Gravity (m/s^2)
    initial_water_table: float = 2.5  # Initial water table height (m)
    
    # Rain event parameters
    rain_start: float = 10800.0     # Rain starts at 3 hours (s)
    rain_duration: float = 7200.0   # Rain lasts 1 hour (s)
    rain_flux: float = 200.0 / 1000.0 / 3600.0  # 200 mm/hr in m/s
    rain_x_min: float = 5.0         # Rain zone start (m)
    rain_x_max: float = 8.0         # Rain zone end (m)
    
    # Numerical parameters
    epsilon: float = 0.01       # Smoothing parameter near water table (m)
    kr_min: float = 1e-6        # Minimum relative permeability
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
    max_iter: int = 1000
    
    def __post_init__(self):
        """Compute derived quantities and set defaults"""
        self.num_steps = int(self.t_end / self.dt)
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.rain_end = self.rain_start + self.rain_duration
        
        # Set default monitoring points if not provided
        if self.monitor_points is None:
            self.monitor_points = [
                (self.Lx/2, 2.0, "Point at y=2m (below water table)"),
                (self.Lx/2, 3.0, "Point at y=3m (near water table)"),
                (self.Lx/2, 4.0, "Point at y=4m (above water table)")
            ]
        
        # Set default snapshot times if not provided
        if self.snapshot_times is None:
            self.snapshot_times = [
                0, 
                self.t_end/10, 
                self.rain_start + 1800, 
                self.rain_end + 7200, 
                self.t_end*0.7, 
                self.t_end
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