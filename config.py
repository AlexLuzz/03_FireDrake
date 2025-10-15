"""
Configuration module for Richards equation simulation
Contains domain geometry, time stepping, and solver parameters
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class SimulationConfig:
    """Global simulation configuration - geometry, time, and numerics"""
    
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
    num_steps: int = field(init=False)
    
    # Physical constants
    g: float = 9.81             # Gravity (m/s^2)
    
    # Numerical parameters
    epsilon: float = 0.05     # Smoothing parameter near water table (m)
    kr_min: float = 0.03      # Minimum relative permeability
    Ss: float = 1e-2          # Specific storage coefficient (1/m)
    
    # Monitoring points
    monitor_x_positions: List[float] = field(default_factory=lambda: [8.0, 10.0, 12.5])
    
    # Output settings
    output_dir: Path = Path("./results")
    save_frequency: int = 60    # Save every N steps
    
    # Solver parameters
    solver_type: str = 'gmres'
    preconditioner: str = 'hypre'  # 'ilu' for small, 'hypre' for large
    preconditioner_type: str = 'boomeramg'
    rtol: float = 1e-5
    atol: float = 1e-7
    max_iter: int = 100
    
    @property
    def solver_parameters(self):
        """Return solver parameters dict for Firedrake"""
        return {
            'ksp_type': self.solver_type,
            'pc_type': self.preconditioner,
            'pc_hypre_type': self.preconditioner_type,
            'ksp_rtol': self.rtol,
            'ksp_atol': self.atol,
            'ksp_max_it': self.max_iter
        }
    
    def __post_init__(self):
        self.num_steps = int(self.t_end / self.dt)
        self.output_dir.mkdir(parents=True, exist_ok=True)