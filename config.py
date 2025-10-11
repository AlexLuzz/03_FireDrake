# config.py
from dataclasses import dataclass

@dataclass
class VanGenuchtenParams:
    theta_r: float = 0.05
    theta_s: float = 0.40
    alpha: float = 2.0
    n: float = 2.0
    l_param: float = 0.5
    
    @property
    def m(self):
        return 1.0 - 1.0 / self.n

@dataclass
class SimulationConfig:
    # Domain
    Lx: float = 20.0
    Ly: float = 4.0
    nx: int = 80
    ny: int = 40
    
    # Time
    dt: float = 30.0
    t_end: float = 13 * 3600
    
    # Physics
    Ks: float = 1e-6
    initial_water_table: float = 2.0
    
    # Rain event
    rain_start: float = 10800.0
    rain_duration: float = 7200.0
    rain_pressure: float = 0.2
    rain_x_min: float = 3.0
    rain_x_max: float = 7.0