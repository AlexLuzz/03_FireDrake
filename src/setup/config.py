from dataclasses import dataclass, field
import numpy as np
from typing import  Optional
from datetime import datetime, timedelta
from ..tools import *
from .paths import ProjectPaths

@dataclass
class SimulationConfig:
    """Global simulation time configuration"""
    
    # Identification
    project_name: str = None
    user: str = "alexi"
    
    # Time discretization (can use EITHER seconds OR datetime/timedelta)
    # Option 1: Traditional seconds-based (backward compatible)
    dt: float = 3600        # Time step (seconds)
    t_end: float = 168 * 3600    # End time (seconds)

    # Option 2: Datetime-based (optional, overrides seconds if provided)
    start_datetime: Optional[datetime] = None  # Simulation start time
    end_datetime: Optional[datetime] = None     # Simulation end time (exclusive with duration_td)
    dt_td: Optional[timedelta] = None           # Time step as timedelta
    
    def __post_init__(self):
        """Initialize computed fields and handle datetime conversion"""
        self.real_time_start = datetime.now()

        # Always calculate t_end_hours (regardless of datetime or seconds-based config)
        if self.dt_td is not None:
            self.dt = self.dt_td.total_seconds()

        # 2. Calculate t_end from date range if provided
        if self.start_datetime and self.end_datetime:
            self.t_end = (self.end_datetime - self.start_datetime).total_seconds()
        elif self.t_end is None:
            self.t_end = 168 * 3600.0  # Default to 7 days if no end info

        self.num_steps = int(self.t_end / self.dt)
        self.dt_hours = self.dt / 3600.0
        self.t_end_hours = self.t_end / 3600.0
        self.time_steps = np.arange(0, self.t_end + self.dt, self.dt)

        self.num_steps = int(self.t_end / self.dt)
        self.time_steps = np.arange(0, self.t_end + self.dt, self.dt)
        
        # Initialize project paths
        self.paths = ProjectPaths(user=self.user, project_name=self.project_name)
        
        # Final output destination
        self.output_dir = self.paths.OUTPUT_DIR
        
        # Print configuration summary
        print(f"\n--- Simulation: {self.project_name} ---")
        if self.start_datetime:
            end_date = self.start_datetime + timedelta(seconds=self.t_end)
            print(f"Simulation time: {self.start_datetime.strftime('%Y-%m-%d %H:%M')} to "
                  f"{end_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Duration: {self.t_end/86400:.1f} days ({self.t_end/3600:.1f} hours)")
        else:
            print(f"Simulation duration: {self.t_end/3600:.1f} hours")
        
        print(f"  Time step: {self.dt}s ({self.dt/3600:.2f} hours)")
        print(f"  Number of steps: {self.num_steps}")

    def get_sim_duration(self):
        self.real_time_duration = datetime.now() - self.real_time_start
        return self.real_time_duration
