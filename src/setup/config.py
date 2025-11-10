"""
Configuration module for Richards equation simulation
Contains domain geometry, time stepping, and solver parameters
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import  Optional
from datetime import datetime, timedelta
from ..tools import *
from .paths import ProjectPaths

@dataclass
class SimulationConfig:
    """Global simulation configuration - geometry, time, and numerics"""
    
    # Identification
    project_name: str = None
    user: str = "alexi"
    
    # Time discretization (can use EITHER seconds OR datetime/timedelta)
    # Option 1: Traditional seconds-based (backward compatible)
    dt: float = 300.0            # Time step (seconds)
    t_end: float = 168 * 3600    # End time (seconds)
    t_end_hours: float = field(init=False)  # End time in hours (computed)
    dt_hours: float = field(init=False)  # Time step in hours (computed)

    # Option 2: Datetime-based (optional, overrides seconds if provided)
    start_datetime: Optional[datetime] = None  # Simulation start time
    end_datetime: Optional[datetime] = None     # Simulation end time (exclusive with duration_td)
    duration_td: Optional[timedelta] = None     # Simulation duration (exclusive with end_datetime)
    dt_td: Optional[timedelta] = None           # Time step as timedelta
    
    # Computed fields
    num_steps: int = field(init=False)
    time_converter: Optional[object] = field(init=False, default=None)
    real_time_start: datetime = field(init=False, default=None)
    real_time_duration: datetime = None
    
    # Output settings
    output_dir: Path = Path("./results")
    data_input_dir: Path = Path("./data_input")
    data_output_dir: Path = Path("./data_output")
    save_frequency: int = 60    # Save every N steps
    
    def __post_init__(self):
        """Initialize computed fields and handle datetime conversion"""
        self.real_time_start = datetime.now()

        # If datetime parameters provided, convert to seconds
        if self.start_datetime is not None:
            self.time_converter = TimeConverter(self.start_datetime)
            
            # Convert duration/end time
            if self.end_datetime is not None and self.duration_td is not None:
                raise ValueError("Specify either 'end_datetime' or 'duration_td', not both")
            
            if self.end_datetime is not None:
                self.t_end = (self.end_datetime - self.start_datetime).total_seconds()
            elif self.duration_td is not None:
                self.t_end = self.duration_td.total_seconds()
            # else: keep the default t_end in seconds
        
        if self.dt_td is not None:
            self.dt = self.dt_td.total_seconds()

        # Always calculate t_end_hours (regardless of datetime or seconds-based config)
        self.t_end_hours = self.t_end / 3600.0
        self.dt_hours = self.dt / 3600.0
        
        # Calculate number of steps
        self.num_steps = int(self.t_end / self.dt)
        
        # Create output directory
        self.paths = ProjectPaths(user=self.user, 
                                  project_name=self.project_name 
                                  )
        self.output_dir = self.paths.OUTPUT_DIR
        
        # Print configuration summary
        if self.start_datetime is not None:
            print(f"Simulation time: {self.start_datetime.strftime('%Y-%m-%d %H:%M')} to "
                  f"{self.time_converter.to_datetime(self.t_end).strftime('%Y-%m-%d %H:%M')}")
            print(f"  Duration: {self.t_end/86400:.1f} days ({self.t_end/3600:.1f} hours)")
        else:
            print(f"Simulation duration: {self.t_end/3600:.1f} hours")
        
        print(f"  Time step: {self.dt}s ({self.dt/3600:.2f} hours)")
        print(f"  Number of steps: {self.num_steps}")

    def get_sim_duration(self):
        self.real_time_duration = datetime.now() - self.real_time_start
        return self.real_time_duration
