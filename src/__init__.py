"""
FireDrake Simulation Package

A comprehensive package for Richards equation simulation with coupled transport
and geophysical modeling capabilities.

Main modules:
- physics: Domain geometry, materials, and physical models
- solver: Numerical solvers for Richards and transport equations
- setup: Configuration, monitoring, and source terms
- visualization: Result plotting and animation tools
- optimization: Parameter optimization and sensitivity analysis
- tools: Utility functions for data handling and conversion
"""

# Core physics modules
from .physics import (
    Domain,
    MaterialField, 
    VanGenuchtenModel,
    CurveBasedHydraulicModel,
    HydraulicCurves,
    till,
    terreau,
    AnalyticalTransportModel,
    archie_resistivity,
    chloride_transport,
)

# Solver modules
from .solver import (
    RichardsSolver,
    TransportSolver,
    BoundaryConditionManager
)

# Setup and configuration
from .setup import (
    SimulationConfig,
    SourceScenario,
    Zone,
    rainfall_scenario,
    ProbeManager,
    SnapshotManager
)

# Visualization
from .visualization import (
    ResultsPlotter,
    GifAnimator
)


# Tools
from .tools import (
    CSVLoader,
    TimeConverter,
    fetch_rainfall_meteostat,
    import_results
)

__version__ = "0.1.0"

# Define what gets imported with "from firedrake_sim import *"
__all__ = [
    # Physics
    'Domain',
    'MaterialField',
    'VanGenuchtenModel', 
    'CurveBasedHydraulicModel',
    'HydraulicCurves',
    'till',
    'terreau',
    'AnalyticalTransportModel',
    'archie_resistivity',
    'chloride_transport',
    
    # Solver
    'RichardsSolver',
    'TransportSolver', 
    'BoundaryConditionManager',
    
    # Setup
    'SimulationConfig',
    'SourceScenario',
    'Zone',
    'rainfall_scenario',
    'ProbeManager',
    'SnapshotManager',
    
    # Visualization
    'ResultsPlotter',
    'GifAnimator',
    
    # Tools
    'CSVLoader',
    'TimeConverter',
    'fetch_rainfall_meteostat',
    'import_results',
]
