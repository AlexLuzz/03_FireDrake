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

# Soils and contaminants library
from .library import (
    SoilParameters,
    ContaminantProperties,
    HydraulicCurves
)

# Core physics modules
from .physics import (
    VanGenuchtenModel,
    CurveBasedHydraulicModel,
    AnalyticalTransportModel,
    archie_resistivity,
)

# Setup and configuration
from .setup import (
    SimulationConfig,
    SourceManager,
    SimulationMonitor
)

# Solver modules
from .solver import (
    RichardsSolver,
    TransportSolver,
    BoundaryConditionManager,
    TransportSolverAnalytical
)

# Visualization
from .visualization import (
    HydrogeoSimulationReport
)

# World
from .world import (
    Domain,
    MaterialField,
    Material
)

__version__ = "0.1.0"

# Define what gets imported with "from firedrake_sim import *"
__all__ = [
    # Library   
    'SoilParameters',
    'ContaminantProperties',
    'HydraulicCurves',

    # Physics
    'Domain',
    'MaterialField',
    'VanGenuchtenModel', 
    'CurveBasedHydraulicModel',
    'AnalyticalTransportModel',
    'archie_resistivity',
    
    # Setup
    'SimulationConfig',
    'SourceManager',
    'SimulationMonitor',

    # Solver
    'RichardsSolver',
    'TransportSolver', 
    'BoundaryConditionManager',
    'TransportSolverAnalytical',
    
    # Visualization
    'HydrogeoSimulationReport',

    # World
    'Domain',
    'MaterialField',
    'Material'
]
