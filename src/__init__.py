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
    SourceScenario,
    Zone,
    rainfall_scenario,
    ProbeManager,
    SnapshotManager
)

# Solver modules
from .solver import (
    RichardsSolver,
    TransportSolver,
    BoundaryConditionManager
)

# Tools
from .tools import (
    CSVLoader,
    TimeConverter,
    fetch_rainfall_meteostat,
    import_results
)

# Visualization
from .visualization import (
    ResultsPlotter,
    GifAnimator,
    SimulationReport,
    plot_material_curves,
    plot_domain_geometry
)

# World
from .world import (
    Domain,
    MaterialField,
    till,
    terreau,
    till_curve_RAF,
    terreau_curve_RAF
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
    'till',
    'terreau',
    'AnalyticalTransportModel',
    'archie_resistivity',
    
    # Setup
    'SimulationConfig',
    'SourceScenario',
    'Zone',
    'rainfall_scenario',
    'ProbeManager',
    'SnapshotManager',

    # Solver
    'RichardsSolver',
    'TransportSolver', 
    'BoundaryConditionManager',
    
    # Tools
    'CSVLoader',
    'TimeConverter',
    'fetch_rainfall_meteostat',
    'import_results',

    # Visualization
    'ResultsPlotter',
    'GifAnimator',
    'SimulationReport',
    'plot_material_curves',
    'plot_domain_geometry',

    # World
    'Domain',
    'MaterialField',
    'till',
    'terreau',
    'till_curve_RAF',
    'terreau_curve_RAF'
]
