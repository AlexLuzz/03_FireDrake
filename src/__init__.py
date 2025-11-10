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

# Optimization modules
from .optimization import (
    ObservationData,
    ParameterBounds,
    AdjointOptimizer,
    create_tight_bounds,
    create_parameter_controls,

    plot_optimization_results,
    print_parameter_comparison,
    plot_residual_analysis,
    validate_optimized_parameters,
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
    BoundaryConditionManager,
    TransportSolverAnalytical
)

# Tools
from .tools import (
    CSVLoader,
    TimeConverter,
    fetch_rainfall_meteostat,
    import_results,
    load_measured_data
)

# Visualization
from .visualization import (
    ResultsPlotter,
    BasicPlotting,
    RichardsReport
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

    # Optimization
    'ObservationData',
    'ParameterBounds',
    'AdjointOptimizer',
    'create_tight_bounds',
    'create_parameter_controls',

    'plot_optimization_results',
    'print_parameter_comparison',
    'plot_residual_analysis',
    'validate_optimized_parameters',

    # Physics
    'Domain',
    'MaterialField',
    'VanGenuchtenModel', 
    'CurveBasedHydraulicModel',
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
    'TransportSolverAnalytical',
    
    # Tools
    'CSVLoader',
    'TimeConverter',
    'fetch_rainfall_meteostat',
    'import_results',
    'load_measured_data',

    # Visualization
    'ResultsPlotter',
    'BasicPlotting',
    'RichardsReport',

    # World
    'Domain',
    'MaterialField',
    'Material'
]
