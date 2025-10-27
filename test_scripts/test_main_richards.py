"""
Test script for Richards equation simulation using the FireDrake package.

This script demonstrates the complete workflow for running a Richards equation
simulation with rainfall scenarios, monitoring, and visualization.

Usage:
    python test_main_richards.py

The script will:
1. Set up a simulation configuration
2. Define rainfall zones and scenarios
3. Create domain geometry and assign materials
4. Set up boundary conditions and monitoring
5. Run the simulation
6. Generate plots and results
"""

import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to Python path to import the main package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firedrake import FunctionSpace
from src import *

def test_richards_simulation():
    """
    Test function for Richards equation simulation.
    
    This function demonstrates the complete workflow and can be used
    for testing modifications to the simulation components.
    """
    print("=" * 60)
    print("Starting Richards Equation Test Simulation")
    print("=" * 60)
    
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    print("Setting up configuration...")
    config = SimulationConfig(
        name="Test_Richards_Simulation",
        start_datetime=datetime(2024, 5, 1),
        end_datetime=datetime(2024, 5, 3),  # Shorter simulation for testing
        dt_td=timedelta(hours=6)
    )
    print(f"Simulation period: {config.start_datetime} to {config.end_datetime}")
    print(f"Time step: {config.dt_td}")
    
    # ==========================================
    # 2. RAIN SCENARIO
    # ==========================================
    print("\nSetting up rainfall scenario...")
    rain_zones = [
        {'name': 'grass', 'x_min': 0.0, 'x_max': 8.0, 'multiplier': 1.0},
        {'name': 'green_infrastructure', 'x_min': 9.0, 'x_max': 11.0, 'multiplier': 6.0},
    ]
    
    try:
        rain_source = rainfall_scenario(
            from_date=config.start_datetime,
            to_date=config.end_datetime,
            meteostat_station='SOK6B',
            meteostat_agg_hours=6,
            zones=rain_zones
        )
        print("Rainfall scenario created successfully")
    except Exception as e:
        print(f"Warning: Could not create rainfall scenario: {e}")
        print("Continuing with dummy rainfall scenario...")
        # Create a simple dummy scenario for testing
        from src.setup.source_term import SourceScenario, Zone
        rain_source = SourceScenario()
        for zone_data in rain_zones:
            zone = Zone(
                name=zone_data['name'],
                x_min=zone_data['x_min'], 
                x_max=zone_data['x_max'],
                multiplier=zone_data['multiplier']
            )
            rain_source.add_zone(zone)
    
    # ==========================================
    # 3. GEOMETRY (Domain - pure geometry)
    # ==========================================
    print("\nCreating domain geometry...")
    domain = Domain(nx=40, ny=20, Lx=20.0, Ly=5.0)  # Smaller mesh for testing
    domain.add_rectangle("GI", 9.0, 11.0, 4.0, 5.0)
    print(f"Domain created: {domain.Lx}m x {domain.Ly}m with {domain.nx} x {domain.ny} elements")
    
    # ==========================================
    # 4. MATERIALS (properties)
    # ==========================================
    print("\nAssigning materials...")
    domain.assign("base", till())
    domain.assign("GI", terreau())
    domain.validate()
    print("Materials assigned and validated")
    
    # ==========================================
    # 5. MAPPING (connect materials to domain)
    # ==========================================
    print("\nCreating field mapping...")
    V = FunctionSpace(domain.mesh, "CG", 1)
    field_map = MaterialField(domain, V)
    print(f"Function space created with {V.dim()} degrees of freedom")
    
    # ==========================================
    # 6. BOUNDARY CONDITIONS
    # ==========================================
    print("\nSetting up boundary conditions...")
    bc_manager = BoundaryConditionManager(
        V,
        left_wt=1.2,
        right_wt=1.8
    )
    print("Boundary conditions configured")
    
    # ==========================================
    # 7. MONITORING
    # ==========================================
    print("\nSetting up monitoring...")
    probe_manager = ProbeManager(domain.mesh)
    
    snapshot_times = [
        0.0,
        5*3600.0,
        config.t_end * 0.1,
        config.t_end * 0.3,
        config.t_end * 0.7,
        config.t_end
    ]
    snapshot_manager = SnapshotManager(snapshot_times)
    print(f"Monitoring configured with {len(snapshot_times)} snapshots")
    
    # ==========================================
    # 8. SOLVER
    # ==========================================
    print("\nInitializing solver...")
    solver = RichardsSolver(
        domain=domain,
        V=V,
        field_map=field_map,
        source_scenario=rain_source,
        bc_manager=bc_manager,
        config=config
    )
    print("Solver initialized successfully")
    
    # ==========================================
    # 9. RUN SIMULATION
    # ==========================================
    print("\n" + "="*40)
    print("RUNNING SIMULATION")
    print("="*40)
    
    try:
        solver.run(probe_manager, snapshot_manager)
        print("Simulation completed successfully!")
    except Exception as e:
        print(f"Error during simulation: {e}")
        print("Simulation failed - check configuration and dependencies")
        return False
    
    # ==========================================
    # 10. VISUALIZATION
    # ==========================================
    print("\nGenerating results...")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        plotter = ResultsPlotter(
            config, domain.mesh, 
            probe_manager=probe_manager, 
            rain_scenario=rain_source, 
            domain=domain,
            snapshot_manager=snapshot_manager
        )
        
        # Configure what to plot (simplified for testing)
        plotting_config = {
            'time_series': True,
            'plot_comsol_comparison': False,
            'plot_measured_comparison': False,
            'plot_snapshots': True
        }
        
        # Create output directory if it doesn't exist
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plotter.plot_complete_results(
            filename=output_dir / f'test_simulation_{now}.png',
            plotting_config=plotting_config
        )
        
        print(f"Results saved to: {output_dir / f'test_simulation_{now}.png'}")
        
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
        print("Simulation data should still be available for manual analysis")
    
    print("\n" + "="*60)
    print("TEST SIMULATION COMPLETED")
    print("="*60)
    
    return True


def run_basic_tests():
    """Run basic tests to verify package imports and functionality."""
    print("Running basic package tests...")
    
    # Test imports
    try:
        from src.physics import Domain, till, terreau
        from src.setup import SimulationConfig
        print("✓ All core imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test basic domain creation
    try:
        domain = Domain(nx=10, ny=10, Lx=5.0, Ly=5.0)
        print("✓ Domain creation successful")
    except Exception as e:
        print(f"✗ Domain creation failed: {e}")
        return False
    
    # Test material assignment
    try:
        domain.assign("base", till())
        print("✓ Material assignment successful")
    except Exception as e:
        print(f"✗ Material assignment failed: {e}")
        return False
    
    # Test configuration
    try:
        config = SimulationConfig(
            name="test",
            start_datetime=datetime(2024, 1, 1),
            end_datetime=datetime(2024, 1, 2),
            dt_td=timedelta(hours=1)
        )
        print("✓ Configuration creation successful")
    except Exception as e:
        print(f"✗ Configuration creation failed: {e}")
        return False
    
    print("All basic tests passed!")
    return True


if __name__ == "__main__":
    print("FireDrake Test Script")
    print("====================")
    
    # First run basic tests
    if not run_basic_tests():
        print("Basic tests failed. Please check your installation.")
        sys.exit(1)
    
    print("\nBasic tests passed. Starting full simulation test...")
    
    # Run the full simulation test
    success = test_richards_simulation()
    
    if success:
        print("\nAll tests completed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the output above.")
        sys.exit(1)