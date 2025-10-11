from config import SimulationConfig, VanGenuchtenParams
from solver.richards_solver import RichardsSolver
from visualization.plotter import ResultsPlotter

def main():
    # Configuration
    config = SimulationConfig()
    vg_params = VanGenuchtenParams()
    
    # Setup solver
    solver = RichardsSolver(config, vg_params)

    solver.set_initial_conditions()
    
    # Setup monitoring
    points = [
        (config.Lx/2, 2.0, "Point at y=2m"),
        (config.Lx/2, 3.0, "Point at y=3m"),
        (config.Lx/2, 4.0, "Point at y=4m")
    ]
    probe = PointProbe(points)
    snapshot_mgr = SnapshotManager([0, config.t_end/10, ...], solver.vg_model)
    
    # Run simulation
    print("Starting simulation...")
    solver.run(monitors=[probe, snapshot_mgr])
    print("Simulation complete!")
    
    # Visualization
    plotter = ResultsPlotter(config)
    plotter.plot_time_series(probe.get_data())
    plotter.plot_snapshots(snapshot_mgr.snapshots)
    plotter.save('richards_simulation.png')

if __name__ == "__main__":
    main()