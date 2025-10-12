
from firedrake import RectangleMesh, FunctionSpace
from config import SimulationConfig
from physics import *
from solver import *
from visualization import *

def main():
    """Main simulation function"""
    
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = SimulationConfig(
        # You can modify config parameters here if needed
        dt=1800,
        t_end=3*86400,
    )
    
    # ==========================================
    # 2. CREATE MESH AND FUNCTION SPACE
    # ==========================================
    mesh = RectangleMesh(config.nx, config.ny, config.Lx, config.Ly)
    V = FunctionSpace(mesh, "CG", 1)

    # ==========================================
    # 3. CREATE SOIL MATERIALS
    # ==========================================
    
    # OPTION 1: Single homogeneous soil (original behavior)
    # ----------------------------------------------------
    soil = SoilMaterial.create_default(
        epsilon=config.epsilon,
        kr_min=config.kr_min,
        Ss=config.Ss
    )
    domain = Domain.homogeneous(mesh, soil)
    
    # OPTION 2: Two horizontal layers (dirt on top, till below)
    # ----------------------------------------------------------
    # dirt = SoilMaterial.create_default(
    #     epsilon=config.epsilon, kr_min=config.kr_min, Ss=config.Ss
    # )
    # till = SoilMaterial.create_sandy_loam(
    #     epsilon=config.epsilon, kr_min=config.kr_min, Ss=config.Ss
    # )
    # domain = Domain.two_layer(mesh, dirt, till, interface_y=2.5)
    
    # OPTION 3: Multiple horizontal layers
    # -------------------------------------
    # from physics.van_genuchten import VanGenuchtenParams, VanGenuchtenModel
    # 
    # # Create custom materials
    # gravel_params = VanGenuchtenParams(theta_r=0.02, theta_s=0.35, alpha=5.0, n=2.5)
    # gravel_model = VanGenuchtenModel(gravel_params, epsilon=config.epsilon)
    # gravel = SoilMaterial("Gravel", gravel_model, Ks=1e-4)
    # 
    # till = SoilMaterial.create_sandy_loam(...)
    # dirt = SoilMaterial.create_default(...)
    # 
    # domain = Domain.horizontal_layers(mesh, [
    #     (gravel, 1.0),  # 0-1m: gravel base
    #     (till, 2.0),    # 1-3m: till
    #     (dirt, 2.0)     # 3-5m: dirt/topsoil
    # ])
    
    # OPTION 4: Left-right split (vertical interface)
    # ------------------------------------------------
    # till = SoilMaterial.create_sandy_loam(...)
    # dirt = SoilMaterial.create_default(...)
    # domain = Domain.left_right_split(mesh, till, dirt, interface_x=7.5)
    
    # OPTION 5: Add geometric features to base layer
    # -----------------------------------------------
    # sand = SoilMaterial.create_default(...)
    # clay = SoilMaterial.create_clay(...)
    # 
    # domain = Domain.homogeneous(mesh, sand)
    # # Add clay lens in the middle
    # domain.add_box(clay, x_min=5, x_max=10, y_min=1.5, y_max=2.5, name="clay_lens")
    # # Add circular inclusion
    # domain.add_circle(clay, center_x=12, center_y=3, radius=0.8, name="clay_pocket")
    
    # OPTION 6: Custom polygonal regions (MOST FLEXIBLE!)
    # ---------------------------------------------------
    # Define any shape with polygon vertices
    # soil = SoilMaterial.create_default(...)
    # domain = Domain.homogeneous(mesh, soil)
    # 
    # # Example: L-shaped region
    # vertices = [(0,0), (10,0), (10,3), (5,3), (5,5), (0,5)]
    # domain.add_polygon(soil, vertices, name="L_shape")
    # 
    # # Example: Notched rectangle (like complex geometry)
    # vertices = [
    #     (0, 0), (22, 0), (22, 5), (11, 5),
    #     (11, 4.3), (10, 4.3), (10, 5), (0, 5)
    # ]
    # domain.add_polygon(soil, vertices, name="notched_rectangle")
    
    # Print domain summary
    domain.print_summary()
    
    # Optional: Visualize material distribution
    # domain.visualize_materials(V, save_path=config.output_dir / "material_map.png")
    
    # ==========================================
    # 4. CREATE BOUNDARY CONDITION MANAGER
    # ==========================================
    bc_manager = BoundaryConditionManager(V, config)
    
    # ==========================================
    # 5. CREATE MONITORING
    # ==========================================
    probe_manager = ProbeManager(config.monitor_points)
    snapshot_manager = SnapshotManager(config.snapshot_times, domain)
    
    # ==========================================
    # 6. CREATE SOLVER
    # ==========================================
    solver = RichardsSolver(mesh, V, domain, bc_manager, config)
        
    # ==========================================
    # 7. RUN SIMULATION
    # ==========================================
    solver.run(probe_manager=probe_manager, snapshot_manager=snapshot_manager)
    
    # ==========================================
    # 8. VISUALIZE RESULTS
    # ==========================================
    print("Creating visualization...")
    plotter = ResultsPlotter(config, mesh)
    plotter.plot_complete_results(
        probe_data=probe_manager.get_data(),
        snapshots=snapshot_manager.snapshots
    )

if __name__ == "__main__":
    main()