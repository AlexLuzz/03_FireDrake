"""
Create animated GIFs from simulation snapshots using matplotlib only
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import LinearNDInterpolator


class GifAnimator:
    """Create animated GIF from snapshots using matplotlib animation"""
    
    def __init__(self, snapshots, config, mesh):
        """
        Initialize animator
        
        Args:
            snapshots: Dictionary from SnapshotManager {time: {field_name: Function}}
            config: SimulationConfig
            mesh: Firedrake mesh
        """
        self.snapshots = snapshots
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data
    
    def create_animation(self, field_name='saturation', filename='animation.gif', 
                        fps=2, dpi=100, cmap='Blues', clabel='Saturation',
                        vmin=None, vmax=None):
        """
        Create GIF animation of field evolution
        
        Args:
            field_name: Which field to animate (must exist in snapshots)
            filename: Output filename
            fps: Frames per second
            dpi: Resolution
            cmap: Colormap
            clabel: Colorbar label
            vmin, vmax: Value limits
        
        Example:
            animator = GifAnimator(snapshot_manager.snapshots, config, mesh)
            animator.create_animation('saturation', 'saturation.gif', fps=3)
            animator.create_animation('chloride', 'chloride.gif', cmap='YlOrRd', 
                                     clabel='Cl⁻ (mol/m³)', vmin=0, vmax=100)
        """
        times = sorted(self.snapshots.keys())
        
        # Check if field exists
        if field_name not in self.snapshots[times[0]]:
            available = list(self.snapshots[times[0]].keys())
            raise ValueError(f"Field '{field_name}' not in snapshots. Available: {available}")
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create interpolation grid
        xi = np.linspace(0, self.config.Lx, 200)
        yi = np.linspace(0, self.config.Ly, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Initialize plot
        field_func = self.snapshots[times[0]][field_name]
        values = field_func.dat.data[:]
        if vmin is not None and vmax is not None:
            values = np.clip(values, vmin, vmax)
        
        interp = LinearNDInterpolator(self.coords, values)
        Zi = interp(Xi, Yi)
        
        contourf = ax.contourf(Xi, Yi, Zi, levels=25, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label(clabel, fontsize=12, fontweight='bold')
        
        title = ax.set_title(f't = 0.0h', fontsize=14, fontweight='bold')
        ax.set_xlabel('x (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(0, self.config.Lx)
        ax.set_ylim(0, self.config.Ly)
        
        def update(frame_idx):
            """Update function for animation"""
            t = times[frame_idx]
            
            # Clear previous contour
            ax.clear()
            
            # Get new data
            field_func = self.snapshots[t][field_name]
            values = field_func.dat.data[:]
            if vmin is not None and vmax is not None:
                values = np.clip(values, vmin, vmax)
            
            interp = LinearNDInterpolator(self.coords, values)
            Zi = interp(Xi, Yi)
            
            # Plot
            ax.contourf(Xi, Yi, Zi, levels=25, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f't = {t/3600:.1f}h', fontsize=14, fontweight='bold')
            ax.set_xlabel('x (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('y (m)', fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.set_xlim(0, self.config.Lx)
            ax.set_ylim(0, self.config.Ly)
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(times), 
                                      interval=1000/fps, repeat=True)
        
        # Save as GIF using PillowWriter (no external dependencies)
        writer = animation.PillowWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=dpi)
        
        plt.close(fig)
        print(f"✓ Animation saved: {filename} ({len(times)} frames at {fps} fps)")
        
        return anim