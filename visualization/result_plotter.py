"""
Results visualization - combines time series and spatial plots
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from datetime import datetime

class ResultsPlotter:
    """Creates complete visualization with time series and spatial snapshots"""
    
    def __init__(self, config, mesh):
        """
        Initialize plotter
        
        Args:
            config: SimulationConfig
            mesh: Firedrake mesh
        """
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data
    
    def plot_complete_results(self, probe_data, snapshots, rain_scenario=None, filename=None):
        """
        Create complete results figure with time series and snapshots
        
        Args:
            probe_data: Dictionary from ProbeManager.get_data()
            snapshots: Dictionary from SnapshotManager.snapshots
            rain_scenario: Optional RainScenario for plotting rain events
            filename: Output filename (optional)
        """
        # Create figure with 3x3 grid
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Top row: Time series (spanning all columns)
        self._plot_time_series(fig, gs, probe_data, rain_scenario)
        
        # Middle and bottom rows: Spatial snapshots (6 plots)
        self._plot_snapshots(fig, gs, snapshots)
        
        # Overall title
        fig.suptitle('Richards Equation: Rain Event Simulation\n'
                    'Pressure Response and Saturation Evolution', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save
        if filename is None:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = self.config.output_dir / f'richards_simulation_{now}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'")
        plt.close()
    
    def _plot_time_series(self, fig, gs, probe_data, rain_scenario=None):
        """Plot time series in top row"""
        ax = fig.add_subplot(gs[0, :])
        
        times_hours = probe_data['times'] / 3600.0
        time_series = probe_data['data']
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        # Plot each probe
        for idx, (name, data) in enumerate(time_series.items()):
            # Convert data to numpy array
            data_array = np.array(data)
            
            # Plot with markers at reasonable intervals
            marker_every = max(1, int(len(times_hours) / 50))
            ax.plot(times_hours, data_array, color=colors[idx % len(colors)], 
                   linewidth=2.5, label=name, marker='o', markersize=2,
                   markevery=marker_every)
        
        # Rain event shading if scenario provided
        if rain_scenario is not None:
            for event in rain_scenario.events:
                ax.axvspan(event.start_time, event.end_time, 
                          alpha=0.15, color='lightblue', label='Rain event')
        
        # Water table reference
        ax.axhline(y=self.config.initial_water_table, color='cyan', 
                  linestyle='--', linewidth=2, 
                  label=f'Water table (p=0)', alpha=0.7)
        
        # Formatting
        ax.set_ylabel('Water Table Elevation (m)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Water Level Sensor Readings (Depth to Water Table)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Handle legend - remove duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=10, framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
        
        # Set limits
        if len(times_hours) > 0:
            ax.set_xlim([0, times_hours[-1]])
        ax.set_ylim([1,2])
    
    def _plot_snapshots(self, fig, gs, snapshots):
        """Plot spatial snapshots in middle and bottom rows"""
        sorted_times = sorted(snapshots.keys())
        
        # Take exactly what we have, up to 6 unique times
        selected_times = sorted_times[:6] if len(sorted_times) >= 6 else sorted_times
        
        # Pad with last snapshot if we have fewer than 6
        while len(selected_times) < 6:
            selected_times.append(selected_times[-1])
        
        x_coords = self.coords[:, 0]
        y_coords = self.coords[:, 1]
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        # Track which times we've plotted to avoid showing duplicate labels
        plotted_times = set()
        
        for idx, snap_time in enumerate(selected_times):
            row = 1 + idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Get saturation data
            saturation_func = snapshots[snap_time]['saturation']
            sat_values = np.clip(saturation_func.dat.data[:], 0.0, 1.0)
            
            # Create regular grid for interpolation
            xi = np.linspace(0, self.config.Lx, 200)
            yi = np.linspace(0, self.config.Ly, 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate
            coord_points = np.column_stack((x_coords, y_coords))
            interp = LinearNDInterpolator(coord_points, sat_values)
            Zi = interp(Xi, Yi)
            
            # Plot
            levels = np.linspace(0.0, 1.0, 25)
            contourf = ax.contourf(Xi, Yi, Zi, levels=levels,
                                  cmap='Blues', extend='neither',
                                  vmin=0.0, vmax=1.0)
            
            # Store first for colorbar
            if idx == 0:
                contour_for_cbar = contourf
            
            # Contour lines
            contour_levels = [0.3, 0.5, 0.7, 0.9]
            ax.contour(Xi, Yi, Zi, levels=contour_levels,
                      colors='gray', alpha=0.3, linewidths=0.8, linestyles='--')
            
            # Water table line
            ax.axhline(y=self.config.initial_water_table, color='cyan',
                      linestyle='--', linewidth=2, label='Initial water table', alpha=0.7)
            
            # Monitoring points (if available in config)
            if hasattr(self.config, 'monitor_x_positions'):
                for pt_idx, x in enumerate(self.config.monitor_x_positions):
                    y_pos = self.config.initial_water_table
                    ax.plot(x, y_pos, '*', color=colors[pt_idx % len(colors)], 
                           markersize=15, markeredgecolor='black', 
                           markeredgewidth=1.0, zorder=10)
            
            # Formatting
            ax.set_xlabel('x (m)', fontsize=10, fontweight='bold')
            ax.set_ylabel('y (m)', fontsize=10, fontweight='bold')
            
            # Time label - mark duplicates and distinguish t=0
            time_hours = snap_time / 3600
            if snap_time == 0:
                time_label = 't = 0.0h (INITIAL)'
            elif snap_time in plotted_times:
                time_label = f't = {time_hours:.1f}h (FINAL)'
            else:
                time_label = f't = {time_hours:.1f}h'
            
            plotted_times.add(snap_time)
            
            ax.set_title(time_label, fontsize=11, fontweight='bold')
            
            ax.set_aspect('equal')
            ax.set_xlim(0, self.config.Lx)
            ax.set_ylim(0, self.config.Ly)
            
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.56])
        cbar = fig.colorbar(contour_for_cbar, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Saturation', fontsize=12, fontweight='bold')
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0%\n(dry)', '20%', '40%', '60%', '80%', '100%\n(saturated)'])