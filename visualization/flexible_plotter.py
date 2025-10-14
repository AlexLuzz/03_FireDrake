"""
Flexible visualization for multiple field types
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.interpolate import LinearNDInterpolator


class FlexiblePlotter:
    """Plot time series and spatial snapshots for any field type"""
    
    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data
    
    def plot_time_series(self, probe_data, field_name, ylabel, filename=None, rain_scenario=None):
        """
        Plot single field time series
        
        Args:
            probe_data: From probe_manager.get_data(field_name)
            field_name: Name of field
            ylabel: Y-axis label with units
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        times_hours = probe_data['times'] / 3600.0
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        for idx, (name, data) in enumerate(probe_data['data'].items()):
            data_array = np.array(data)
            marker_every = max(1, len(times_hours) // 50)
            ax.plot(times_hours, data_array, color=colors[idx % 3], 
                   linewidth=2.5, label=name, marker='o', markersize=2, markevery=marker_every)
        
        if rain_scenario:
            for event in rain_scenario.events:
                ax.axvspan(event.start_time, event.end_time, alpha=0.15, color='lightblue')
        
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{field_name.replace("_", " ").title()} Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, times_hours[-1]])
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_snapshots(self, snapshots, field_name, cmap, clabel, filename=None, vmin=None, vmax=None):
        """
        Plot spatial snapshots of single field
        
        Args:
            snapshots: From snapshot_manager.snapshots
            field_name: Which field to plot
            cmap: Colormap name
            clabel: Colorbar label
        """
        times = sorted(snapshots.keys())[:6]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{field_name.replace("_", " ").title()} Evolution', fontsize=16, fontweight='bold')
        
        for idx, t in enumerate(times):
            ax = axes.flat[idx]
            field_func = snapshots[t][field_name]
            self._plot_field_on_ax(ax, field_func, cmap, vmin, vmax)
            ax.set_title(f't = {t/3600:.1f}h', fontsize=11, fontweight='bold')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin or 0, vmax=vmax or 1))
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.05)
        cbar.set_label(clabel, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        return fig, axes
    
    def plot_multi_field_snapshots(self, snapshots, fields_config, filename=None):
        """
        Plot multiple fields side-by-side
        
        Args:
            fields_config: List of dicts with keys: 'name', 'cmap', 'label', 'vmin', 'vmax'
        
        Example:
            fields_config = [
                {'name': 'saturation', 'cmap': 'Blues', 'label': 'Saturation', 'vmin': 0, 'vmax': 1},
                {'name': 'chloride', 'cmap': 'YlOrRd', 'label': 'Cl⁻ (mol/m³)', 'vmin': 0, 'vmax': 100}
            ]
        """
        times = sorted(snapshots.keys())[:6]
        n_fields = len(fields_config)
        fig, axes = plt.subplots(len(times), n_fields, figsize=(6*n_fields, 3*len(times)))
        if len(times) == 1:
            axes = axes.reshape(1, -1)
        
        for row, t in enumerate(times):
            for col, field_cfg in enumerate(fields_config):
                ax = axes[row, col]
                field_func = snapshots[t][field_cfg['name']]
                self._plot_field_on_ax(ax, field_func, field_cfg['cmap'], 
                                      field_cfg.get('vmin'), field_cfg.get('vmax'))
                if row == 0:
                    ax.set_title(field_cfg['label'], fontsize=12, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f't={t/3600:.1f}h\ny (m)', fontsize=10)
                if row == len(times) - 1:
                    ax.set_xlabel('x (m)', fontsize=10)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        return fig, axes
    
    def _plot_field_on_ax(self, ax, field_func, cmap, vmin=None, vmax=None):
        """Helper: plot field on given axes"""
        values = field_func.dat.data[:]
        if vmin is not None and vmax is not None:
            values = np.clip(values, vmin, vmax)
        
        xi = np.linspace(0, self.config.Lx, 200)
        yi = np.linspace(0, self.config.Ly, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        interp = LinearNDInterpolator(self.coords, values)
        Zi = interp(Xi, Yi)
        
        ax.contourf(Xi, Yi, Zi, levels=25, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.config.Lx)
        ax.set_ylim(0, self.config.Ly)
