"""
Results visualization
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from datetime import datetime
import csv


class ResultsPlotter:
    """Creates complete visualization with time series and spatial snapshots"""
    
    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data

    def plot_complete_results(self, probe_data, snapshots=None, rain_scenario=None, 
                             filename=None, comsol_data_file=None, plot_residuals=False,
                             start_from=0.0):
        """
        Create complete results figure
        
        Args:
            probe_data: Dictionary from ProbeManager.get_data()
            snapshots: Dictionary from SnapshotManager.snapshots
            rain_scenario: Optional RainScenario for plotting rain events
            filename: Output filename (optional)
            comsol_data_file: Path to CSV with COMSOL data
            plot_residuals: Whether to plot residuals
            start_from: Time in days to start COMSOL data from (shifts time axis so this becomes t=0)
        """
        # Load and align COMSOL data if provided
        comsol_data = None
        if comsol_data_file:
            comsol_data = self._load_and_align_comsol_data(
                comsol_data_file, 
                start_from, 
                probe_data['times'][-1] / 3600.0 / 24.0  # sim end in days
            )
        
        # Calculate layout
        n_rows = 1  # Time series
        if plot_residuals and comsol_data:
            n_rows += 1  # Residuals
        if snapshots:
            n_rows += 2  # Snapshots (2 rows)
        
        fig_height = 4 * n_rows
        fig = plt.figure(figsize=(20, fig_height))
        gs = GridSpec(n_rows, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Plot time series (row 0)
        current_row = 0
        self._plot_time_series(fig, gs[current_row, :], probe_data, rain_scenario, comsol_data)
        current_row += 1
        
        # Plot residuals if requested (row 1)
        if plot_residuals and comsol_data:
            self._plot_residuals(fig, gs[current_row, :], probe_data, comsol_data)
            current_row += 1
        
        # Plot snapshots (rows 2-3)
        if snapshots:
            self._plot_snapshots(fig, gs, snapshots, start_row=current_row)
        
        # Save
        if filename is None:
            filename = self.config.output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved: {filename}")
        plt.close()
    
    def _load_and_align_comsol_data(self, csv_path, start_from, sim_duration_days):
        """
        Load COMSOL data and align it with simulation time
        
        Args:
            csv_path: Path to COMSOL CSV file
            start_from: COMSOL time (days) to map to simulation t=0
            sim_duration_days: Simulation duration in days
        
        Returns:
            Dictionary with aligned time and data arrays, or None if error
        """
        try:
            # Load raw data
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline()
                delimiter = ';' if ';' in first_line else ','
                f.seek(0)
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = [{k.strip(): v.strip().replace(',', '.') for k, v in row.items()} 
                        for row in reader]
            
            # Find time column
            time_col = next((k for k in rows[0].keys() if 'time' in k.lower()), None)
            if not time_col:
                print(f"⚠️  No time column found in {csv_path}")
                return None
            
            # Parse all data
            times_raw = np.array([float(row[time_col]) for row in rows])
            data_raw = {}
            for col in rows[0].keys():
                if col.startswith('LTC'):
                    data_raw[col] = np.array([float(row[col]) for row in rows])
            
            print(f"✓ Loaded COMSOL data: {list(data_raw.keys())}")
            print(f"  Raw COMSOL time range: {times_raw[0]:.2f} to {times_raw[-1]:.2f} days ({len(times_raw)} points)")
            
            # Step 1: Filter to start from specified time
            mask_start = times_raw >= start_from
            if not np.any(mask_start):
                print(f"⚠️  No COMSOL data found at or after start_from={start_from:.2f} days")
                return None
            
            times_filtered = times_raw[mask_start]
            data_filtered = {k: v[mask_start] for k, v in data_raw.items()}
            
            # Step 2: Shift time axis so start_from becomes t=0
            times_shifted = times_filtered - start_from
            
            # Step 3: Clip to simulation duration
            mask_clip = times_shifted <= sim_duration_days
            if not np.any(mask_clip):
                print(f"⚠️  No COMSOL data in simulation range after shifting")
                return None
            
            times_aligned = times_shifted[mask_clip]
            data_aligned = {k: v[mask_clip] for k, v in data_filtered.items()}
            
            # Create final aligned dictionary
            aligned_data = {'times': times_aligned}
            aligned_data.update(data_aligned)
            
            print(f"  Aligned COMSOL data:")
            print(f"    - Started from COMSOL day {start_from:.2f} (shifted to simulation t=0)")
            print(f"    - Clipped to simulation duration: 0 to {sim_duration_days:.2f} days")
            print(f"    - Final data points: {len(times_aligned)}")
            
            return aligned_data
            
        except Exception as e:
            print(f"⚠️  Error loading COMSOL data: {e}")
            return None
    
    def _plot_time_series(self, fig, gs_slice, probe_data, rain_scenario, comsol_data):
        """Plot water table time series"""
        ax = fig.add_subplot(gs_slice)
        
        times_h = probe_data['times'] / 3600.0
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        # Plot simulated data
        for i, (name, data) in enumerate(probe_data['data'].items()):
            ax.plot(times_h, data, color=colors[i], linewidth=2.5, 
                   label=f'{name} (Simulated)', marker='o', markersize=3, 
                   markevery=max(1, len(times_h)//30))
        
        # Plot COMSOL data if available
        if comsol_data:
            times_h_comsol = comsol_data['times'] * 24  # days to hours
            
            for i, ltc_name in enumerate(['LTC 101', 'LTC 102', 'LTC 103']):
                if ltc_name in comsol_data:
                    ax.plot(times_h_comsol, comsol_data[ltc_name],
                           color=colors[i], linewidth=2, linestyle='--',
                           label=f'{ltc_name} (COMSOL)', marker='s', markersize=4,
                           markevery=max(1, len(times_h_comsol)//20), alpha=0.8)
        
        # Rain event shading
        if rain_scenario:
            for event in rain_scenario.events:
                ax.axvspan(event.start_time, event.end_time, 
                          alpha=0.15, color='lightblue', label='Rain')
        
        ax.set_ylabel('Water Table Elevation (m)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, times_h[-1]])
        
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10)
    
    def _plot_residuals(self, fig, gs_slice, probe_data, comsol_data):
        """Plot residuals: COMSOL - Firedrake"""
        ax = fig.add_subplot(gs_slice)
        
        times_h_sim = probe_data['times'] / 3600.0
        times_h_comsol = comsol_data['times'] * 24  # days to hours
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        probe_names = list(probe_data['data'].keys())
        ltc_names = ['LTC 101', 'LTC 102', 'LTC 103']
        
        for i in range(min(len(probe_names), len(ltc_names))):
            ltc_name = ltc_names[i]
            probe_name = probe_names[i]
            
            if ltc_name not in comsol_data:
                continue
            
            # Get data
            sim_data = np.array(probe_data['data'][probe_name])
            comsol_values = comsol_data[ltc_name]
            
            # Interpolate simulation to COMSOL times
            interp_func = interp1d(times_h_sim, sim_data, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            sim_interp = interp_func(times_h_comsol)
            
            # Calculate residuals
            residuals = comsol_values - sim_interp
            
            # Plot
            ax.plot(times_h_comsol, residuals, color=colors[i], linewidth=2.5,
                   label=f'{ltc_name} - {probe_name}', marker='o', markersize=3,
                   markevery=max(1, len(times_h_comsol)//30))
            
            # Statistics
            print(f"  {ltc_name} residuals: mean={residuals.mean():.4f}m, "
                  f"std={residuals.std():.4f}m, max_abs={np.abs(residuals).max():.4f}m")
        
        # Zero line
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='Perfect match')
        
        ax.set_ylabel('Residual (m)\n[COMSOL - Firedrake]', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Residuals: COMSOL vs Firedrake', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, times_h_sim[-1]])
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10)
    
    def _plot_snapshots(self, fig, gs, snapshots, start_row):
        """Plot spatial snapshots (6 plots in 2x3 grid)"""
        sorted_times = sorted(snapshots.keys())[:6]
        while len(sorted_times) < 6:
            sorted_times.append(sorted_times[-1])
        
        x_coords = self.coords[:, 0]
        y_coords = self.coords[:, 1]
        
        for idx, t in enumerate(sorted_times):
            row = start_row + idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Get saturation
            sat = snapshots[t]['saturation'].dat.data[:]
            sat = np.clip(sat, 0, 1)
            
            # Interpolate to regular grid
            xi = np.linspace(0, self.config.Lx, 200)
            yi = np.linspace(0, self.config.Ly, 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            interp = LinearNDInterpolator(np.column_stack((x_coords, y_coords)), sat)
            Zi = interp(Xi, Yi)
            
            # Plot
            cf = ax.contourf(Xi, Yi, Zi, levels=np.linspace(0, 1, 25),
                            cmap='Blues', vmin=0, vmax=1)
            
            if idx == 0:
                contour_for_cbar = cf
            
            # Monitoring points
            if hasattr(self.config, 'monitor_x_positions'):
                colors = ['#1f77b4', '#2ca02c', '#d62728']
                for i, x in enumerate(self.config.monitor_x_positions):
                    ax.plot(x, self.config.Ly, '*', color=colors[i], 
                           markersize=12, markeredgecolor='black', markeredgewidth=0.8)
            
            ax.set_xlabel('x (m)', fontsize=10)
            ax.set_ylabel('y (m)', fontsize=10)
            ax.set_title(f't = {t/3600:.1f}h', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.set_xlim(0, self.config.Lx)
            ax.set_ylim(0, self.config.Ly)
        
        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.56])
        cbar = fig.colorbar(contour_for_cbar, cax=cbar_ax)
        cbar.set_label('Saturation', fontsize=12, fontweight='bold')