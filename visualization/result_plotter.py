"""
Results visualization - combines time series and spatial plots
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
        """
        Initialize plotter
        
        Args:
            config: SimulationConfig
            mesh: Firedrake mesh
        """
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data
    
    def plot_complete_results(self, probe_data, snapshots=None, rain_scenario=None, 
                             filename=None, measured_data_csv=None, plot_residuals=False):
        """
        Create complete results figure with time series and snapshots
        
        Args:
            probe_data: Dictionary from ProbeManager.get_data()
            snapshots: Dictionary from SnapshotManager.snapshots
            rain_scenario: Optional RainScenario for plotting rain events
            filename: Output filename (optional)
            measured_data_csv: Optional path to CSV with measured data (e.g., 'RAF_PZ_CG.csv')
            plot_residuals: Whether to plot residuals (requires measured_data_csv)
        """
        # Smart grid layout calculation
        # Layout: Water level (1 row) + Residuals (1 row if requested) + Snapshots (2 rows if provided)
        n_time_series_rows = 1  # Water level time series
        n_residual_rows = 1 if plot_residuals and measured_data_csv else 0
        n_snapshot_rows = 2 if snapshots is not None else 0
        total_rows = n_time_series_rows + n_residual_rows + n_snapshot_rows
        
        # Adjust figure height based on number of rows
        fig_height = 5 * total_rows if total_rows > 0 else 10
        fig = plt.figure(figsize=(20, fig_height))
        gs = GridSpec(total_rows, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Track which row we're on
        current_row = 0
        
        # Row 0: Water level time series (spanning all columns)
        self._plot_time_series(fig, gs, probe_data, rain_scenario, measured_data_csv, row=current_row)
        current_row += 1

        # Row 1 (optional): Residuals
        if plot_residuals and measured_data_csv:
            self._plot_residuals(fig, gs, probe_data, measured_data_csv, row=current_row)
            current_row += 1

        # Rows 2-3 (optional): Spatial snapshots (6 plots in 2 rows)
        if snapshots is not None:
            self._plot_snapshots(fig, gs, snapshots, start_row=current_row)
        
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
    
    def _plot_time_series(self, fig, gs, probe_data, rain_scenario=None, measured_data_csv=None, row=0):
        """Plot time series with optional measured data overlay"""
        ax = fig.add_subplot(gs[row, :])
        
        times_hours = probe_data['times'] / 3600.0
        time_series = probe_data['data']
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        # Plot simulated data for each probe
        for idx, (name, data) in enumerate(time_series.items()):
            # Convert data to numpy array
            data_array = np.array(data)
            
            # Plot with markers at reasonable intervals
            marker_every = max(1, int(len(times_hours) / 50))
            ax.plot(times_hours, data_array, color=colors[idx % len(colors)], 
                   linewidth=2.5, label=f'{name} (Simulated)', marker='o', markersize=2,
                   markevery=marker_every, linestyle='-')
        
        # Load and plot measured data if provided
        if measured_data_csv is not None:
            measured_data = self._load_measured_data(measured_data_csv)
            if measured_data is not None:
                # Map LTC columns to probe indices
                ltc_mapping = {
                    'LTC 101': 0,  # Maps to first probe (LTC 1)
                    'LTC 102': 1,  # Maps to second probe (LTC 2)
                    'LTC 103': 2   # Maps to third probe (LTC 3)
                }
                
                for ltc_name, probe_idx in ltc_mapping.items():
                    if ltc_name in measured_data and probe_idx < len(colors):
                        times_days = measured_data['times']
                        times_hours_measured = times_days * 24.0  # Convert days to hours
                        values = measured_data[ltc_name]
                        
                        ax.plot(times_hours_measured, values, 
                               color=colors[probe_idx],
                               linewidth=2.0, 
                               label=f'{ltc_name} (Measured)',
                               linestyle='--',  # Dashed line for measured data
                               marker='s',
                               markersize=3,
                               markevery=max(1, len(times_hours_measured) // 20),
                               alpha=0.8)
        
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
        
        # Handle legend - remove duplicates and place outside plot
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='center left', bbox_to_anchor=(1.01, 0.5), 
                 fontsize=10, framealpha=0.95, borderaxespad=0)
        
        ax.grid(True, alpha=0.3)
        
        # Set limits
        if len(times_hours) > 0:
            ax.set_xlim([0, times_hours[-1]])
        ax.set_ylim([1,2])

    def _plot_residuals(self, fig, gs, probe_data, measured_data_csv, row=1):
        """
        Plot residuals (Measured - Simulated) time series
        
        Residuals are calculated as: LTC 101 - LTC 1, LTC 102 - LTC 2, LTC 103 - LTC 3
        Uses interpolation to match measured and simulated time points
        """
        ax = fig.add_subplot(gs[row, :])
        
        # Load measured data
        measured_data = self._load_measured_data(measured_data_csv)
        if measured_data is None:
            ax.text(0.5, 0.5, 'Measured data not available', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Residuals (Measured - Simulated)', fontsize=14, fontweight='bold')
            return
        
        times_hours_sim = probe_data['times'] / 3600.0
        time_series_sim = probe_data['data']
        
        times_hours_meas = measured_data['times'] * 24.0  # Convert days to hours
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        ltc_numbers = [1, 2, 3]  # LTC probe numbers
        ltc_measured_names = ['LTC 101', 'LTC 102', 'LTC 103']
        
        # Debug: print available probe names
        print("\n=== Residuals Calculation Debug ===")
        print(f"Available simulated probe names: {list(time_series_sim.keys())}")
        print(f"Available measured data columns: {[k for k in measured_data.keys() if k != 'times']}")
        
        # Calculate and plot residuals for each probe
        for idx, (ltc_num, ltc_meas_name) in enumerate(zip(ltc_numbers, ltc_measured_names)):
            if ltc_meas_name not in measured_data:
                print(f"  ⚠️  Skipping {ltc_meas_name}: not in measured data")
                continue
            
            # Find matching simulated probe (look for "LTC {ltc_num}" in probe name)
            sim_probe_name = None
            for probe_name in time_series_sim.keys():
                if f"LTC {ltc_num}" in probe_name:
                    sim_probe_name = probe_name
                    break
            
            if sim_probe_name is None:
                print(f"  ⚠️  Skipping LTC {ltc_num}: no matching simulated probe")
                continue
            
            print(f"  ✓ Matching {ltc_meas_name} with {sim_probe_name}")
            
            sim_data = np.array(time_series_sim[sim_probe_name])
            meas_data = measured_data[ltc_meas_name]
            
            # Interpolate both to a common time grid (use measured time points)
            # This assumes measured data is sparser than simulated
            
            # Create interpolator for simulated data
            sim_interp = interp1d(times_hours_sim, sim_data, 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # Evaluate simulated data at measured time points
            sim_at_meas_times = sim_interp(times_hours_meas)
            
            # Calculate residuals: Measured - Simulated
            residuals = meas_data - sim_at_meas_times
            
            print(f"    Residual range: [{np.min(residuals):.4f}, {np.max(residuals):.4f}] m")
            
            # Plot residuals
            marker_every = max(1, len(times_hours_meas) // 30)
            ax.plot(times_hours_meas, residuals, 
                   color=colors[idx],
                   linewidth=2.5, 
                   label=f'{ltc_meas_name} - LTC {ltc_num}',
                   marker='o',
                   markersize=4,
                   markevery=marker_every,
                   alpha=0.85)
        
        print("=== End Debug ===\n")
        
        # Zero reference line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, 
                  label='Perfect match', alpha=0.5)
        
        # Formatting
        ax.set_ylabel('Residual (m)\n[Measured - Simulated]', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Residuals Analysis: Measured vs Simulated Water Level', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Handle legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='center left', bbox_to_anchor=(1.01, 0.5), 
                 fontsize=10, framealpha=0.95, borderaxespad=0)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set limits
        if len(times_hours_meas) > 0:
            ax.set_xlim([0, times_hours_meas[-1]])
            
        # Auto-scale y-axis but add some padding
        ax.margins(y=0.1)
    
    def _plot_snapshots(self, fig, gs, snapshots, start_row=2):
        """
        Plot spatial snapshots in 2 rows × 3 columns grid
        
        Args:
            fig: Figure object
            gs: GridSpec object
            snapshots: Dictionary of snapshots
            start_row: Starting row index in the grid (default 2)
        """
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
            # Calculate row and column: 2 rows × 3 columns starting at start_row
            row = start_row + idx // 3
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
    
    def _load_measured_data(self, csv_path):
        """
        Load measured data from CSV file
        
        Args:
            csv_path: Path to CSV file (e.g., 'RAF_PZ_CG.csv')
        
        Returns:
            Dictionary with 'times' (days) and LTC columns, or None if error
        
        Expected format:
            Time (d);LTC 101;LTC 102;LTC 103
            0;1.2;1.2;1.2
            0.1;1.2;1.2;1.2
        """
        try:
            # Read CSV with UTF-8 BOM handling and semicolon delimiter
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                # Auto-detect delimiter
                first_line = f.readline()
                f.seek(0)
                delimiter = ';' if ';' in first_line else ','
                
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = list(reader)
            
            if not rows:
                print(f"Warning: No data in {csv_path}")
                return None
            
            # Strip whitespace from column names
            rows = [{k.strip(): v for k, v in row.items()} for row in rows]
            
            # Find time column (flexible naming)
            time_col = None
            for possible_name in ['Time (d)', 'Time', 'time', 'Time(d)', 'time(d)']:
                if possible_name in rows[0]:
                    time_col = possible_name
                    break
            
            if time_col is None:
                print(f"Warning: Could not find time column in {csv_path}")
                print(f"Available columns: {list(rows[0].keys())}")
                return None
            
            # Parse data (handle comma as decimal separator)
            def parse_float(value_str):
                return float(value_str.strip().replace(',', '.'))
            
            data = {'times': np.array([parse_float(row[time_col]) for row in rows])}
            
            # Load all LTC columns
            for col_name in rows[0].keys():
                if col_name.startswith('LTC'):
                    data[col_name] = np.array([parse_float(row[col_name]) for row in rows])
            
            print(f"✓ Loaded measured data from '{csv_path}'")
            print(f"  Columns: {[k for k in data.keys() if k != 'times']}")
            print(f"  Time range: {data['times'][0]:.2f} to {data['times'][-1]:.2f} days")
            
            return data
            
        except FileNotFoundError:
            print(f"Warning: Measured data file not found: {csv_path}")
            return None
        except Exception as e:
            print(f"Warning: Error loading measured data from {csv_path}: {e}")
            return None