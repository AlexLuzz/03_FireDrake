"""
Results visualization
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d
from datetime import datetime
from ..tools.import_results import (
    load_comsol_data, load_measured_data, calculate_residuals, 
    preview_data
)

class ResultsPlotter:
    """Creates complete visualization with time series and spatial snapshots"""
    
    def __init__(self, config, mesh, 
                 probe_manager=None,
                 rain_scenario=None, 
                 domain=None,
                 snapshot_manager=None
                 ):
        """
        Initialize plotter with configuration and optional managers
        
        Args:
            config: SimulationConfig object
            mesh: Firedrake mesh
            probe_manager: ProbeManager instance (optional)
            rain_scenario: Rain/source scenario (optional)
            domain: Domain object for bounds (optional, can be inferred from probe_manager)
            snapshot_manager: SnapshotManager instance (optional) 
        """
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data
        self.probe_manager = probe_manager
        self.snapshot_manager = snapshot_manager
        self.rain_scenario = rain_scenario
        self.domain = domain
    
    
    def plot_complete_results(self, filename=None, plot_dates=True, plotting_config=None):
        """
        Create complete results figure with configurable plotting options
        
        Args:
            filename: Output filename (optional)
            plot_dates: If True and config has time_converter, plot with datetime x-axis (default: True)
            plotting_config: Dictionary with plotting configuration:
                {
                    'time_series': True,                    # Always plot time series
                    'plot_comsol_comparison': True,         # Plot COMSOL comparison and residuals
                    'comsol_data_file': 'path/to/comsol.csv',
                    'comsol_ref_date': datetime(2024, 2, 22),
                    'plot_measured_comparison': True,       # Plot measured data comparison and residuals  
                    'measured_data_file': 'path/to/measured.csv',
                    'measured_ref_date': None,              # Auto-infer from config if None
                    'measured_offsets': {'LTC 101': 0.6},   # or single float
                    'plot_snapshots': True                  # Plot snapshots if snapshot_manager available
                }
        """
        # Set default plotting config
        default_config = {
            'time_series': True,
            'plot_comsol_comparison': False,
            'plot_measured_comparison': False,
            'plot_snapshots': True,
            'field_name': 'water_table',           # Default field to plot
            'field_units': 'm',                    # Default units
            'field_label': 'Water Table Elevation', # Default label
            'colormap': 'Blues'                    # Default colormap for snapshots
        }
        
        if plotting_config is None:
            plotting_config = default_config
        else:
            # Merge with defaults
            for key, default_value in default_config.items():
                if key not in plotting_config:
                    plotting_config[key] = default_value
        
        # Get probe data (required)
        probe_data = self.probe_manager.get_data() if self.probe_manager else None
        if probe_data is None:
            raise ValueError("probe_manager must be set and contain data")
        
        # Get snapshots if requested and available
        snapshots = None
        if plotting_config['plot_snapshots'] and self.snapshot_manager:
            snapshots = self.snapshot_manager.snapshots
        
        # Determine if we can use datetime axis
        use_datetime = plot_dates and hasattr(self.config, 'time_converter') and self.config.time_converter is not None
        
        # Load COMSOL data if requested
        comsol_data = None
        if plotting_config['plot_comsol_comparison']:
            try:
                # Calculate start_from automatically if we have datetime config
                start_from = 0.0
                if hasattr(self.config, 'start_datetime') and self.config.start_datetime:
                    # Use provided ref_date or import_results default
                    from tools.import_results import DEFAULT_COMSOL_REF_DATE
                    comsol_ref_date = plotting_config.get('comsol_ref_date', DEFAULT_COMSOL_REF_DATE)
                    start_from = (self.config.start_datetime - comsol_ref_date).total_seconds() / 86400.0
                    print(f"ℹ️  Auto-calculated start_from: {start_from:.1f} days")
                
                # load_comsol_data will use default file if csv_path=None
                comsol_data = load_comsol_data(
                    csv_path=plotting_config.get('comsol_data_file'),
                    start_from_days=start_from,
                    sim_duration_days=probe_data['times'][-1] / 3600.0 / 24.0
                )
            except Exception as e:
                print(f"⚠️  Could not load COMSOL data: {e}")
        
        # Load measured data if requested
        measured_data = None
        if plotting_config['plot_measured_comparison']:
            try:
                # Auto-infer ref_date from config if not provided
                ref_date = plotting_config.get('measured_ref_date')
                if ref_date is None and hasattr(self.config, 'start_datetime'):
                    ref_date = self.config.start_datetime
                    print(f"ℹ️  Using config start_datetime as measured_ref_date: {ref_date.strftime('%Y-%m-%d')}")
                
                # Calculate simulation period for filtering
                start_datetime = end_datetime = None
                if ref_date and hasattr(self.config, 'time_converter'):
                    sim_duration_days = probe_data['times'][-1] / 3600.0 / 24.0
                    start_datetime = ref_date
                    end_datetime = ref_date + pd.Timedelta(days=sim_duration_days)
                
                # load_measured_data will use defaults if parameters are None
                measured_data = load_measured_data(
                    csv_path=plotting_config.get('measured_data_file'),
                    time_converter=self.config.time_converter,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    offsets=plotting_config.get('measured_offsets'),  # Will use default if None
                    smooth_window=30,
                    hampel_window=120,
                    hampel_sigma=3.0
                )
            except Exception as e:
                print(f"⚠️  Could not load measured data: {e}")
        
        # Calculate layout
        n_rows = 1  # Time series (always present)
        
        # Add residual rows if requested and data available
        if plotting_config['plot_comsol_comparison'] and comsol_data is not None:
            n_rows += 1
        if plotting_config['plot_measured_comparison'] and measured_data is not None:
            n_rows += 1
        if snapshots:
            n_rows += 2  # Snapshots (2 rows)
        
        fig_height = 4 * n_rows
        fig = plt.figure(figsize=(20, fig_height))
        gs = GridSpec(n_rows, 3, figure=fig, hspace=0.25, wspace=0.35)
        
        # Overall title
        title = 'Richards Equation Simulation - Water Table Response'
        if use_datetime and self.config.start_datetime:
            title += f'\n{self.config.start_datetime.strftime("%Y-%m-%d")} to {self.config.time_converter.to_datetime(self.config.t_end).strftime("%Y-%m-%d")}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        # Track current row and which is the last time-series row (for x-axis labels)
        current_row = 0
        last_timeseries_row = n_rows - (2 if snapshots else 0) - 1
        
        # Plot time series (row 0)
        is_last = (current_row == last_timeseries_row)
        self._plot_time_series(fig, gs[current_row, :], probe_data, self.rain_scenario, 
                              comsol_data, measured_data, plotting_config, show_xlabel=is_last,
                              use_datetime=use_datetime)
        current_row += 1
        
        # Plot residuals if requested
        if plotting_config['plot_comsol_comparison'] and comsol_data is not None:
            is_last = (current_row == last_timeseries_row)
            self._plot_residuals(fig, gs[current_row, :], probe_data, comsol_data, 
                                'COMSOL', plotting_config, show_xlabel=is_last, use_datetime=use_datetime)
            current_row += 1
        
        if plotting_config['plot_measured_comparison'] and measured_data is not None:
            is_last = (current_row == last_timeseries_row)
            self._plot_residuals(fig, gs[current_row, :], probe_data, measured_data,
                                'Measured', plotting_config, show_xlabel=is_last, use_datetime=use_datetime)
            current_row += 1
        
        # Plot snapshots if available and requested
        if snapshots:
            self._plot_snapshots(fig, gs, snapshots, plotting_config, start_row=current_row,
                               use_datetime=use_datetime)
        
        # Save
        if filename is None:
            filename = self.config.output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved: {filename}")
        plt.close()

    
    def _plot_time_series(self, fig, gs_slice, probe_data, rain_scenario, 
                         comsol_data, measured_data, plotting_config, show_xlabel=True, use_datetime=False):
        """Plot time series for specified field (water table or concentration)"""
        ax = fig.add_subplot(gs_slice)
        
        # Get field info from config
        field_name = plotting_config['field_name']
        field_units = plotting_config['field_units']
        field_label = plotting_config['field_label']
        
        # Prepare time data
        times_sec = probe_data['times']
        if use_datetime:
            times_plot = [self.config.time_converter.to_datetime(t) for t in times_sec]
        else:
            times_plot = times_sec / 3600.0  # Convert to hours
        
        # Generate enough colors for all probes
        num_probes = len(probe_data['data'])
        
        if num_probes <= 3:
            colors = ['#1f77b4', '#2ca02c', '#d62728'][:num_probes]
        else:
            # Use matplotlib's default color cycle for more colors
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] 
                     for i in range(num_probes)]
        
        # Plot simulated data (Firedrake)
        for i, (name, probe_fields) in enumerate(probe_data['data'].items()):
            # Extract the requested field data from the probe fields
            if field_name in probe_fields:
                field_data = probe_fields[field_name]
                # Extract clean probe name (e.g., "LTC 1 (x=8.0m, y=1.0m)" -> "LTC 1")
                clean_name = name.split('(')[0].strip()
                ax.plot(times_plot, field_data, color=colors[i], linewidth=2.5, 
                       label=f'{clean_name} - Firedrake simulation', marker='o', markersize=3, 
                       markevery=max(1, len(times_plot)//30))
        
        # Determine x-axis limits based on available data
        x_min = times_plot[0]
        x_max = times_plot[-1]
        has_reference_data = False
        
        # Plot COMSOL data if available
        if comsol_data:
            has_reference_data = True
            if use_datetime:
                # COMSOL data times are in days from simulation start
                times_comsol = [self.config.time_converter.to_datetime(t * 86400) for t in comsol_data['times']]
            else:
                times_comsol = comsol_data['times'] * 24  # days to hours
            
            # Update x-limits to where COMSOL data exists
            x_min = max(x_min, times_comsol[0])
            x_max = min(x_max, times_comsol[-1])
            
            for i, ltc_name in enumerate(['LTC 101', 'LTC 102', 'LTC 103']):
                if ltc_name in comsol_data:
                    # Map LTC 101/102/103 to LTC 1/2/3 for consistency
                    display_name = f"LTC {i+1}"
                    ax.plot(times_comsol, comsol_data[ltc_name],
                           color=colors[i], linewidth=2, linestyle='--',
                           label=f'{display_name} - COMSOL model', marker='s', markersize=4,
                           markevery=max(1, len(times_comsol)//20), alpha=0.8)
        
        # Plot measured data if available
        if measured_data:
            has_reference_data = True
            if use_datetime:
                times_measured = [self.config.time_converter.to_datetime(t * 86400) for t in measured_data['times']]
            else:
                times_measured = measured_data['times'] * 24
            
            # Update x-limits to where measured data exists
            x_min = max(x_min, times_measured[0])
            x_max = min(x_max, times_measured[-1])
            
            for i, ltc_name in enumerate(['LTC 101', 'LTC 102', 'LTC 103']):
                if ltc_name in measured_data:
                    # Map LTC 101/102/103 to LTC 1/2/3 for consistency
                    display_name = f"LTC {i+1}"
                    # Filter out NaN values for cleaner plotting
                    valid_mask = ~np.isnan(measured_data[ltc_name])
                    times_valid = [t for t, v in zip(times_measured, valid_mask) if v]
                    data_valid = measured_data[ltc_name][valid_mask]
                    
                    ax.plot(times_valid, data_valid,
                           color=colors[i], linewidth=1.5, linestyle=':',
                           label=f'{display_name} - Field measurements', marker='^', markersize=3,
                           markevery=max(1, len(times_valid)//15), alpha=0.7)
        
        # Rain event shading
        if rain_scenario:
            for idx, event in enumerate(rain_scenario.events):
                if use_datetime:
                    # event.start_datetime and end_datetime are datetime objects
                    start_dt = event.start_datetime
                    end_dt = event.end_datetime
                    # Only add label for first rain event to avoid duplicates
                    if start_dt and end_dt:  # Check they exist
                        ax.axvspan(start_dt, end_dt, alpha=0.15, color='lightblue', 
                                  label='Rain event' if idx == 0 else '')
                else:
                    # Convert simulation time (seconds) to time units
                    start_time = event.start / 3600.0  # seconds to hours
                    end_time = event.end / 3600.0      # seconds to hours
                    ax.axvspan(start_time, end_time, 
                              alpha=0.15, color='lightblue', 
                              label='Rain event' if idx == 0 else '')
        
        ax.set_ylabel(f'{field_label} ({field_units})', fontsize=12, fontweight='bold')
        if show_xlabel:
            if use_datetime:
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                # Format date axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)
        
        ax.grid(True, alpha=0.3)
        
        # Set x-limits to where reference data exists (if any), otherwise full simulation
        if has_reference_data:
            ax.set_xlim([x_min, x_max])
        else:
            if use_datetime:
                ax.set_xlim([times_plot[0], times_plot[-1]])
            else:
                ax.set_xlim([0, times_plot[-1]])
        
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10)
    
    def _plot_residuals(self, fig, gs_slice, probe_data, reference_data, 
                       reference_name, plotting_config, show_xlabel=True, use_datetime=False):
        """Plot residuals: Reference - Firedrake"""
        ax = fig.add_subplot(gs_slice)
        
        # Get field info from config
        field_name = plotting_config['field_name']
        
        # Prepare simulation time data
        times_sec_sim = probe_data['times']
        if use_datetime:
            times_sim = [self.config.time_converter.to_datetime(t) for t in times_sec_sim]
        else:
            times_sim = times_sec_sim / 3600.0
        
        # Prepare reference time data
        if use_datetime:
            times_ref = [self.config.time_converter.to_datetime(t * 86400) for t in reference_data['times']]
        else:
            times_ref = reference_data['times'] * 24
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        probe_names = list(probe_data['data'].keys())
        ltc_names = ['LTC 101', 'LTC 102', 'LTC 103']
        
        for i in range(min(len(probe_names), len(ltc_names))):
            ltc_name = ltc_names[i]
            probe_name = probe_names[i]
            
            if ltc_name not in reference_data:
                continue
            
            # Get data
            sim_data = np.array(probe_data['data'][probe_name][field_name])
            ref_values = reference_data[ltc_name]
            
            # Interpolate simulation to reference times (need numeric values)
            times_sec_sim_numeric = times_sec_sim
            times_ref_numeric = reference_data['times'] * 86400  # days to seconds
            
            interp_func = interp1d(times_sec_sim_numeric, sim_data, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            sim_interp = interp_func(times_ref_numeric)
            
            # Calculate residuals (Reference - Firedrake)
            residuals = ref_values - sim_interp
            
            # Plot using prepared time arrays
            ax.plot(times_ref, residuals, color=colors[i], linewidth=2.5,
                   marker='o', markersize=3, markevery=max(1, len(times_ref)//30))
            
            # Statistics
            print(f"  {reference_name} - {ltc_name} residuals: mean={residuals.mean():.4f}m, "
                  f"std={residuals.std():.4f}m, max_abs={np.abs(residuals).max():.4f}m")
        
        # Zero line
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        
        ax.set_ylabel(f'Residual (m)\n[{reference_name} - Firedrake]', 
                     fontsize=11, fontweight='bold')
        if show_xlabel:
            if use_datetime:
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)
        
        ax.grid(True, alpha=0.3)
        
        # Set x-limits to match reference data extent
        ax.set_xlim([times_ref[0], times_ref[-1]])
    
    def _plot_snapshots(self, fig, gs, snapshots, plotting_config, start_row, use_datetime=False):
        """Plot spatial snapshots (6 plots in 2x3 grid)"""
        sorted_times = sorted(snapshots.keys())[:6]
        while len(sorted_times) < 6:
            sorted_times.append(sorted_times[-1])
        
        # Get field info from config
        field_name = plotting_config['field_name']
        field_label = plotting_config['field_label']
        colormap = plotting_config['colormap']
        
        x_coords = self.coords[:, 0]
        y_coords = self.coords[:, 1]
        
        # Calculate global vmin/vmax across all snapshots for consistent colorbar
        all_data = []
        for t in sorted_times:
            if field_name == 'water_table':
                field_data = snapshots[t]['saturation'].dat.data[:]
                field_data = np.clip(field_data, 0, 1)
            else:
                field_data = snapshots[t][field_name].dat.data[:]
            all_data.extend(field_data.flatten())
        
        if field_name == 'water_table':
            vmin_global, vmax_global = 0, 1
        else:
            vmin_global, vmax_global = np.min(all_data), np.max(all_data)
            # Ensure we have a valid range
            if vmax_global == vmin_global:
                vmax_global = vmin_global + 1e-10
        
        for idx, t in enumerate(sorted_times):
            row = start_row + idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Get field data (could be saturation, concentration, etc.)
            if field_name == 'water_table':
                # Special case: need to compute saturation from pressure
                field_data = snapshots[t]['saturation'].dat.data[:]
                field_data = np.clip(field_data, 0, 1)
            else:
                # General case: use the field directly
                field_data = snapshots[t][field_name].dat.data[:]

            # Interpolate to regular grid
            if self.domain:
                xi = np.linspace(0, self.domain.Lx, 200)
                yi = np.linspace(0, self.domain.Ly, 100)
            else:
                # Fall back to mesh bounds
                x_coords = self.coords[:, 0]
                y_coords = self.coords[:, 1]
                xi = np.linspace(x_coords.min(), x_coords.max(), 200)
                yi = np.linspace(y_coords.min(), y_coords.max(), 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            interp = LinearNDInterpolator(np.column_stack((x_coords, y_coords)), field_data)
            Zi = interp(Xi, Yi)
            
            # Plot with appropriate levels and colormap
            if field_name == 'water_table':
                levels = np.linspace(0, 1, 25)
            else:
                # For concentration, use more levels for smoother appearance and global range
                levels = np.linspace(vmin_global, vmax_global, 50)
                
            cf = ax.contourf(Xi, Yi, Zi, levels=levels,
                            cmap=colormap, vmin=vmin_global, vmax=vmax_global)
            
            if idx == 0:
                contour_for_cbar = cf
            
            # Monitoring points (plot probe positions)
            if self.probe_manager and hasattr(self.probe_manager, 'probe_positions'):
                colors = ['#1f77b4', '#2ca02c', '#d62728']
                for i, (x, y) in enumerate(self.probe_manager.probe_positions):
                    color = colors[i % len(colors)]  # Cycle through colors if more than 3 probes
                    ax.plot(x, y, '*', color=color, 
                           markersize=12, markeredgecolor='black', markeredgewidth=0.8,
                           label=self.probe_manager.names[i] if i == 0 else "")  # Only label first for legend
            
            ax.set_xlabel('x (m)', fontsize=10)
            ax.set_ylabel('y (m)', fontsize=10)
            
            # Title with time or datetime
            if use_datetime:
                dt_title = self.config.time_converter.to_datetime(t)
                ax.set_title(f'{dt_title.strftime("%Y-%m-%d %H:%M")}', 
                           fontsize=11, fontweight='bold')
            else:
                ax.set_title(f't = {t/3600:.1f}h', fontsize=11, fontweight='bold')
            
            ax.set_aspect('equal')
            if self.domain:
                ax.set_xlim(0, self.domain.Lx)
                ax.set_ylim(0, self.domain.Ly)
            else:
                # Use mesh bounds
                x_coords = self.coords[:, 0]
                y_coords = self.coords[:, 1]
                ax.set_xlim(x_coords.min(), x_coords.max())
                ax.set_ylim(y_coords.min(), y_coords.max())

        # Colorbar with better formatting
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.56])
        cbar = fig.colorbar(contour_for_cbar, cax=cbar_ax)
        cbar.set_label(f'{field_label}', fontsize=12, fontweight='bold')
        
        # Improve colorbar formatting based on field type
        if field_name == 'concentration':
            # Format concentration values nicely
            data_max = max([snapshots[t][field_name].dat.data[:].max() for t in sorted_times])
            if data_max < 0.1:
                # Very small values - use scientific notation
                cbar.formatter.set_powerlimits((-3, -1))
                cbar.update_ticks()
            elif data_max < 10:
                # Small values - use 2 decimal places
                cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            else:
                # Larger values - use 1 decimal place
                cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        cbar.ax.tick_params(labelsize=10)