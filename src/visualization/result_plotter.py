"""
Results visualization - Refactored for flexibility

KEY IMPROVEMENTS:
1. Rain intensity bars on twin axis (instead of axvspan shading)
2. Separated PlotFieldConfig (field metadata) and PlotTimeSeriesConfig (plotting style)
3. Support for multiple fields on same time series plot
4. Snapshot field overlay capability
5. Data transformation pipeline
6. Clean, extensible architecture

QUICK START:
    plotter = ResultsPlotter(config, mesh, probe_manager, rain_scenario)
    plotter.plot_complete_results()  # Rain bars automatic!

CUSTOM FIELDS:
    field_configs = {
        'nitrate': PlotFieldConfig(
            field_name='nitrate',
            label='NO₃⁻',
            units='mg/L',
            colormap='YlOrRd',
            vmin=0, vmax=100
        )
    }
    plotter = ResultsPlotter(..., field_configs=field_configs)
    
MULTIPLE FIELDS:
    plotting_config = {
        'time_series_fields': ['water_table', 'concentration'],
        'snapshot_fields': ['saturation', 'concentration'],
        'snapshot_overlay': True
    }
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d
from datetime import datetime
from typing import Dict, List, Tuple
from .plot_configs import (PlotFieldConfig, PlotTimeSeriesConfig, 
                           get_default_field_configs, get_default_timeseries_configs)


class ResultsPlotter:
    """Creates complete visualization with time series and spatial snapshots"""
    
    def __init__(
        self, 
        config, 
        mesh, 
        probe_manager=None,
        rain_scenario=None, 
        domain=None,
        snapshot_manager=None,
        field_configs: Dict[str, PlotFieldConfig] = None,
        timeseries_configs: Dict[str, PlotTimeSeriesConfig] = None
    ):
        """
        Initialize plotter
        
        Args:
            config: SimulationConfig object
            mesh: Firedrake mesh
            probe_manager: ProbeManager instance
            rain_scenario: Rain/source scenario
            domain: Domain object for bounds
            snapshot_manager: SnapshotManager instance
            field_configs: Custom field configurations
            timeseries_configs: Custom time series style configurations
        """
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data
        self.probe_manager = probe_manager
        self.snapshot_manager = snapshot_manager
        self.rain_scenario = rain_scenario
        self.domain = domain
        
        # Initialize configurations
        self.field_configs = get_default_field_configs()
        if field_configs:
            self.field_configs.update(field_configs)
        
        self.timeseries_configs = get_default_timeseries_configs()
        if timeseries_configs:
            self.timeseries_configs.update(timeseries_configs)
    
    def get_field_config(self, field_name: str) -> PlotFieldConfig:
        """Get configuration for a field"""
        if field_name in self.field_configs:
            return self.field_configs[field_name]
        else:
            return PlotFieldConfig(field_name=field_name)
    
    def get_timeseries_config(self, style_name: str) -> PlotTimeSeriesConfig:
        """Get time series style configuration"""
        if style_name in self.timeseries_configs:
            return self.timeseries_configs[style_name]
        else:
            return PlotTimeSeriesConfig()
    
    def plot_complete_results(self, filename=None, plot_dates=True, plotting_config=None):
        """
        Create complete results figure
        
        Args:
            filename: Output filename
            plot_dates: Use datetime x-axis if possible
            plotting_config: Dict with:
                {
                    'time_series_fields': ['water_table'],
                    'plot_comsol_comparison': True,
                    'comsol_data_file': 'path/to/comsol.csv',
                    'comsol_ref_date': datetime(2024, 2, 22),
                    'plot_measured_comparison': True,
                    'measured_data_file': 'path/to/measured.csv',
                    'measured_ref_date': None,
                    'measured_offsets': {'LTC 101': 0.6},
                    'plot_snapshots': True,
                    'snapshot_fields': ['water_table'],
                    'snapshot_overlay': False
                }
        """
        default_config = {
            'time_series_fields': ['water_table'],
            'plot_comsol_comparison': False,
            'plot_measured_comparison': False,
            'plot_snapshots': True,
            'snapshot_fields': ['water_table'],
            'snapshot_overlay': False
        }
        
        if plotting_config is None:
            plotting_config = default_config
        else:
            for key, default_value in default_config.items():
                if key not in plotting_config:
                    plotting_config[key] = default_value
        
        probe_data = self.probe_manager.get_data() if self.probe_manager else None
        if probe_data is None:
            raise ValueError("probe_manager must be set and contain data")
        
        snapshots = None
        if plotting_config['plot_snapshots'] and self.snapshot_manager:
            snapshots = self.snapshot_manager.snapshots
        
        use_datetime = plot_dates and hasattr(self.config, 'time_converter') and self.config.time_converter is not None
        
        # Load comparison data
        comsol_data = self._load_comsol_data(plotting_config, probe_data) if plotting_config['plot_comsol_comparison'] else None
        measured_data = self._load_measured_data(plotting_config, probe_data) if plotting_config['plot_measured_comparison'] else None
        
        # Calculate layout
        n_rows = len(plotting_config['time_series_fields'])
        if plotting_config['plot_comsol_comparison'] and comsol_data is not None:
            n_rows += len(plotting_config['time_series_fields'])
        if plotting_config['plot_measured_comparison'] and measured_data is not None:
            n_rows += len(plotting_config['time_series_fields'])
        if snapshots:
            n_rows += 2
        
        fig_height = 4 * n_rows
        fig = plt.figure(figsize=(20, fig_height))
        gs = GridSpec(n_rows, 3, figure=fig, hspace=0.25, wspace=0.35)
        
        # Title
        title = 'Richards Equation Simulation Results'
        if use_datetime and self.config.start_datetime:
            title += f'\n{self.config.start_datetime.strftime("%Y-%m-%d")} to {self.config.time_converter.to_datetime(self.config.t_end).strftime("%Y-%m-%d")}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        current_row = 0
        last_timeseries_row = n_rows - (2 if snapshots else 0) - 1
        
        # Plot time series
        for field_name in plotting_config['time_series_fields']:
            is_last = (current_row == last_timeseries_row)
            self.plot_time_series(
                fig, gs[current_row, :], probe_data, 
                field_names=[field_name],
                comsol_data=comsol_data, 
                measured_data=measured_data,
                show_xlabel=is_last,
                use_datetime=use_datetime,
                plot_rain=True
            )
            current_row += 1
        
        # Plot residuals
        if plotting_config['plot_comsol_comparison'] and comsol_data is not None:
            for field_name in plotting_config['time_series_fields']:
                is_last = (current_row == last_timeseries_row)
                self._plot_residuals(
                    fig, gs[current_row, :], probe_data, comsol_data, 
                    'COMSOL', field_name, 
                    show_xlabel=is_last, use_datetime=use_datetime
                )
                current_row += 1
        
        if plotting_config['plot_measured_comparison'] and measured_data is not None:
            for field_name in plotting_config['time_series_fields']:
                is_last = (current_row == last_timeseries_row)
                self._plot_residuals(
                    fig, gs[current_row, :], probe_data, measured_data,
                    'Measured', field_name,
                    show_xlabel=is_last, use_datetime=use_datetime
                )
                current_row += 1
        
        # Plot snapshots
        if snapshots:
            self.plot_snapshots(
                fig, gs, snapshots, 
                field_names=plotting_config['snapshot_fields'],
                overlay=plotting_config['snapshot_overlay'],
                start_row=current_row,
                use_datetime=use_datetime
            )
        
        # Save
        if filename is None:
            filename = self.config.output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved: {filename}")
        plt.close()
    
    def _load_comsol_data(self, plotting_config, probe_data):
        """Load COMSOL comparison data"""
        try:
            from ..tools.import_results import load_comsol_data, DEFAULT_COMSOL_REF_DATE
            start_from = 0.0
            if hasattr(self.config, 'start_datetime') and self.config.start_datetime:
                comsol_ref_date = plotting_config.get('comsol_ref_date', DEFAULT_COMSOL_REF_DATE)
                start_from = (self.config.start_datetime - comsol_ref_date).total_seconds() / 86400.0
                print(f"ℹ️  Auto-calculated start_from: {start_from:.1f} days")
            
            return load_comsol_data(
                csv_path=plotting_config.get('comsol_data_file'),
                start_from_days=start_from,
                sim_duration_days=probe_data['times'][-1] / 3600.0 / 24.0
            )
        except Exception as e:
            print(f"⚠️  Could not load COMSOL data: {e}")
            return None
    
    def _load_measured_data(self, plotting_config, probe_data):
        """Load measured comparison data"""
        try:
            from ..tools.import_results import load_measured_data
            ref_date = plotting_config.get('measured_ref_date')
            if ref_date is None and hasattr(self.config, 'start_datetime'):
                ref_date = self.config.start_datetime
                print(f"ℹ️  Using config start_datetime: {ref_date.strftime('%Y-%m-%d')}")
            
            start_datetime = end_datetime = None
            if ref_date and hasattr(self.config, 'time_converter'):
                sim_duration_days = probe_data['times'][-1] / 3600.0 / 24.0
                start_datetime = ref_date
                end_datetime = ref_date + pd.Timedelta(days=sim_duration_days)
            
            return load_measured_data(
                csv_path=plotting_config.get('measured_data_file'),
                time_converter=self.config.time_converter,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                offsets=plotting_config.get('measured_offsets'),
                smooth_window=30,
                hampel_window=120,
            )
        except Exception as e:
            print(f"⚠️  Could not load measured data: {e}")
            return None

    def plot_time_series(
        self, 
        fig, 
        gs_slice, 
        probe_data: Dict,
        field_names: List[str] = None,
        comsol_data: Dict = None, 
        measured_data: Dict = None,
        show_xlabel: bool = True, 
        use_datetime: bool = False,
        plot_rain: bool = True,
        probe_colors: List[str] = None
    ):
        """
        Plot time series for one or more fields
        
        Args:
            fig: Matplotlib figure
            gs_slice: GridSpec slice
            probe_data: Probe data dictionary
            field_names: List of fields to plot
            comsol_data: COMSOL comparison data
            measured_data: Measured comparison data
            show_xlabel: Show x-axis label
            use_datetime: Use datetime axis
            plot_rain: Plot rain intensity on twin axis
            probe_colors: Custom colors for probes
        """
        if field_names is None:
            field_names = ['water_table']
        
        ax = fig.add_subplot(gs_slice)
        
        # Get field configs
        field_cfg = self.get_field_config(field_names[0])
        sim_cfg = self.get_timeseries_config('simulation')
        
        # Prepare time data
        times_sec = probe_data['times']
        if use_datetime:
            times_plot = [self.config.time_converter.to_datetime(t) for t in times_sec]
        else:
            times_plot = times_sec / 3600.0
        
        # Generate probe colors
        num_probes = len(probe_data['data'])
        if probe_colors is None:
            if num_probes <= 3:
                probe_colors = ['#1f77b4', '#2ca02c', '#d62728'][:num_probes]
            else:
                prop_cycle = plt.rcParams['axes.prop_cycle']
                probe_colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] 
                               for i in range(num_probes)]
        
        # Plot each field for each probe
        for field_idx, field_name in enumerate(field_names):
            field_cfg = self.get_field_config(field_name)
            
            # Adjust style if multiple fields
            if len(field_names) > 1:
                linestyle = ['-', '--', '-.', ':'][field_idx % 4]
                marker = ['o', 's', '^', 'D'][field_idx % 4]
            else:
                linestyle = sim_cfg.linestyle
                marker = sim_cfg.marker
            
            for probe_idx, (probe_name, probe_fields) in enumerate(probe_data['data'].items()):
                if field_cfg.data_key in probe_fields:
                    field_data = probe_fields[field_cfg.data_key]
                    
                    if field_cfg.transform_func:
                        field_data = field_cfg.transform_func(field_data)
                    
                    clean_name = probe_name.split('(')[0].strip()
                    label = f'{clean_name}'
                    if len(field_names) > 1:
                        label += f' - {field_cfg.label}'
                    label += ' - Firedrake'
                    
                    markevery = sim_cfg.markevery or max(1, len(times_plot)//30)
                    
                    ax.plot(
                        times_plot, field_data, 
                        color=probe_colors[probe_idx], 
                        linewidth=sim_cfg.linewidth,
                        linestyle=linestyle,
                        alpha=sim_cfg.alpha,
                        label=label, 
                        marker=marker, 
                        markersize=sim_cfg.markersize,
                        markevery=markevery
                    )
        
        # X-axis limits
        x_min = times_plot[0]
        x_max = times_plot[-1]
        has_reference_data = False
        
        # Plot COMSOL data
        if comsol_data:
            has_reference_data = True
            comsol_cfg = self.get_timeseries_config('comsol')
            
            if use_datetime:
                times_comsol = [self.config.time_converter.to_datetime(t * 86400) for t in comsol_data['times']]
            else:
                times_comsol = comsol_data['times'] * 24
            
            x_min = max(x_min, times_comsol[0])
            x_max = min(x_max, times_comsol[-1])
            
            for i, ltc_name in enumerate(['LTC 101', 'LTC 102', 'LTC 103']):
                if ltc_name in comsol_data:
                    display_name = f"LTC {i+1}"
                    markevery = comsol_cfg.markevery or max(1, len(times_comsol)//20)
                    ax.plot(
                        times_comsol, comsol_data[ltc_name],
                        color=probe_colors[i], 
                        linewidth=comsol_cfg.linewidth, 
                        linestyle=comsol_cfg.linestyle,
                        label=f'{display_name} - COMSOL', 
                        marker=comsol_cfg.marker, 
                        markersize=comsol_cfg.markersize,
                        markevery=markevery, 
                        alpha=comsol_cfg.alpha
                    )
        
        # Plot measured data
        if measured_data:
            has_reference_data = True
            meas_cfg = self.get_timeseries_config('measured')
            
            if use_datetime:
                times_measured = [self.config.time_converter.to_datetime(t * 86400) for t in measured_data['times']]
            else:
                times_measured = measured_data['times'] * 24
            
            x_min = max(x_min, times_measured[0])
            x_max = min(x_max, times_measured[-1])
            
            for i, ltc_name in enumerate(['LTC 101', 'LTC 102', 'LTC 103']):
                if ltc_name in measured_data:
                    display_name = f"LTC {i+1}"
                    valid_mask = ~np.isnan(measured_data[ltc_name])
                    times_valid = [t for t, v in zip(times_measured, valid_mask) if v]
                    data_valid = measured_data[ltc_name][valid_mask]
                    
                    markevery = meas_cfg.markevery or max(1, len(times_valid)//15)
                    ax.plot(
                        times_valid, data_valid,
                        color=probe_colors[i], 
                        linewidth=meas_cfg.linewidth, 
                        linestyle=meas_cfg.linestyle,
                        label=f'{display_name} - Measured', 
                        marker=meas_cfg.marker, 
                        markersize=meas_cfg.markersize,
                        markevery=markevery, 
                        alpha=meas_cfg.alpha
                    )
        
        # Plot rain intensity on twin axis
        if plot_rain and self.rain_scenario:
            ax_rain = ax.twinx()
            
            rain_times = []
            rain_intensities = []
            
            sorted_events = sorted(self.rain_scenario.events, key=lambda e: e.start)
            
            for i, event in enumerate(sorted_events):
                # Get time values in plot units
                if use_datetime:
                    t_start, t_end = event.start_datetime, event.end_datetime
                else:
                    t_start, t_end = event.start / 3600.0, event.end / 3600.0
                
                # Insert zero gap before event if needed
                if i > 0:
                    prev_end = sorted_events[i-1].end_datetime if use_datetime else sorted_events[i-1].end / 3600.0
                    if t_start > prev_end:
                        rain_times.extend([prev_end, t_start])
                        rain_intensities.extend([0, 0])
                
                # Add event
                rain_times.extend([t_start, t_end])
                rain_intensities.extend([event.rate, event.rate])
            
            # Add zero after last event
            if rain_times:
                rain_times.append(rain_times[-1])
                rain_intensities.append(0)
                
                ax_rain.fill_between(rain_times, 0, rain_intensities,
                                    step='post', alpha=0.3, color='skyblue', label='Rain Intensity')
                ax_rain.plot(rain_times, rain_intensities,
                            drawstyle='steps-post', color='steelblue', linewidth=2, alpha=0.7)
                
                ax_rain.set_ylabel('Rain Intensity (mm/hr)', fontsize=11, fontweight='bold', color='steelblue')
                ax_rain.tick_params(axis='y', labelcolor='steelblue')
                ax_rain.set_ylim(bottom=0)
                
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_rain.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2,
                        loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
        
        # Set labels
        ylabel = f'{field_cfg.label} ({field_cfg.units})'
        if len(field_names) > 1:
            ylabel = ' / '.join([f"{self.get_field_config(fn).label}" for fn in field_names])
        
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
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
        
        if has_reference_data:
            ax.set_xlim([x_min, x_max])
        else:
            if use_datetime:
                ax.set_xlim([times_plot[0], times_plot[-1]])
            else:
                ax.set_xlim([0, times_plot[-1]])
        
        if not (plot_rain and self.rain_scenario):
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    def _plot_residuals(
        self, 
        fig, 
        gs_slice, 
        probe_data: Dict, 
        reference_data: Dict, 
        reference_name: str,
        field_name: str,
        show_xlabel: bool = True, 
        use_datetime: bool = False
    ):
        """Plot residuals: Reference - Firedrake"""
        ax = fig.add_subplot(gs_slice)
        
        field_config = self.get_field_config(field_name)
        
        times_sec_sim = probe_data['times']
        if use_datetime:
            times_sim = [self.config.time_converter.to_datetime(t) for t in times_sec_sim]
        else:
            times_sim = times_sec_sim / 3600.0
        
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
            
            sim_data = np.array(probe_data['data'][probe_name][field_config.data_key])
            ref_values = reference_data[ltc_name]
            
            times_sec_sim_numeric = times_sec_sim
            times_ref_numeric = reference_data['times'] * 86400
            
            interp_func = interp1d(
                times_sec_sim_numeric, sim_data, kind='linear', 
                bounds_error=False, fill_value='extrapolate'
            )
            sim_interp = interp_func(times_ref_numeric)
            
            residuals = ref_values - sim_interp
            
            ax.plot(
                times_ref, residuals, color=colors[i], linewidth=2.5,
                marker='o', markersize=3, markevery=max(1, len(times_ref)//30)
            )
            
            print(f"  {reference_name} - {ltc_name} residuals: mean={residuals.mean():.4f}{field_config.units}, "
                  f"std={residuals.std():.4f}{field_config.units}, max_abs={np.abs(residuals).max():.4f}{field_config.units}")
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        
        ax.set_ylabel(
            f'Residual ({field_config.units})\n[{reference_name} - Firedrake]', 
            fontsize=11, fontweight='bold'
        )
        
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
        ax.set_xlim([times_ref[0], times_ref[-1]])
    
    def plot_snapshots(
        self, 
        fig, 
        gs, 
        snapshots: Dict,
        field_names: List[str] = None,
        overlay: bool = False,
        start_row: int = 0,
        use_datetime: bool = False,
        n_snapshots: int = 6
    ):
        """
        Plot spatial snapshots
        
        Args:
            fig: Matplotlib figure
            gs: GridSpec
            snapshots: Dict of snapshots {time: {field_name: Function}}
            field_names: List of fields to plot
            overlay: Overlay multiple fields on same plots
            start_row: Starting row in GridSpec
            use_datetime: Use datetime for titles
            n_snapshots: Number of snapshots to plot
        """
        if field_names is None:
            field_names = ['saturation']
        
        sorted_times = sorted(snapshots.keys())[:n_snapshots]
        while len(sorted_times) < n_snapshots:
            sorted_times.append(sorted_times[-1])
        
        x_coords = self.coords[:, 0]
        y_coords = self.coords[:, 1]
        
        if self.domain:
            xi = np.linspace(0, self.domain.Lx, 200)
            yi = np.linspace(0, self.domain.Ly, 100)
        else:
            xi = np.linspace(x_coords.min(), x_coords.max(), 200)
            yi = np.linspace(y_coords.min(), y_coords.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        if overlay:
            # Calculate ranges for each field
            field_ranges = {}
            for field_name in field_names:
                field_config = self.get_field_config(field_name)
                all_data = []
                
                for t in sorted_times:
                    field_data = self._extract_field_data(snapshots[t], field_name, field_config)
                    field_data = field_data[np.isfinite(field_data)]
                    if len(field_data) > 0:
                        p995 = np.percentile(field_data, 99.5)
                        p005 = np.percentile(field_data, 0.5)
                        field_data = field_data[(field_data >= p005) & (field_data <= p995)]
                        all_data.extend(field_data.flatten())
                
                vmin, vmax = self._calculate_colorbar_range(all_data, field_config)
                field_ranges[field_name] = (vmin, vmax)
            
            # Plot overlaid
            for idx, t in enumerate(sorted_times):
                row = start_row + idx // 3
                col = idx % 3
                ax = fig.add_subplot(gs[row, col])
                
                for field_idx, field_name in enumerate(field_names):
                    field_config = self.get_field_config(field_name)
                    vmin, vmax = field_ranges[field_name]
                    
                    field_data = self._extract_field_data(snapshots[t], field_name, field_config)
                    field_data = np.clip(field_data, vmin, vmax)
                    
                    interp = LinearNDInterpolator(
                        np.column_stack((x_coords, y_coords)), field_data
                    )
                    Zi = interp(Xi, Yi)
                    
                    if field_config.use_log_scale and vmin > 0:
                        levels = np.logspace(
                            np.log10(vmin), np.log10(vmax), field_config.contour_levels
                        )
                    else:
                        levels = np.linspace(vmin, vmax, field_config.contour_levels)
                    
                    alpha = 0.6 / len(field_names) if len(field_names) > 1 else 1.0
                    
                    if field_idx == 0:
                        cf = ax.contourf(
                            Xi, Yi, Zi, levels=levels,
                            cmap=field_config.colormap, 
                            vmin=vmin, vmax=vmax, 
                            alpha=alpha, extend='max'
                        )
                    else:
                        ax.contour(
                            Xi, Yi, Zi, levels=levels,
                            cmap=field_config.colormap, 
                            linewidths=1.5, alpha=0.8
                        )
                
                self._add_snapshot_decorations(ax, t, idx, use_datetime)
            
        else:
            # Separate plots for each field
            for field_idx, field_name in enumerate(field_names):
                field_config = self.get_field_config(field_name)
                
                all_data = []
                for t in sorted_times:
                    field_data = self._extract_field_data(snapshots[t], field_name, field_config)
                    field_data = field_data[np.isfinite(field_data)]
                    if len(field_data) > 0:
                        p995 = np.percentile(field_data, 99.5)
                        p005 = np.percentile(field_data, 0.5)
                        field_data = field_data[(field_data >= p005) & (field_data <= p995)]
                        all_data.extend(field_data.flatten())
                
                vmin, vmax = self._calculate_colorbar_range(all_data, field_config)
                
                for idx, t in enumerate(sorted_times):
                    row = start_row + (field_idx * 2) + (idx // 3)
                    col = idx % 3
                    ax = fig.add_subplot(gs[row, col])
                    
                    field_data = self._extract_field_data(snapshots[t], field_name, field_config)
                    field_data = np.clip(field_data, vmin, vmax)
                    
                    interp = LinearNDInterpolator(
                        np.column_stack((x_coords, y_coords)), field_data
                    )
                    Zi = interp(Xi, Yi)
                    
                    if field_config.use_log_scale and vmin > 0:
                        levels = np.logspace(
                            np.log10(vmin), np.log10(vmax), field_config.contour_levels
                        )
                    else:
                        levels = np.linspace(vmin, vmax, field_config.contour_levels)
                    
                    cf = ax.contourf(
                        Xi, Yi, Zi, levels=levels,
                        cmap=field_config.colormap, 
                        vmin=vmin, vmax=vmax, extend='max'
                    )
                    
                    if idx == 0:
                        contour_for_cbar = cf
                    
                    self._add_snapshot_decorations(ax, t, idx, use_datetime)
                
                # Colorbar
                cbar_ax = fig.add_axes([0.92, 0.11 + field_idx * 0.3, 0.02, 0.25])
                cbar = fig.colorbar(contour_for_cbar, cax=cbar_ax)
                cbar.set_label(
                    f'{field_config.label} ({field_config.units})', 
                    fontsize=12, fontweight='bold'
                )
                
                if vmax > 1000:
                    cbar.ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                elif vmax > 100:
                    cbar.ax.ticklabel_format(style='plain', axis='y')
                    ticks = cbar.get_ticks()
                    cbar.set_ticks(ticks[::2])
                
                cbar.ax.tick_params(labelsize=10)
    
    def _extract_field_data(
        self, 
        snapshot: Dict, 
        field_name: str, 
        field_config: PlotFieldConfig
    ) -> np.ndarray:
        """Extract field data from snapshot"""
        if field_name == 'saturation' or field_config.data_key == 'saturation':
            if 'saturation' in snapshot:
                field_data = snapshot['saturation'].dat.data[:]
            else:
                field_data = snapshot['pressure'].dat.data[:]
            field_data = np.clip(field_data, 0, 1)
        else:
            field_data = snapshot[field_config.data_key].dat.data[:]
        
        if field_config.transform_func:
            field_data = field_config.transform_func(field_data)
        
        return field_data
    
    def _calculate_colorbar_range(
        self, 
        all_data: List, 
        field_config: PlotFieldConfig
    ) -> Tuple[float, float]:
        """Calculate vmin/vmax for colorbar"""
        if field_config.vmin is not None and field_config.vmax is not None:
            return field_config.vmin, field_config.vmax
        
        if len(all_data) > 0:
            vmin = max(0.0, np.min(all_data)) if field_config.vmin is None else field_config.vmin
            vmax = np.max(all_data) if field_config.vmax is None else field_config.vmax
            
            if vmax == vmin:
                vmax = vmin + 1e-10
            elif vmax <= 0:
                vmin, vmax = 0, 1
            
            if vmax / max(vmin, 1e-10) > 1e6:
                vmax = vmin + 1000
            
            return vmin, vmax
        else:
            return 0, 1
    
    def _add_snapshot_decorations(
        self, 
        ax, 
        time: float, 
        idx: int, 
        use_datetime: bool
    ):
        """Add decorations to snapshot subplot"""
        if self.probe_manager and hasattr(self.probe_manager, 'probe_positions'):
            colors = ['#1f77b4', '#2ca02c', '#d62728']
            for i, (x, y) in enumerate(self.probe_manager.probe_positions):
                color = colors[i % len(colors)]
                ax.plot(
                    x, y, '*', color=color, 
                    markersize=12, markeredgecolor='black', markeredgewidth=0.8
                )
        
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)
        
        if use_datetime:
            dt_title = self.config.time_converter.to_datetime(time)
            ax.set_title(
                f'{dt_title.strftime("%Y-%m-%d %H:%M")}', 
                fontsize=11, fontweight='bold'
            )
        else:
            ax.set_title(f't = {time/3600:.1f}h', fontsize=11, fontweight='bold')
        
        ax.set_aspect('equal')
        
        if self.domain:
            ax.set_xlim(0, self.domain.Lx)
            ax.set_ylim(0, self.domain.Ly)
        else:
            x_coords = self.coords[:, 0]
            y_coords = self.coords[:, 1]
            ax.set_xlim(x_coords.min(), x_coords.max())
            ax.set_ylim(y_coords.min(), y_coords.max())