"""
Results visualization - Refactored for simplicity and versatility

ARCHITECTURE:
1. Data extraction functions (from managers or any source)
2. Generic plotting functions (take raw data, return styled plots)
3. High-level orchestrator (plot_complete_results)

QUICK START:
    plotter = ResultsPlotter(config, mesh, probe_manager, rain_scenario)
    plotter.plot_complete_results()

CUSTOM USAGE:
    # Extract your data however you want
    times, probe_data = extract_timeseries(...)
    
    # Plot it
    fig, ax = plt.subplots()
    plot_timeseries(ax, times, probe_data, field_config)
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from .plot_configs import (PlotFieldConfig, PlotTimeSeriesConfig, 
                           get_default_field_configs, get_default_timeseries_configs)


# ============================================================================
# DATA EXTRACTION FUNCTIONS
# ============================================================================

def extract_probe_timeseries(probe_data: Dict, field_name: str, field_config: PlotFieldConfig) -> Dict:
    """
    Extract time series data from probe manager format
    
    Returns:
        {'times': array, 'probe1': array, 'probe2': array, ...}
    """
    result = {'times': probe_data['times']}
    
    for probe_name, probe_fields in probe_data['data'].items():
        if field_config.data_key in probe_fields:
            data = np.array(probe_fields[field_config.data_key])
            if field_config.transform_func:
                data = field_config.transform_func(data)
            
            clean_name = probe_name.split('(')[0].strip()
            result[clean_name] = data
    
    return result


def extract_snapshot_field(snapshot: Dict, field_name: str, field_config: PlotFieldConfig) -> np.ndarray:
    """Extract and transform field data from a single snapshot"""
    # Handle special cases
    if field_name == 'saturation' or field_config.data_key == 'saturation':
        data = snapshot.get('saturation', snapshot.get('pressure')).dat.data[:]
        data = np.clip(data, 0, 1)
    else:
        data = snapshot[field_config.data_key].dat.data[:]
    
    if field_config.transform_func:
        data = field_config.transform_func(data)
    
    return data


def compute_residuals(times1: np.ndarray, data1: np.ndarray, 
                     times2: np.ndarray, data2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute residuals: data2 - data1 (interpolated to times2)
    
    Returns:
        (times2, residuals)
    """
    interp_func = interp1d(times1, data1, kind='linear', bounds_error=False, fill_value='extrapolate')
    data1_interp = interp_func(times2)
    return times2, data2 - data1_interp


def calculate_colorbar_range(data: np.ndarray, field_config: PlotFieldConfig) -> Tuple[float, float]:
    """Calculate smart vmin/vmax, respecting config overrides"""
    if field_config.vmin is not None and field_config.vmax is not None:
        return field_config.vmin, field_config.vmax
    
    # Filter outliers
    data_clean = data[np.isfinite(data)]
    if len(data_clean) > 0:
        p995, p005 = np.percentile(data_clean, [99.5, 0.5])
        data_clean = data_clean[(data_clean >= p005) & (data_clean <= p995)]
    
    if len(data_clean) == 0:
        return 0, 1
    
    vmin = max(0.0, np.min(data_clean)) if field_config.vmin is None else field_config.vmin
    vmax = np.max(data_clean) if field_config.vmax is None else field_config.vmax
    
    # Safety checks
    if vmax == vmin:
        vmax = vmin + 1e-10
    if vmax / max(vmin, 1e-10) > 1e6:
        vmax = vmin + 1000
    
    return vmin, vmax


# ============================================================================
# GENERIC PLOTTING FUNCTIONS
# ============================================================================

def plot_timeseries(ax, times: np.ndarray, data_dict: Dict[str, np.ndarray],
                   field_config: PlotFieldConfig,
                   style_config: PlotTimeSeriesConfig = None,
                   colors: List[str] = None,
                   label_suffix: str = ''):
    """
    Generic time series plotter
    
    Args:
        ax: Matplotlib axis
        times: Time array
        data_dict: {'probe1': array, 'probe2': array, ...}
        field_config: Field styling
        style_config: Line styling
        colors: List of colors for each series
        label_suffix: Add to legend labels (e.g., ' - Firedrake')
    """
    if style_config is None:
        style_config = PlotTimeSeriesConfig()
    
    if colors is None:
        # Build a color list based on matplotlib default color cycle to support >3 series
        base_colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['#1f77b4', '#2ca02c', '#d62728'])
        n_series = len([k for k in data_dict.keys() if k != 'times'])
        colors = [base_colors[i % len(base_colors)] for i in range(n_series)]
    
    markevery = style_config.markevery or max(1, len(times)//30)
    
    for i, (name, data) in enumerate(data_dict.items()):
        if name == 'times':
            continue
        
        label = f'{name}{label_suffix}' if label_suffix else name
        ax.plot(times, data,
                color=colors[i % len(colors)],
                linewidth=style_config.linewidth,
                linestyle=style_config.linestyle,
                marker=style_config.marker,
                markersize=style_config.markersize,
                markevery=markevery,
                alpha=style_config.alpha,
                label=label)
    
    ax.set_ylabel(f'{field_config.label} ({field_config.units})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)


def add_rain_bars(ax, rain_events: List, use_datetime: bool = False):
    """Add rain intensity as filled bars on twin axis"""
    ax_rain = ax.twinx()
    
    times, intensities = [], []
    sorted_events = sorted(rain_events, key=lambda e: e.start)
    
    for i, event in enumerate(sorted_events):
        t_start = event.start_datetime if use_datetime else event.start / 3600.0
        t_end = event.end_datetime if use_datetime else event.end / 3600.0
        
        # Gap before event
        if i > 0:
            prev_end = sorted_events[i-1].end_datetime if use_datetime else sorted_events[i-1].end / 3600.0
            if t_start > prev_end:
                times.extend([prev_end, t_start])
                intensities.extend([0, 0])
        
        times.extend([t_start, t_end])
        intensities.extend([event.rate, event.rate])
    
    if times:
        times.append(times[-1])
        intensities.append(0)
        
        ax_rain.fill_between(times, 0, intensities, step='post', alpha=0.3, 
                             color='skyblue', label='Rain Intensity')
        ax_rain.plot(times, intensities, drawstyle='steps-post', 
                    color='steelblue', linewidth=2, alpha=0.7)
        
        ax_rain.set_ylabel('Rain (mm/hr)', fontsize=11, fontweight='bold', color='steelblue')
        ax_rain.tick_params(axis='y', labelcolor='steelblue')
        ax_rain.set_ylim(bottom=0)
        
        # Merge legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_rain.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, 
                 loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    return ax_rain


def plot_snapshot(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 field_config: PlotFieldConfig,
                 vmin: float = None, vmax: float = None,
                 add_colorbar: bool = False):
    """
    Generic 2D spatial field plotter
    
    Args:
        ax: Matplotlib axis
        x, y: Coordinate arrays (1D)
        z: Field values (1D, same length as x,y)
        field_config: Field styling
        vmin, vmax: Color limits (auto-calculated if None)
        add_colorbar: Add colorbar to axis
    
    Returns:
        contourf object (for colorbar)
    """
    # Interpolate to grid
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    interp = LinearNDInterpolator(np.column_stack((x, y)), z)
    Zi = interp(Xi, Yi)
    
    # Auto range if needed
    if vmin is None or vmax is None:
        vmin, vmax = calculate_colorbar_range(z, field_config)
    
    # Clip data
    Zi = np.clip(Zi, vmin, vmax)
    
    # Levels
    if field_config.use_log_scale and vmin > 0:
        levels = np.logspace(np.log10(vmin), np.log10(vmax), field_config.contour_levels)
    else:
        levels = np.linspace(vmin, vmax, field_config.contour_levels)
    
    cf = ax.contourf(Xi, Yi, Zi, levels=levels,
                     cmap=field_config.colormap,
                     vmin=vmin, vmax=vmax, extend='max')
    
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)
    ax.set_aspect('equal')
    
    if add_colorbar:
        plt.colorbar(cf, ax=ax, label=f'{field_config.label} ({field_config.units})')
    
    return cf


def format_time_axis(ax, use_datetime: bool, show_xlabel: bool = True):
    """Apply time axis formatting"""
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


def add_probe_markers(ax, probe_positions: List[Tuple], colors: List[str] = None):
    """Add probe location markers to spatial plot"""
    if colors is None:
        colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for i, (x, y) in enumerate(probe_positions):
        ax.plot(x, y, '*', color=colors[i % len(colors)],
                markersize=12, markeredgecolor='black', markeredgewidth=0.8)


# ============================================================================
# HIGH-LEVEL PLOTTER CLASS
# ============================================================================

class ResultsPlotter:
    """Orchestrates complete visualization from managers"""
    
    def __init__(self, config, mesh, probe_manager=None, rain_scenario=None,
                 domain=None, snapshot_manager=None, bc_manager=None,
                 field_configs: Dict[str, PlotFieldConfig] = None,
                 timeseries_configs: Dict[str, PlotTimeSeriesConfig] = None):
        
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data
        self.probe_manager = probe_manager
        self.snapshot_manager = snapshot_manager
        self.rain_scenario = rain_scenario
        # Optional boundary condition manager (to retrieve water table boundaries)
        self.bc_manager = bc_manager
        self.domain = domain
        
        self.field_configs = get_default_field_configs()
        if field_configs:
            self.field_configs.update(field_configs)
        
        self.timeseries_configs = get_default_timeseries_configs()
        if timeseries_configs:
            self.timeseries_configs.update(timeseries_configs)
    
    def plot_complete_results(self, filename=None, plot_dates=True, 
                             plotting_config=None, return_figure=False):
        """Create complete results figure (main entry point)"""
        
        # Default config
        config = {
            'time_series_fields': ['water_table'],
            'plot_comsol_comparison': False,
            'plot_measured_comparison': False,
            'plot_snapshots': True,
            'snapshot_fields': ['water_table'],
        }
        if plotting_config:
            config.update(plotting_config)
        
        # Setup
        probe_data = self.probe_manager.get_data()
        snapshots = self.snapshot_manager.snapshots if (config['plot_snapshots'] and self.snapshot_manager) else None
        use_datetime = plot_dates and hasattr(self.config, 'time_converter')
        
        # Load comparison data
        comsol_data = self._load_comparison_data('comsol', config) if config['plot_comsol_comparison'] else None
        measured_data = self._load_comparison_data('measured', config) if config['plot_measured_comparison'] else None
        
        # Calculate layout
        n_rows = len(config['time_series_fields'])
        if comsol_data:
            n_rows += len(config['time_series_fields'])
        if measured_data:
            n_rows += len(config['time_series_fields'])
        if snapshots:
            n_rows += 2
        
        # Create figure
        fig = plt.figure(figsize=(20, 4 * n_rows))
        gs = GridSpec(n_rows, 3, figure=fig, hspace=0.25, wspace=0.35)
        
        # Title
        title = 'Richards Equation Simulation Results'
        if use_datetime and self.config.start_datetime:
            title += f'\n{self.config.start_datetime.strftime("%Y-%m-%d")}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        # Plot time series
        row = 0
        for field_name in config['time_series_fields']:
            ax = fig.add_subplot(gs[row, :])
            self._plot_timeseries_panel(ax, probe_data, field_name, 
                                       comsol_data, measured_data, use_datetime)
            format_time_axis(ax, use_datetime, show_xlabel=(row == n_rows - (2 if snapshots else 0) - 1))
            row += 1
        
        # Plot residuals
        for comparison_name, comparison_data in [('COMSOL', comsol_data), ('Measured', measured_data)]:
            if comparison_data:
                for field_name in config['time_series_fields']:
                    ax = fig.add_subplot(gs[row, :])
                    self._plot_residuals_panel(ax, probe_data, comparison_data, 
                                              field_name, comparison_name, use_datetime)
                    format_time_axis(ax, use_datetime, show_xlabel=(row == n_rows - (2 if snapshots else 0) - 1))
                    row += 1
        
        # Plot snapshots
        if snapshots:
            self._plot_snapshots_grid(fig, gs, snapshots, config['snapshot_fields'], 
                                     row, use_datetime)
        
        # Save or return
        if return_figure:
            return fig
        
        if filename is None:
            filename = self.config.output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved: {filename}")
        plt.close()
    
    def _plot_timeseries_panel(self, ax, probe_data, field_name, 
                               comsol_data, measured_data, use_datetime):
        """Plot time series with comparisons"""
        field_config = self.field_configs[field_name]
        
        # Convert times
        times_sim = probe_data['times']
        if use_datetime:
            times_plot = [self.config.time_converter.to_datetime(t) for t in times_sim]
        else:
            times_plot = times_sim / 3600.0
        
        # Extract and plot simulation
        sim_data = extract_probe_timeseries(probe_data, field_name, field_config)
        sim_data['times'] = times_plot
        plot_timeseries(ax, times_plot, sim_data, field_config,
                       self.timeseries_configs['simulation'], label_suffix=' - Firedrake')
        
        # Plot comparisons
        # Build a color list for comparison series
        base_colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['#1f77b4', '#2ca02c', '#d62728'])
        
        for comp_name, comp_data, comp_style in [
            ('COMSOL', comsol_data, 'comsol'),
            ('Measured', measured_data, 'measured')
        ]:
            if comp_data:
                times_comp = comp_data['times'] * 24  # days to hours
                if use_datetime:
                    times_comp = [self.config.time_converter.to_datetime(t * 86400) for t in comp_data['times']]
                
                comp_dict = {f'LTC {i+1}': comp_data[f'LTC {100+i+1}'] 
                            for i in range(3) if f'LTC {100+i+1}' in comp_data}
                
                # determine colors for this comparison set
                n_comp = len(comp_dict)
                comp_colors = [base_colors[i % len(base_colors)] for i in range(n_comp)]
                plot_timeseries(ax, times_comp, comp_dict, field_config,
                              self.timeseries_configs[comp_style], colors=comp_colors,
                              label_suffix=f' - {comp_name}')
        
        # Add rain
        if self.rain_scenario:
            add_rain_bars(ax, self.rain_scenario.events, use_datetime)

        # Plot boundary-condition water table for each probe (shows slope)
        try:
            if self.bc_manager and self.domain and self.probe_manager:
                # Get simulation time range for BC plotting
                t_start = times_sim[0] 
                t_end = times_sim[-1]
                t_mid = (t_start + t_end) / 2.0
                bc_times = [t_start, t_mid, t_end]
                bc_labels = ['BC t=0', 'BC t=mid', 'BC t=end']
                bc_alphas = [0.9, 0.7, 0.5]
                
                # Determine colors used for the simulation series
                sim_keys = [k for k in sim_data.keys() if k != 'times']
                n_sim = len(sim_keys)
                sim_colors = [base_colors[i % len(base_colors)] for i in range(n_sim)]

                for bc_idx, (bc_time, bc_label, bc_alpha) in enumerate(zip(bc_times, bc_labels, bc_alphas)):
                    left_wt, right_wt = self.bc_manager.get_water_table(bc_time)
                    
                    for idx, pname in enumerate(sim_keys):
                        # probe positions assumed aligned with sim_data order
                        if hasattr(self.probe_manager, 'probe_positions') and len(self.probe_manager.probe_positions) > idx:
                            x_pos = self.probe_manager.probe_positions[idx][0]
                            wt_probe = left_wt + (right_wt - left_wt) * (x_pos / float(self.domain.Lx))
                            
                            # Convert time to plot coordinates
                            if use_datetime:
                                bc_time_plot = self.config.time_converter.to_datetime(bc_time)
                            else:
                                bc_time_plot = bc_time / 3600.0
                            
                            # Plot vertical line at this time showing BC water table
                            ax.axvline(bc_time_plot, color=sim_colors[idx], linestyle=':', linewidth=1.5,
                                      alpha=bc_alpha, label=(bc_label if idx == 0 else None))
                            
                            # Plot horizontal line at this elevation for short duration around this time
                            time_window = (times_plot[-1] - times_plot[0]) * 0.02  # 2% of total time
                            if use_datetime:
                                import pandas as pd
                                t_start_win = bc_time_plot - pd.Timedelta(seconds=time_window*3600 if not use_datetime else time_window)
                                t_end_win = bc_time_plot + pd.Timedelta(seconds=time_window*3600 if not use_datetime else time_window)
                            else:
                                t_start_win = bc_time_plot - time_window
                                t_end_win = bc_time_plot + time_window
                            
                            ax.hlines(wt_probe, t_start_win, t_end_win, colors=sim_colors[idx], 
                                     linestyles='-', linewidth=2.5, alpha=bc_alpha)
        except Exception as e:
            # Non-fatal: if BC plotting fails, continue without BC lines
            print(f"Warning: BC plotting failed: {e}")
            pass

        # Set tight x-axis limits to data range
        if len(times_plot) > 0:
            ax.set_xlim(times_plot[0], times_plot[-1])
    
    def _plot_residuals_panel(self, ax, probe_data, ref_data, field_name, ref_name, use_datetime):
        """Plot residuals panel"""
        field_config = self.field_configs[field_name]
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        times_sim = probe_data['times']
        times_ref = ref_data['times'] * 86400  # days to seconds
        
        if use_datetime:
            times_ref_plot = [self.config.time_converter.to_datetime(t) for t in times_ref]
        else:
            times_ref_plot = times_ref / 3600.0
        
        for i, (ltc_name, probe_name) in enumerate(zip(['LTC 101', 'LTC 102', 'LTC 103'], 
                                                        list(probe_data['data'].keys())[:3])):
            if ltc_name in ref_data:
                sim_data = np.array(probe_data['data'][probe_name][field_config.data_key])
                ref_vals = ref_data[ltc_name]
                
                _, residuals = compute_residuals(times_sim, sim_data, times_ref, ref_vals)
                
                ax.plot(times_ref_plot, residuals, color=colors[i], linewidth=2.5,
                       marker='o', markersize=3, markevery=max(1, len(times_ref)//30))
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        ax.set_ylabel(f'Residual ({field_config.units})\n[{ref_name} - Firedrake]',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_snapshots_grid(self, fig, gs, snapshots, field_names, start_row, use_datetime):
        """Plot spatial snapshots grid"""
        sorted_times = sorted(snapshots.keys())[:6]
        x_coords, y_coords = self.coords[:, 0], self.coords[:, 1]
        
        for field_idx, field_name in enumerate(field_names):
            field_config = self.field_configs[field_name]
            
            # Calculate global color range
            all_data = []
            for t in sorted_times:
                data = extract_snapshot_field(snapshots[t], field_name, field_config)
                all_data.append(data)
            vmin, vmax = calculate_colorbar_range(np.concatenate(all_data), field_config)
            
            # Plot each snapshot
            for idx, t in enumerate(sorted_times):
                row = start_row + (field_idx * 2) + (idx // 3)
                col = idx % 3
                ax = fig.add_subplot(gs[row, col])
                
                z_data = all_data[idx]
                cf = plot_snapshot(ax, x_coords, y_coords, z_data, field_config, vmin, vmax)
                
                # Decorations
                if self.probe_manager and hasattr(self.probe_manager, 'probe_positions'):
                    add_probe_markers(ax, self.probe_manager.probe_positions)
                
                if use_datetime:
                    dt = self.config.time_converter.to_datetime(t)
                    ax.set_title(dt.strftime("%Y-%m-%d %H:%M"), fontsize=11, fontweight='bold')
                else:
                    ax.set_title(f't = {t/3600:.1f}h', fontsize=11, fontweight='bold')
                
                if self.domain:
                    ax.set_xlim(0, self.domain.Lx)
                    ax.set_ylim(0, self.domain.Ly)
            
            # Colorbar
            cbar_ax = fig.add_axes([0.92, 0.11 + field_idx * 0.3, 0.02, 0.25])
            cbar = fig.colorbar(cf, cax=cbar_ax)
            cbar.set_label(f'{field_config.label} ({field_config.units})', 
                          fontsize=12, fontweight='bold')
    
    def _load_comparison_data(self, data_type, config):
        """Load COMSOL or measured data"""
        try:
            from ..tools.import_results import load_comsol_data, load_measured_data, DEFAULT_COMSOL_REF_DATE
            
            if data_type == 'comsol':
                # Calculate proper start_from_days based on simulation start time
                start_from = 0.0
                if hasattr(self.config, 'start_datetime') and self.config.start_datetime:
                    comsol_ref_date = config.get('comsol_ref_date', DEFAULT_COMSOL_REF_DATE)
                    start_from = (self.config.start_datetime - comsol_ref_date).total_seconds() / 86400.0
                    print(f"ℹ️  Auto-calculated start_from: {start_from:.1f} days")
                
                return load_comsol_data(
                    csv_path=config.get('comsol_data_file'),
                    start_from_days=start_from,
                    sim_duration_days=self.probe_manager.get_data()['times'][-1] / 86400
                )
            elif data_type == 'measured': 
                # Calculate proper date range for measured data
                ref_date = config.get('measured_ref_date')
                if ref_date is None and hasattr(self.config, 'start_datetime'):
                    ref_date = self.config.start_datetime
                    print(f"ℹ️  Using config start_datetime: {ref_date.strftime('%Y-%m-%d')}")
                
                start_datetime = end_datetime = None
                if ref_date and hasattr(self.config, 'time_converter'):
                    sim_duration_days = self.probe_manager.get_data()['times'][-1] / 86400
                    start_datetime = ref_date
                    end_datetime = ref_date + pd.Timedelta(days=sim_duration_days)
                
                return load_measured_data(
                    csv_path=config.get('measured_data_file'),
                    time_converter=self.config.time_converter,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    offsets=config.get('measured_offsets'),
                    smooth_window=30,
                    hampel_window=120,
                )
            else:
                return None
        except Exception as e:
            print(f"⚠️  Could not load {data_type} data: {e}")
            return None