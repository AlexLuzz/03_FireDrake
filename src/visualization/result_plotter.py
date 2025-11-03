from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict
from .basic_plotting import BasicPlotting
from .plot_configs import DEFAULT_FIELDS, DEFAULT_STYLES, FieldConfig, TimeSeriesStyle


class ResultsPlotter(BasicPlotting):
    
    def __init__(self, config=None, mesh=None, probe_manager=None, rain_scenario=None,
                 domain=None, snapshot_manager=None, bc_manager=None,
                 field_configs: Dict[str, FieldConfig] = None,
                 timeseries_styles: Dict[str, TimeSeriesStyle] = None):
        
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data if mesh else None
        self.probe_manager = probe_manager
        self.snapshot_manager = snapshot_manager
        self.rain_scenario = rain_scenario
        self.bc_manager = bc_manager
        self.domain = domain
        
        self.field_configs = DEFAULT_FIELDS.copy()
        if field_configs:
            self.field_configs.update(field_configs)
        
        self.timeseries_styles = DEFAULT_STYLES.copy()
        if timeseries_styles:
            self.timeseries_styles.update(timeseries_styles)
    
    def _extract_probe_timeseries(self, probe_data: Dict, field_name: str) -> Dict:
        field_config = self.field_configs[field_name]
        result = {'times': probe_data['times']}
        
        # Get probe names in sorted order to ensure consistent numbering
        probe_names = sorted(probe_data['data'].keys())
        
        for i, probe_name in enumerate(probe_names):
            probe_fields = probe_data['data'][probe_name]
            if field_config.data_key in probe_fields:
                data = np.array(probe_fields[field_config.data_key])
                if field_config.transform_func:
                    data = field_config.transform_func(data)
                
                # Use consistent LTC naming to match COMSOL/Measured data
                standardized_name = f"LTC {i+1}"
                result[standardized_name] = data
        
        return result
    
    def _extract_snapshot_field(self, snapshot: Dict, field_name: str) -> np.ndarray:
        field_config = self.field_configs[field_name]
        
        if field_name == 'saturation' or field_config.data_key == 'saturation':
            data = snapshot.get('saturation', snapshot.get('pressure')).dat.data[:]
            data = np.clip(data, 0, 1)
        else:
            data = snapshot[field_config.data_key].dat.data[:]
        
        if field_config.transform_func:
            data = field_config.transform_func(data)
        
        return data
    
    def plot_complete_results(self, filename=None, plot_dates=True,
                             plotting_config=None, return_figure=False):
        config = {
            'time_series_fields': ['water_table'],
            'plot_comsol_comparison': False,
            'plot_measured_comparison': False,
            'plot_snapshots': True,
            'snapshot_fields': ['water_table'],
        }
        if plotting_config:
            config.update(plotting_config)
        
        probe_data = self.probe_manager.get_data()
        snapshots = self.snapshot_manager.snapshots if (config['plot_snapshots'] and self.snapshot_manager) else None
        use_datetime = plot_dates and hasattr(self.config, 'time_converter')
        
        comsol_data = self._load_comparison_data('comsol', config) if config['plot_comsol_comparison'] else None
        measured_data = self._load_comparison_data('measured', config) if config['plot_measured_comparison'] else None
        
        n_rows = len(config['time_series_fields'])
        if comsol_data:
            n_rows += len(config['time_series_fields'])
        if measured_data:
            n_rows += len(config['time_series_fields'])
        if snapshots:
            n_rows += 2
        
        fig = plt.figure(figsize=(20, 4 * n_rows))
        gs = GridSpec(n_rows, 3, figure=fig, hspace=0.3, wspace=0.35)
        
        title = 'Richards Equation Simulation Results'
        if use_datetime and self.config.start_datetime:
            title += f'\n{self.config.start_datetime.strftime("%Y-%m-%d")}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.975)
        
        # Collect all time axes for alignment
        time_axes = []
        
        # Determine x-axis limits for alignment
        times_sim = probe_data['times']
        if use_datetime:
            x_min = self.config.time_converter.to_datetime(times_sim[0])
            x_max = self.config.time_converter.to_datetime(times_sim[-1])
        else:
            x_min = times_sim[0] / 3600.0
            x_max = times_sim[-1] / 3600.0

        row = 0
        for field_name in config['time_series_fields']:
            ax = fig.add_subplot(gs[row, :])
            time_axes.append(ax)
            self._plot_timeseries_panel(ax, probe_data, field_name,
                                       comsol_data, measured_data, use_datetime)
            self.format_time_axis(ax, use_datetime, show_xlabel=(row == n_rows - (2 if snapshots else 0) - 1))
            row += 1

        for comparison_name, comparison_data in [('COMSOL', comsol_data), ('Measured', measured_data)]:
            if comparison_data:
                for field_name in config['time_series_fields']:
                    ax = fig.add_subplot(gs[row, :])
                    time_axes.append(ax)
                    self._plot_residuals_panel(ax, probe_data, comparison_data,
                                              field_name, comparison_name, use_datetime)
                    self.format_time_axis(ax, use_datetime, show_xlabel=(row == n_rows - (2 if snapshots else 0) - 1))
                    row += 1
        
        # Align all time-based x-axes
        for ax in time_axes:
            ax.set_xlim(x_min, x_max)
        
        if snapshots:
            self._plot_snapshots_grid(fig, gs, snapshots, config['snapshot_fields'],
                                     row, use_datetime)
        
        if return_figure:
            return fig
        
        if filename is None:
            filename = self.config.output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved: {filename}")
        plt.close()
    
    def _plot_timeseries_panel(self, ax, probe_data, field_name,
                               comsol_data, measured_data, use_datetime):
        field_config = self.field_configs[field_name]
        
        # Define consistent color mapping that can handle any number of probes
        base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        times_sim = probe_data['times']
        if use_datetime:
            times_plot = [self.config.time_converter.to_datetime(t) for t in times_sim]
        else:
            times_plot = times_sim / 3600.0

        sim_data = self._extract_probe_timeseries(probe_data, field_name)
        sim_data['times'] = times_plot
        
        # For Firedrake data, use consistent colors based on LTC number order
        # Create color mapping that matches LTC numbers exactly
        ltc_names = sorted([k for k in sim_data.keys() if k.startswith('LTC')])
        
        # Create a color map that assigns colors based on LTC number
        color_map = {}
        for i, ltc_name in enumerate(ltc_names):
            color_map[ltc_name] = base_colors[i % len(base_colors)]
        
        self.plot_timeseries(ax, times_plot, sim_data, field_config,
                            self.timeseries_styles['simulation'], color_map=color_map,
                            label_suffix=' - Firedrake')
        
        for comp_name, comp_data, comp_style in [
            ('COMSOL', comsol_data, 'comsol'),
            ('Measured', measured_data, 'measured')
        ]:
            if comp_data:
                times_comp = comp_data['times'] * 24
                if use_datetime:
                    times_comp = [self.config.time_converter.to_datetime(t * 86400) for t in comp_data['times']]
                
                comp_dict = {f'LTC {i+1}': comp_data[f'LTC {100+i+1}']
                            for i in range(3) if f'LTC {100+i+1}' in comp_data}
                
                # Use same color cycle for comparison data, matching probe indices
                n_comp = len(comp_dict)
                comp_colors = [base_colors[i % len(base_colors)] for i in range(n_comp)]
                self.plot_timeseries(ax, times_comp, comp_dict, field_config,
                                    self.timeseries_styles[comp_style], colors=comp_colors,
                                    label_suffix=f' - {comp_name}')

        if self.rain_scenario:
            self.add_rain_bars(ax, self.rain_scenario.events, use_datetime)
        
        if len(times_plot) > 0:
            ax.set_xlim(times_plot[0], times_plot[-1])

    def _plot_residuals_panel(self, ax, probe_data, ref_data, field_name, ref_name, use_datetime):
        field_config = self.field_configs[field_name]
        # Use same consistent color cycle as in time series
        base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        times_sim = probe_data['times']
        times_ref = ref_data['times'] * 86400
        
        if use_datetime:
            times_ref_plot = [self.config.time_converter.to_datetime(t) for t in times_ref]
        else:
            times_ref_plot = times_ref / 3600.0
        
        # Get probe names in sorted order to ensure consistent numbering
        probe_names = sorted(probe_data['data'].keys())
        
        for i, probe_name in enumerate(probe_names):
            ltc_name = f'LTC {100+i+1}'  # LTC 101, 102, 103...
            
            if ltc_name in ref_data:
                sim_data = np.array(probe_data['data'][probe_name][field_config.data_key])
                ref_vals = ref_data[ltc_name]
                
                _, residuals = self.compute_residuals(times_sim, sim_data, times_ref, ref_vals)
                
                # Use consistent colors and add probe labels to legend
                probe_label = f'LTC {i+1}'  # Standardized label
                color = base_colors[i % len(base_colors)]
                ax.plot(times_ref_plot, residuals, color=color, linewidth=2.5,
                       marker='o', markersize=3, markevery=max(1, len(times_ref)//30),
                       label=probe_label)
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        ax.set_ylabel(f'Residual ({field_config.units})\n[{ref_name} - Firedrake]',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)  # Add legend to show which line is which probe

    # -----------------------------
    # Simple generic timeseries plotter for report pages
    # -----------------------------
    def plot_simple_timeseries(self, ax, series_dict, label='Value', units=''):
        """
        Plot simple time series from a dict {display_name: values} on the given axis.
        Uses the report's standard styling and the time base from the simulation config.
        """
        if not self.probe_manager:
            raise ValueError('probe_manager is required to plot time series')

        data_all = self.probe_manager.get_data()
        times_sim = data_all['times']
        use_datetime = hasattr(self.config, 'time_converter') and self.config.time_converter is not None
        times_plot = [self.config.time_converter.to_datetime(t) for t in times_sim] if use_datetime else (times_sim / 3600.0)

        # Align times to series lengths (e.g., global series like mass loss often lack the initial t=0)
        series_lengths = [len(v) for v in series_dict.values() if hasattr(v, '__len__')]
        if series_lengths:
            n = min(len(times_plot), min(series_lengths))
            # Prefer aligning to the most recent n samples to match step-based series (skip initial t=0)
            if len(times_plot) != n:
                times_plot = times_plot[-n:]
            # Trim each series to the last n entries
            series_dict = {k: (np.asarray(v)[-n:] if hasattr(v, '__len__') else v)
                           for k, v in series_dict.items()}

        # Build data dict with 'times' + named series
        data_for_plot = {'times': times_plot}
        data_for_plot.update(series_dict)

        tmp_field = FieldConfig(field_name='custom', label=label, units=units)
        self.plot_timeseries(ax, times_plot, data_for_plot, tmp_field, self.timeseries_styles['simulation'])
        self.format_time_axis(ax, use_datetime, show_xlabel=True)
    
    def _plot_snapshots_grid(self, fig, gs, snapshots, field_names, start_row, use_datetime):
        sorted_times = sorted(snapshots.keys())[:6]
        x_coords, y_coords = self.coords[:, 0], self.coords[:, 1]
        
        for field_idx, field_name in enumerate(field_names):
            field_config = self.field_configs[field_name]
            
            all_data = []
            for t in sorted_times:
                data = self._extract_snapshot_field(snapshots[t], field_name)
                all_data.append(data)
            vmin, vmax = self.calculate_colorbar_range(np.concatenate(all_data), field_config)
            
            for idx, t in enumerate(sorted_times):
                row = start_row + (field_idx * 2) + (idx // 3)
                col = idx % 3
                ax = fig.add_subplot(gs[row, col])
                
                z_data = all_data[idx]
                cf = self.plot_snapshot(ax, x_coords, y_coords, z_data, field_config, vmin, vmax)
                
                if self.probe_manager and hasattr(self.probe_manager, 'probe_positions'):
                    self.add_probe_markers(ax, self.probe_manager.probe_positions)
                
                if use_datetime:
                    dt = self.config.time_converter.to_datetime(t)
                    ax.set_title(dt.strftime("%Y-%m-%d %H:%M"), fontsize=11, fontweight='bold')
                else:
                    ax.set_title(f't = {t/3600:.1f}h', fontsize=11, fontweight='bold')
                
                if self.domain:
                    ax.set_xlim(0, self.domain.Lx)
                    ax.set_ylim(0, self.domain.Ly)
            
            # Position colorbar lower and make it smaller
            # Adjust position based on the figure layout: [left, bottom, width, height]
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.25])
            cbar = fig.colorbar(cf, cax=cbar_ax)
            cbar.set_label(f'{field_config.label} ({field_config.units})',
                          fontsize=11, fontweight='bold')

            # For log-scale fields, prefer linearly spaced tick values over decades
            if getattr(field_config, 'use_log_scale', False) and cf.norm is not None:
                vmin = float(cf.norm.vmin)
                vmax = float(cf.norm.vmax)
                n_ticks = 6
                # Choose tick values spaced evenly in colorbar position (log-spaced in value)
                ticks = np.logspace(np.log10(max(vmin, np.finfo(float).tiny)), np.log10(vmax), n_ticks)
                cbar.set_ticks(ticks)
                # Nicely formatted labels (scientific if very small/large)
                def _fmt(v):
                    av = abs(v)
                    if (av > 0 and av < 1e-2) or av >= 1e4:
                        return f"{v:.1e}"
                    return f"{v:.3g}"
                cbar.set_ticklabels([_fmt(t) for t in ticks])
    
    def _load_comparison_data(self, data_type, config):
        try:
            from ..tools.import_results import load_comsol_data, load_measured_data, DEFAULT_COMSOL_REF_DATE
            
            if data_type == 'comsol':
                start_from = 0.0
                if hasattr(self.config, 'start_datetime') and self.config.start_datetime:
                    comsol_ref_date = config.get('comsol_ref_date', DEFAULT_COMSOL_REF_DATE)
                    start_from = (self.config.start_datetime - comsol_ref_date).total_seconds() / 86400.0
                
                return load_comsol_data(
                    csv_path=config.get('comsol_data_file'),
                    start_from_days=start_from,
                    sim_duration_days=self.probe_manager.get_data()['times'][-1] / 86400
                )
            elif data_type == 'measured':
                ref_date = config.get('measured_ref_date')
                if ref_date is None and hasattr(self.config, 'start_datetime'):
                    ref_date = self.config.start_datetime
                
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
        except Exception as e:
            print(f"⚠️  Could not load {data_type} data: {e}")
            return None