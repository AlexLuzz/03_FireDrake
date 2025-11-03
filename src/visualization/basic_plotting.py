import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from typing import Dict, List, Tuple, Optional
from .plot_configs import FieldConfig, TimeSeriesStyle
from matplotlib.colors import LogNorm


class BasicPlotting:
    
    @staticmethod
    def plot_timeseries(ax, times: np.ndarray, data_dict: Dict[str, np.ndarray],
                       field_config: FieldConfig,
                       style: TimeSeriesStyle = None,
                       colors: List[str] = None,
                       color_map: Dict[str, str] = None,
                       label_suffix: str = ''):
        if style is None:
            style = TimeSeriesStyle()
        
        if colors is None and color_map is None:
            base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            n_series = len([k for k in data_dict.keys() if k != 'times'])
            colors = [base_colors[i % len(base_colors)] for i in range(n_series)]
        
        markevery = style.markevery or max(1, len(times)//30)
        
        for i, (name, data) in enumerate(data_dict.items()):
            if name == 'times':
                continue
            
            # Use color_map if provided, otherwise use colors list
            if color_map is not None:
                color = color_map.get(name, plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])])
            else:
                color = colors[i % len(colors)]
            
            label = f'{name}{label_suffix}' if label_suffix else name
            ax.plot(times, data,
                    color=color,
                    linewidth=style.linewidth,
                    linestyle=style.linestyle,
                    marker=style.marker,
                    markersize=style.markersize,
                    markevery=markevery,
                    alpha=style.alpha,
                    label=label)
        
        ax.set_ylabel(f'{field_config.label} ({field_config.units})', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    @staticmethod
    def plot_snapshot(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     field_config: FieldConfig,
                     vmin: float = None, vmax: float = None,
                     add_colorbar: bool = False):
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        interp = LinearNDInterpolator(np.column_stack((x, y)), z)
        Zi = interp(Xi, Yi)
        
        if vmin is None or vmax is None:
            vmin, vmax = BasicPlotting.calculate_colorbar_range(z, field_config)

        # Replace masked/NaNs from interpolation (outside convex hull) to avoid white gaps
        # Use a safe fill value depending on scale
        if np.ma.isMaskedArray(Zi):
            Zi = Zi.filled(np.nan)
        if not np.isfinite(Zi).all():
            # For log scale we only enable LogNorm when vmin>0
            fill_val = vmin if not (field_config.use_log_scale and vmin > 0) else max(vmin, np.finfo(float).tiny)
            Zi = np.where(np.isfinite(Zi), Zi, fill_val)

        # For log scale, ensure zeros and tiny values are lifted to vmin to avoid transparent/white
        if field_config.use_log_scale and vmin > 0:
            Zi = np.maximum(Zi, max(vmin, np.finfo(float).tiny))
        
        Zi = np.clip(Zi, vmin, vmax)
        
        if field_config.use_log_scale and vmin > 0:
            levels = np.logspace(np.log10(vmin), np.log10(vmax), field_config.contour_levels)
            norm = LogNorm(vmin=max(vmin, np.finfo(float).tiny), vmax=vmax)
        else:
            levels = np.linspace(vmin, vmax, field_config.contour_levels)
            norm = None
        
        cf = ax.contourf(
            Xi, Yi, Zi,
            levels=levels,
            cmap=field_config.colormap,
            norm=norm if norm is not None else None,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            extend='both'
        )
        
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)
        ax.set_aspect('equal')
        
        if add_colorbar:
            plt.colorbar(cf, ax=ax, label=f'{field_config.label} ({field_config.units})')
        
        return cf
    
    @staticmethod
    def add_rain_bars(ax, rain_events: List, use_datetime: bool = False):
        ax_rain = ax.twinx()
        
        times, intensities = [], []
        sorted_events = sorted(rain_events, key=lambda e: e.start)
        
        for i, event in enumerate(sorted_events):
            t_start = event.start_datetime if use_datetime else event.start / 3600.0
            t_end = event.end_datetime if use_datetime else event.end / 3600.0
            
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
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_rain.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                     loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
        
        return ax_rain
    
    @staticmethod
    def format_time_axis(ax, use_datetime: bool, show_xlabel: bool = True):
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
    
    @staticmethod
    def add_probe_markers(ax, probe_positions: List[Tuple], colors: List[str] = None):
        if colors is None:
            base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors = [base_colors[i % len(base_colors)] for i in range(len(probe_positions))]
        
        for i, (x, y) in enumerate(probe_positions):
            ax.plot(x, y, '*', color=colors[i % len(colors)],
                    markersize=12, markeredgecolor='black', markeredgewidth=0.8)
    
    @staticmethod
    def compute_residuals(times1: np.ndarray, data1: np.ndarray,
                         times2: np.ndarray, data2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        interp_func = interp1d(times1, data1, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        data1_interp = interp_func(times2)
        return times2, data2 - data1_interp
    
    @staticmethod
    def calculate_colorbar_range(data: np.ndarray, field_config: FieldConfig) -> Tuple[float, float]:
        # If explicitly set, honor fixed bounds
        if field_config.vmin is not None and field_config.vmax is not None:
            return field_config.vmin, field_config.vmax

        data_clean = data[np.isfinite(data)]
        if len(data_clean) == 0:
            return 0, 1

        if field_config.use_log_scale:
            # For log scale, ignore non-positive values and use robust percentiles
            pos = data_clean[data_clean > 0]
            if len(pos) == 0:
                # No positive values; fall back to linear small range
                return 0, 1

            vmax = np.percentile(pos, 99.0) if field_config.vmax is None else field_config.vmax
            # pick a small, but positive vmin based on low percentile of positive values
            vmin_candidate = np.percentile(pos, 1.0) if field_config.vmin is None else field_config.vmin
            # Guard against vmin >= vmax
            vmin = min(vmin_candidate, vmax / 10.0)
            # Ensure strictly positive vmin
            vmin = max(vmin, np.min(pos[pos > 0]) * 0.5, np.finfo(float).tiny)

            # Cap extreme dynamic ranges to avoid unreadable colorbars
            ratio = vmax / max(vmin, np.finfo(float).tiny)
            if ratio > 1e8:
                vmin = vmax / 1e8

            return float(vmin), float(vmax)
        else:
            # Linear scale: trim extremes and compute bounds
            p995, p005 = np.percentile(data_clean, [99.5, 0.5])
            trimmed = data_clean[(data_clean >= p005) & (data_clean <= p995)]
            if len(trimmed) == 0:
                trimmed = data_clean

            vmin = max(0.0, float(np.min(trimmed))) if field_config.vmin is None else field_config.vmin
            vmax = float(np.max(trimmed)) if field_config.vmax is None else field_config.vmax

            if vmax <= vmin:
                vmax = vmin + 1e-12
            return vmin, vmax
    
    @staticmethod
    def plot_water_table(ax, t: float, bc_manager, domain):
        if not bc_manager or not domain:
            return
        
        try:
            left_wt, right_wt = bc_manager.get_water_table(t)
            
            if abs(left_wt - right_wt) < 0.01:
                ax.axhline(y=left_wt, color='blue', linestyle='--',
                          linewidth=2, label=f'Water Table ({left_wt:.1f}m)', alpha=0.8)
            else:
                x_vals = np.linspace(0, domain.Lx, 100)
                wt_vals = np.linspace(left_wt, right_wt, 100)
                ax.plot(x_vals, wt_vals, color='blue', linestyle='--',
                       linewidth=2, label=f'Water Table ({left_wt:.1f}m - {right_wt:.1f}m)', 
                       alpha=0.8)
        except Exception:
            pass
    
    @staticmethod
    def plot_material_curves(materials_dict: Dict, figsize=(20, 4)):
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        hp_range = np.linspace(-10, 0, 1000)
        param_text = "SOIL PARAMETERS\n\n"
        
        for mat_name, material in materials_dict.items():
            hydraulic = material.hydraulic
            
            param_text += f"{mat_name}:\n"
            param_text += f"  Ks = {material.soil.Ks:.2e} m/s\n"
            param_text += f"  φ = {material.soil.porosity:.3f} -\n"
            
            if hasattr(hydraulic, 'params'):
                params = hydraulic.params
                param_text += f"  Model: Van Genuchten\n"
                param_text += f"  θs = {params.theta_s:.3f} m³/m³\n"
                param_text += f"  θr = {params.theta_r:.3f} m³/m³\n"
                param_text += f"  α = {params.alpha:.4f} m⁻¹\n"
                param_text += f"  n = {params.n:.3f} -\n"
                param_text += f"  m = {params.m:.3f} -\n"
            elif hasattr(hydraulic, '_theta_r'):
                param_text += f"  Model: Curve-based\n"
                param_text += f"  θs = {hydraulic._theta_s:.3f} m³/m³\n"
                param_text += f"  θr = {hydraulic._theta_r:.3f} m³/m³\n"
                param_text += f"  ε = {hydraulic.epsilon:.4f} m\n"
                param_text += f"  Ss = {hydraulic.Ss:.2e} 1/m\n"
            
            param_text += "\n"
            
            if hasattr(hydraulic, 'params'):
                theta_values = [hydraulic._theta(hp) for hp in hp_range]
                kr_values = [hydraulic._kr(hp) for hp in hp_range]
                se_values = [hydraulic._Se(hp) for hp in hp_range]
                
                axes[0].plot(hp_range, theta_values, label=mat_name, linewidth=2.5)
                axes[1].plot(hp_range, kr_values, label=mat_name, linewidth=2.5)
                axes[2].plot(hp_range, se_values, label=mat_name, linewidth=2.5)
                
            elif hasattr(hydraulic, '_theta_interp'):
                theta_values = [hydraulic._theta(hp) for hp in hp_range]
                kr_values = [hydraulic._kr(hp) for hp in hp_range]
                se_values = [(hydraulic._theta(hp) - hydraulic._theta_r) /
                            (hydraulic._theta_s - hydraulic._theta_r) for hp in hp_range]
                
                axes[0].plot(hp_range, theta_values, label=f"{mat_name} (interp)", linewidth=2.5)
                axes[1].plot(hp_range, kr_values, label=f"{mat_name} (interp)", linewidth=2.5)
                axes[2].plot(hp_range, se_values, label=mat_name, linewidth=2.5)
                
                if hasattr(hydraulic._theta_interp, 'curve'):
                    curve = hydraulic._theta_interp.curve
                    axes[0].scatter(curve.x_values, curve.y_values,
                                  marker='o', s=40, alpha=0.7, label=f"{mat_name} (data)")
                
                if hasattr(hydraulic._kr_interp, 'curve'):
                    curve = hydraulic._kr_interp.curve
                    axes[1].scatter(curve.x_values, curve.y_values,
                                  marker='o', s=40, alpha=0.7, label=f"{mat_name} (data)")
        
        axes[0].set_xlabel('Pressure Head Hp (m)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Water Content θ (m/m³)', fontsize=11, fontweight='bold')
        axes[0].set_title('Water Content Curve', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)
        axes[0].set_xlim(-10, 0)
        
        axes[1].set_xlabel('Pressure Head Hp (m)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Relative Permeability Kr (-)', fontsize=11, fontweight='bold')
        axes[1].set_title('Relative Permeability Curve', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)
        axes[1].set_xlim(-10, 0)
        axes[1].set_ylim(0, 1.05)
        
        axes[2].set_xlabel('Pressure Head Hp (m)', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Effective Saturation Se (-)', fontsize=11, fontweight='bold')
        axes[2].set_title('Effective Saturation Curve', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=9)
        axes[2].set_xlim(-10, 0)
        axes[2].set_ylim(0, 1.05)
        
        axes[3].axis('off')
        axes[3].text(0.05, 0.95, param_text, transform=axes[3].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[3].set_title('Material Parameters', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_domain_geometry(Lx: float, Ly: float, regions: Dict, figsize=(14, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        
        material_colors = {
            'base': '#8B4513',
            'GI': '#228B22',
        }
        
        base_rect = mpatches.Rectangle(
            (0, 0), Lx, Ly,
            linewidth=2, edgecolor='black',
            facecolor=material_colors.get('base', '#8B4513'),
            alpha=0.7, label='Till'
        )
        ax.add_patch(base_rect)
        
        for region_name, region_data in regions.items():
            if region_name == 'base':
                continue
            
            x_min, x_max = region_data['x_bounds']
            y_min, y_max = region_data['y_bounds']
            width = x_max - x_min
            height = y_max - y_min
            
            color = material_colors.get(region_name, '#32CD32')
            
            rect = mpatches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=3, edgecolor='darkgreen',
                facecolor=color,
                alpha=0.8, label=region_data.get('label', region_name)
            )
            ax.add_patch(rect)
            
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            ax.text(cx, cy, region_name.upper(), ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen',
                            edgecolor='white', linewidth=2, alpha=0.9))
        
        ax.set_xlabel('x (m)', fontsize=13, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=13, fontweight='bold')
        ax.set_title('Domain Geometry', fontsize=14, fontweight='bold')
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        return fig, ax