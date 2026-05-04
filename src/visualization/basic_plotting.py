import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Dict
import matplotlib.dates as mdates

def plot_timeseries(ax, data_dict, field, source_mgr=None, use_datetime=True):
    """
    Plots probe data and overlays rain events from the SourceManager as bars.
    """
    # 1. Convert simulation seconds to datetimes if requested
    times = data_dict["times"]
    if use_datetime and source_mgr:
        plot_times = [source_mgr.config.start_datetime + pd.Timedelta(seconds=t) for t in times]
    else:
        plot_times = [t / 3600.0 for t in times] # Convert to hours

    # 2. Plot the probe data
    for name, probe in data_dict["data"].items():
        ax.plot(plot_times, probe[field], lw=2, label=name)

    # 3. Overlay Rain from SourceManager
    if source_mgr is not None:
        ax_rain = ax.twinx()
        _plot_rain_bars(ax_rain, source_mgr, use_datetime)
        ax_rain.set_ylabel("Rain Intensity (mm/hr)")
    
    _format_time_axis(ax, use_datetime)

    ax.set_ylabel(field)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

def _plot_rain_bars(ax, source_mgr, use_datetime):
    """
    Extracts data from SourceManager and plots as bar chart.
    """
    df = source_mgr.data
    # Aggregate all zones if they share columns, or pick the 'rain' column specifically
    # Converting m/s to mm/hr: value * 1000 (mm) * 3600 (s)
    rain_series = df['rain'] * 3.6e6 if 'rain' in df.columns else df.iloc[:, 0] * 3.6e6

    if use_datetime:
        x_axis = [source_mgr.config.start_datetime + pd.Timedelta(seconds=t) for t in df.index]
        # Calculate bar width in days for matplotlib dates (dt in days)
        width = source_mgr.config.dt / 86400.0
    else:
        x_axis = df.index / 3600.0
        width = source_mgr.config.dt / 3600.0

    # Plot as bars
    ax.bar(x_axis, rain_series, width=width, align='edge', 
           alpha=0.3, color='blue', label='Rainfall')
    
    # Invert Y-axis for the classic 'top-down' rain look (optional)
    ax.set_ylim(rain_series.max() * 3, 0) 

def _format_time_axis(ax, use_datetime):
    if use_datetime:
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    else:
        ax.set_xlabel("Time (hours)")

def plot_snapshot(ax, field, cfg, vmin=None, vmax=None, add_colorbar=False):

    mesh = field.function_space().mesh()
    coords = mesh.coordinates.dat.data
    values = field.dat.data

    triang = tri.Triangulation(coords[:, 0], coords[:, 1])

    if vmin is None or vmax is None:
        vmin, vmax = values.min(), values.max()

    cf = ax.tricontourf(triang, values,
                        levels=cfg.contour_levels,
                        cmap=cfg.colormap,
                        norm=norm)

    ax.set_aspect("equal")
    ax.set(xlabel="x (m)", ylabel="y (m)")

    if add_colorbar:
        plt.colorbar(cf, ax=ax, label=f"{cfg.label} ({cfg.units})")

    return cf


    
def add_probe_markers(ax, probe_positions: List[Tuple], colors: List[str] = None):
    if colors is None:
        base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [base_colors[i % len(base_colors)] for i in range(len(probe_positions))]
    
    for i, (x, y) in enumerate(probe_positions):
        ax.plot(x, y, '*', color=colors[i % len(colors)],
                markersize=12, markeredgecolor='black', markeredgewidth=0.8)
