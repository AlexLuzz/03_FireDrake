import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Dict
import matplotlib.dates as mdates
from matplotlib.tri import Triangulation

def plot_timeseries(ax, times, data, field, source_mgr=None, ref_time=None):
    """
    Plots probe data and overlays rain events from the SourceManager as bars.
    """
    # 1. Ensure we have a reference datetime
    ref = ref_time or (source_mgr.config.start_datetime if source_mgr else None)
    
    # 2. Convert numeric seconds to datetimes if necessary
    if np.issubdtype(np.array(times).dtype, np.number):
        times = [ref + pd.Timedelta(seconds=t) for t in times]

    # 2. Plot the probe data
    for name, probe in data.items():
        ax.plot(times, probe[field], lw=2, label=name)

    # 3. Overlay Rain from SourceManager
    if source_mgr is not None:
        ax_rain = ax.twinx()
        _plot_rain_bars(ax_rain, source_mgr)
        ax_rain.set_ylabel("Rain Intensity (mm/hr)")
    
    _format_time_axis(ax)

    ax.set_ylabel(field)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

def _plot_rain_bars(ax, source_mgr):
    """
    Extracts data from SourceManager and plots as bar chart.
    """
    df = source_mgr.data
    # Aggregate all zones if they share columns, or pick the 'rain' column specifically
    # Converting m/s to mm/hr: value * 1000 (mm) * 3600 (s)
    rain_series = df['rain'] * 3.6e6 if 'rain' in df.columns else df.iloc[:, 0]

    x_axis = [source_mgr.config.start_datetime + pd.Timedelta(seconds=t) for t in df.index]
    # Calculate bar width in days for matplotlib dates (dt in days)
    width = source_mgr.config.dt / 86400.0

    # Plot as bars
    ax.bar(x_axis, rain_series, width=width, align='edge', 
           alpha=0.3, color='blue', label='Rainfall')
    
    # Invert Y-axis for the classic 'top-down' rain look (optional)
    ax.set_ylim(rain_series.max() * 3, 0) 

def _format_time_axis(ax, ):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def plot_snapshot(ax, data, cfg, mesh=None, vmin=None, vmax=None,
                  scale="linear"):

    if hasattr(data, "function_space"):
        mesh = data.function_space().mesh()
        values = data.dat.data
    else:
        values = data
        if mesh is None:
            raise ValueError("mesh required for numpy data")

    coords = mesh.coordinates.dat.data
    triang = Triangulation(coords[:, 0], coords[:, 1])

    norm = None
    if scale == "log":
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)

    cf = ax.tricontourf(
        triang,
        values,
        levels=cfg.contour_levels,
        cmap=cfg.colormap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extend='both'
    )

    ax.set_aspect("equal")
    ax.set(xlabel="x (m)", ylabel="y (m)")

    return cf

def plot_snapshot_grid(ax, fields, cfg, mesh=None, rows=3, cols=2):

    fig = ax.figure
    ax.remove()

    gs = fig.add_gridspec(rows, cols, hspace=0.1, wspace=0.1)

    axes = np.array([
        fig.add_subplot(gs[r, c])
        for r in range(rows)
        for c in range(cols)
    ])
    vals = []
    for f in fields.values():
        v = f["saturation"].dat.data
        vals.append(np.asarray(v))

    vmin = np.min([v.min() for v in vals])
    vmax = np.max([v.max() for v in vals])

    cf = None
    # plots
    for i, f in enumerate(fields.values()):
        if i < len(axes):
            cf = plot_snapshot(axes[i], f["saturation"], cfg, mesh, vmin=vmin, vmax=vmax)
            axes[i].set_title(f"Snapshot {i+1}")
    for j in range(len(fields), len(axes)):
        axes[j].axis("off")

    # axis labels (outer only)
    for i, ax_i in enumerate(axes):
        r, c = divmod(i, cols)
        if r != rows - 1:
            ax_i.set_xlabel(""); ax_i.set_xticklabels([])
        if c != 0:
            ax_i.set_ylabel(""); ax_i.set_yticklabels([])

    cbar = fig.colorbar(cf, ax=axes, label=f"{cfg.label} ({cfg.units})")
    cbar.ax.set_position([0.85, 0.15, 0.02, 0.7])

    return fig, axes
    
def add_probe_markers(ax, probe_positions: List[Tuple], colors: List[str] = None):
    if colors is None:
        base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [base_colors[i % len(base_colors)] for i in range(len(probe_positions))]
    
    for i, (x, y) in enumerate(probe_positions):
        ax.plot(x, y, '*', color=colors[i % len(colors)],
                markersize=12, markeredgecolor='black', markeredgewidth=0.8)
