import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import numpy as np
from typing import Callable, List, Tuple, Dict
import matplotlib.dates as mdates

def plot_timeseries(ax, data_dict, field, rain_events=None, use_datetime=True):
    times = data_dict["times"]

    for name, probe in data_dict["data"].items():
        ax.plot(times, probe[field], lw=2, label=name)

    if rain_events:
        ax_rain = ax.twinx()
        _plot_rain(ax_rain, rain_events, use_datetime)
    else:
        ax_rain = None

    _format_time_axis(ax, use_datetime)

    ax.set_ylabel(field)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

def _format_time_axis(ax, use_datetime):
    if use_datetime:
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.figure.autofmt_xdate()
    else:
        ax.set_xlabel("Time (hours)")

def _plot_rain(ax, events, use_datetime):
    t, r = [], []

    for e in sorted(events, key=lambda x: x.start):
        t0 = e.start_datetime if use_datetime else e.start / 3600
        t1 = e.end_datetime if use_datetime else e.end / 3600

        t += [t0, t1]
        r += [e.rate, e.rate]

    if t:
        t.append(t[-1])
        r.append(0)

        ax.fill_between(t, 0, r, step="post", alpha=0.3)
        ax.plot(t, r, drawstyle="steps-post", lw=1.5, label="Rain")
    
    ax.set_ylabel("Rain (mm)")
    ax.set_ylim(bottom=0)

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
