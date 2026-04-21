import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import LinearNDInterpolator
import numpy as np

def plot_timeseries(ax, times, data, ylabel="water level (m)", **plot_kwargs):
    ax.plot(times, data, **plot_kwargs)


    return ax

def plot_domain(ax, domain):
    # Iterate through metadata to draw patches
    for name, meta in domain.region_metadata.items():
        if meta['type'] in ['layer', 'rectangle']:
            x0, x1 = meta['x_bounds']
            y0, y1 = meta['y_bounds']
            ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0, label=name, alpha=0.5))
        
        elif meta['type'] == 'polygon':
            ax.add_patch(mpatches.Polygon(meta['vertices'], label=name, alpha=0.5))
            
    ax.set_xlim(0, domain.Lx)
    ax.set_ylim(0, domain.Ly)
    ax.legend()

def plot_snapshot(ax, x, y, z, title="", cmap="viridis", vmin=0, vmax=1):
    """Draws a spatial contour of the domain."""
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    interp = LinearNDInterpolator(np.column_stack((x, y)), z)
    Zi = np.clip(interp(Xi, Yi), vmin, vmax)
    
    cf = ax.contourf(Xi, Yi, Zi, levels=11, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax.set_aspect('equal')
    ax.set_title(title)
    return cf  # Return contour object in case the caller wants to add a colorbar