import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import LinearNDInterpolator
import numpy as np

def plot_results(results, points, V, sim_p):

    time_vals = results['time']
    time_series = results['time_series']
    snapshots = results['snapshots']

    rain_start = sim_p['start'] * 3600.0  # Convert to seconds
    rain_end = sim_p['end'] * 3600.0      # Convert to
     
    # Convert to hours for plotting
    time_hours = np.array(time_vals) / 3600.0
    rain_start_hr = rain_start / 3600.0
    rain_end_hr = rain_end / 3600.0

    # Create 3x3 grid figure
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ============================================
    # TOP ROW: All time series on one plot (spanning all 3 columns)
    # ============================================
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, green, red
    ax_top = fig.add_subplot(gs[0, :])  # Span all columns in top row

    for idx, (x, y, name) in enumerate(points):
        ax_top.plot(time_hours, time_series[name], color=colors[idx], 
                    linewidth=2.5, label=name, marker='o', markersize=2, 
                    markevery=int(len(time_hours)/50))

    # Add rain event shading
    ax_top.axvspan(rain_start_hr, rain_end_hr, alpha=0.2, color='skyblue', 
                label='Rain event', zorder=0)

    # Add water table reference line
    ax_top.axhline(y=0, color='cyan', linestyle='--', linewidth=2, 
                label='Water table (p=0)', alpha=0.7)

    # Formatting
    ax_top.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax_top.set_ylabel('Pressure head (m)', fontsize=12, fontweight='bold')
    ax_top.set_title('Pressure Response to Rain Event at Different Vertical Positions', 
                    fontsize=14, fontweight='bold', pad=15)
    ax_top.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax_top.legend(loc='best', fontsize=10, framealpha=0.9, edgecolor='black')

    # Add statistics box
    stats_lines = []
    for idx, (x, y, name) in enumerate(points):
        peak = max(time_series[name])
        initial = time_series[name][0]
        stats_lines.append(f'{name.split("(")[0].strip()}: Δp = {peak-initial:.3f}m')

    stats_text = '\n'.join(stats_lines)
    ax_top.text(0.02, 0.98, stats_text, 
                transform=ax_top.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, 
                        edgecolor='black', linewidth=1.5))

    # ============================================
    # MIDDLE AND BOTTOM ROWS: Saturation snapshots
    # ============================================
    # Select 6 snapshots evenly distributed
    sorted_times = sorted(snapshots.keys())
    selected_times = sorted_times[:6] if len(sorted_times) >= 6 else sorted_times

    # Pad with last snapshot if needed
    while len(selected_times) < 6:
        selected_times.append(selected_times[-1])

    # Get coordinate arrays for plotting
    coords = V.mesh().coordinates.dat.data
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    Lx = x_coords.max()
    Ly = y_coords.max()

    for idx, snap_time in enumerate(selected_times):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Get saturation data
        saturation_func = snapshots[snap_time]['saturation']
        sat_values = saturation_func.dat.data[:]
        
        # Clip values to valid range
        sat_values = np.clip(sat_values, 0.0, 1.0)
        
        # Create regular grid for smooth plotting
        xi = np.linspace(0, Lx, 200)
        yi = np.linspace(0, Ly, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate saturation onto regular grid
        coord_points = np.column_stack((x_coords, y_coords))  # (N, 2) array of coordinates
        interp = LinearNDInterpolator(coord_points, sat_values)
        Zi = interp(Xi, Yi)
        
        # Plot saturation field on regular grid
        levels = np.linspace(0.0, 1.0, 25)
        contourf = ax.contourf(Xi, Yi, Zi, levels=levels, 
                            cmap='Blues', extend='neither',
                            vmin=0.0, vmax=1.0)
        
        # Store first contour plot for colorbar
        if idx == 0:
            contour_for_cbar = contourf
        
        # Add contour lines for clarity
        contour_levels = [0.3, 0.5, 0.7, 0.9]
        contour = ax.contour(Xi, Yi, Zi, levels=contour_levels, 
                            colors='gray', alpha=0.3, linewidths=0.8, linestyles='--')
        
        # Add water table line
        ax.axhline(y=Ly/2.0, color='cyan', linestyle='--', 
                linewidth=2, label='Initial water table', alpha=0.7)
        
        # Mark monitoring points with colors matching time series
        for pt_idx, (x, y, name) in enumerate(points):
            ax.plot(x, y, '*', color=colors[pt_idx], markersize=15, 
                    markeredgecolor='black', markeredgewidth=1.0, zorder=10)
        
        # Formatting
        ax.set_xlabel('x (m)', fontsize=10, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=10, fontweight='bold')
        
        time_label = f't = {snap_time/3600:.1f}h'
        if rain_start <= snap_time <= rain_end:
            time_label += ' (RAIN)'
        ax.set_title(time_label, fontsize=11, fontweight='bold')
        
        ax.set_aspect('equal')
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        
        # Add legend only to first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    # Add single colorbar for all saturation plots
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.56])  # [left, bottom, width, height]
    cbar = fig.colorbar(contour_for_cbar, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Saturation', fontsize=12, fontweight='bold')

    # Set ticks from 0 to 1.0, then invert axis so 0 is at top
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0%\n(dry)', '20%', '40%', '60%', '80%', '100%\n(saturated)'])
    cbar.ax.invert_yaxis()  # Invert: 0% at top, 100% at bottom

    # Add overall title
    fig.suptitle('Richards Equation: 10-Hour Rain Event Simulation\nPressure Response and Saturation Evolution', 
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('richards_simulation.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'richards_simulation.png'")
