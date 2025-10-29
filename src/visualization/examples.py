"""
Usage Examples - Clean Version
"""
from results_plotter_clean import (
    ResultsPlotter, 
    PlotFieldConfig, 
    PlotTimeSeriesConfig,
    get_default_field_configs,
    get_default_timeseries_configs
)
import numpy as np


# ============================================================================
# Example 1: Basic usage (backward compatible)
# ============================================================================
plotter = ResultsPlotter(
    config=config,
    mesh=mesh,
    probe_manager=probe_manager,
    rain_scenario=rain_scenario,
    domain=domain,
    snapshot_manager=snapshot_manager
)

# Rain bars appear automatically on twin axis!
plotter.plot_complete_results(filename='results.png')


# ============================================================================
# Example 2: Custom field configuration
# ============================================================================
field_configs = {
    'nitrate': PlotFieldConfig(
        field_name='nitrate',
        label='NO₃⁻ Concentration',
        units='mg/L',
        data_key='concentration',
        colormap='YlOrRd',
        vmin=0,
        vmax=100,
        contour_levels=15
    ),
    'chloride': PlotFieldConfig(
        field_name='chloride',
        label='Cl⁻ Concentration',
        units='mg/L',
        data_key='concentration_cl',
        colormap='Greens',
        vmin=0,
        vmax=50
    )
}

plotter = ResultsPlotter(
    config, mesh, probe_manager, rain_scenario, domain, snapshot_manager,
    field_configs=field_configs
)

plotting_config = {
    'time_series_fields': ['nitrate', 'chloride'],
    'snapshot_fields': ['nitrate', 'chloride'],
    'snapshot_overlay': False
}

plotter.plot_complete_results(plotting_config=plotting_config)


# ============================================================================
# Example 3: Custom time series styling
# ============================================================================
timeseries_configs = {
    'simulation': PlotTimeSeriesConfig(
        linewidth=3.0,
        linestyle='-',
        marker='o',
        markersize=4,
        alpha=0.9
    ),
    'comsol': PlotTimeSeriesConfig(
        linewidth=2.5,
        linestyle='--',
        marker='s',
        markersize=5,
        alpha=0.7
    )
}

plotter = ResultsPlotter(
    config, mesh, probe_manager, rain_scenario, domain, snapshot_manager,
    timeseries_configs=timeseries_configs
)

plotter.plot_complete_results()


# ============================================================================
# Example 4: Multiple fields on same plot + overlay in snapshots
# ============================================================================
plotting_config = {
    'time_series_fields': ['water_table', 'concentration'],  # Both on same plot
    'snapshot_fields': ['saturation', 'concentration'],      # Overlay in snapshots
    'snapshot_overlay': True
}

plotter.plot_complete_results(plotting_config=plotting_config)


# ============================================================================
# Example 5: Field with transform function
# ============================================================================
def log_transform(data):
    """Log transform for large dynamic range"""
    return np.log10(data + 1e-10)

def pressure_to_head(pressure, z=0.0):
    """Convert pressure to hydraulic head"""
    return pressure + z

field_configs = {
    'velocity': PlotFieldConfig(
        field_name='velocity',
        label='Velocity',
        units='m/s',
        data_key='velocity',
        transform_func=log_transform,
        colormap='plasma',
        use_log_scale=True
    ),
    'hydraulic_head': PlotFieldConfig(
        field_name='hydraulic_head',
        label='Hydraulic Head',
        units='m',
        data_key='pressure',
        transform_func=lambda p: pressure_to_head(p, z=1.5),
        colormap='viridis'
    )
}

plotter = ResultsPlotter(
    config, mesh, probe_manager, rain_scenario, domain, snapshot_manager,
    field_configs=field_configs
)


# ============================================================================
# Example 6: Standalone time series plot
# ============================================================================
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

probe_data = probe_manager.get_data()

fig = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 1, figure=fig)

plotter.plot_time_series(
    fig=fig,
    gs_slice=gs[0, :],
    probe_data=probe_data,
    field_names=['water_table', 'concentration'],
    show_xlabel=True,
    use_datetime=True,
    plot_rain=True  # Rain intensity bars on twin axis
)

plt.tight_layout()
plt.savefig('timeseries.png', dpi=300)


# ============================================================================
# Example 7: Standalone snapshot plot
# ============================================================================
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.35)

snapshots = snapshot_manager.snapshots

plotter.plot_snapshots(
    fig=fig,
    gs=gs,
    snapshots=snapshots,
    field_names=['saturation', 'concentration'],
    overlay=True,  # Overlay fields
    start_row=0,
    use_datetime=True
)

plt.savefig('snapshots.png', dpi=300, bbox_inches='tight')


# ============================================================================
# Example 8: Modify configs at runtime
# ============================================================================
plotter = ResultsPlotter(config, mesh, probe_manager, rain_scenario, domain, snapshot_manager)

# Add new field
plotter.field_configs['my_field'] = PlotFieldConfig(
    field_name='my_field',
    label='My Custom Field',
    units='units',
    colormap='coolwarm'
)

# Modify existing field
plotter.field_configs['water_table'].colormap = 'terrain'
plotter.field_configs['water_table'].vmax = 5.0

# Modify time series style
plotter.timeseries_configs['simulation'].linewidth = 3.5
plotter.timeseries_configs['simulation'].marker = 'D'


# ============================================================================
# Example 9: Complete workflow with comparisons
# ============================================================================
# 1. Define fields
field_configs = get_default_field_configs()
field_configs['nitrate'] = PlotFieldConfig(
    field_name='nitrate',
    label='NO₃⁻',
    units='mg/L',
    colormap='YlOrRd',
    vmin=0,
    vmax=100
)

# 2. Define time series styles (optional)
timeseries_configs = get_default_timeseries_configs()
timeseries_configs['simulation'].linewidth = 3.0

# 3. Initialize plotter
plotter = ResultsPlotter(
    config, mesh, probe_manager, rain_scenario, domain, snapshot_manager,
    field_configs=field_configs,
    timeseries_configs=timeseries_configs
)

# 4. Configure plotting
plotting_config = {
    'time_series_fields': ['water_table', 'nitrate'],
    'plot_comsol_comparison': True,
    'comsol_data_file': 'data/comsol.csv',
    'plot_measured_comparison': True,
    'measured_data_file': 'data/measured.csv',
    'measured_offsets': {'LTC 101': 0.6},
    'plot_snapshots': True,
    'snapshot_fields': ['saturation', 'nitrate'],
    'snapshot_overlay': True
}

# 5. Generate
plotter.plot_complete_results(
    filename='complete_results.png',
    plot_dates=True,
    plotting_config=plotting_config
)


# ============================================================================
# Example 10: Different probe colors
# ============================================================================
probe_data = probe_manager.get_data()

fig = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 1, figure=fig)

custom_colors = ['darkblue', 'darkgreen', 'darkred']

plotter.plot_time_series(
    fig=fig,
    gs_slice=gs[0, :],
    probe_data=probe_data,
    field_names=['water_table'],
    probe_colors=custom_colors,  # Custom colors
    plot_rain=True
)

plt.savefig('custom_colors.png', dpi=300)