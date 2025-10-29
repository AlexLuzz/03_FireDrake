from typing import Dict, Callable


class PlotFieldConfig:
    """Configuration for field metadata and spatial plotting"""
    def __init__(
        self,
        field_name: str,
        label: str = None,
        units: str = None,
        data_key: str = None,
        transform_func: Callable = None,
        # Spatial/snapshot parameters
        colormap: str = 'viridis',
        vmin: float = None,
        vmax: float = None,
        contour_levels: int = 11,
        use_log_scale: bool = False
    ):
        self.field_name = field_name
        self.label = label or field_name
        self.units = units or ''
        self.data_key = data_key or field_name
        self.transform_func = transform_func
        # Snapshot parameters
        self.colormap = colormap
        self.vmin = vmin
        self.vmax = vmax
        self.contour_levels = contour_levels
        self.use_log_scale = use_log_scale


class PlotTimeSeriesConfig:
    """Configuration for time series plotting style"""
    def __init__(
        self,
        linewidth: float = 2.5,
        linestyle: str = '-',
        marker: str = 'o',
        markersize: float = 3,
        markevery: int = None,
        alpha: float = 1.0,
        color: str = None
    ):
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.marker = marker
        self.markersize = markersize
        self.markevery = markevery
        self.alpha = alpha
        self.color = color


def get_default_field_configs() -> Dict[str, PlotFieldConfig]:
    """Returns default configurations for common fields"""
    return {
        'water_table': PlotFieldConfig(
            field_name='water_table',
            label='Water Table Elevation',
            units='m',
            colormap='Blues',
            vmin=0,
            vmax=1,
            data_key='water_table'
        ),
        'saturation': PlotFieldConfig(
            field_name='saturation',
            label='Saturation',
            units='-',
            colormap='Blues',
            vmin=0,
            vmax=1,
            data_key='saturation'
        ),
        'concentration': PlotFieldConfig(
            field_name='concentration',
            label='Concentration',
            units='kg/m³',
            colormap='YlOrRd',
            vmin=0,
            vmax=None,
            data_key='concentration',
            use_log_scale=False
        ),
        'pressure': PlotFieldConfig(
            field_name='pressure',
            label='Pressure Head',
            units='m',
            colormap='RdBu_r',
            vmin=None,
            vmax=None,
            data_key='pressure'
        ),
    }


def get_default_timeseries_configs() -> Dict[str, PlotTimeSeriesConfig]:
    """Returns default time series plotting configurations"""
    return {
        'simulation': PlotTimeSeriesConfig(
            linewidth=2.5,
            linestyle='-',
            marker='o',
            markersize=3,
            alpha=1.0
        ),
        'comsol': PlotTimeSeriesConfig(
            linewidth=2.0,
            linestyle='--',
            marker='s',
            markersize=4,
            alpha=0.8
        ),
        'measured': PlotTimeSeriesConfig(
            linewidth=1.5,
            linestyle=':',
            marker='^',
            markersize=3,
            alpha=0.7
        ),
    }
