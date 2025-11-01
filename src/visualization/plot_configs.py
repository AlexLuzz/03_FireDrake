from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class FieldConfig:
    field_name: str
    label: str = None
    units: str = ''
    data_key: str = None
    transform_func: Optional[Callable] = None
    # transform_func=lambda x: np.array(x) * 1e3  # Convert kg/m³ to mg/L
    colormap: str = 'viridis'
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    contour_levels: int = 11
    use_log_scale: bool = False
    
    def __post_init__(self):
        if self.label is None:
            self.label = self.field_name
        if self.data_key is None:
            self.data_key = self.field_name


@dataclass
class TimeSeriesStyle:
    linewidth: float = 2.5
    linestyle: str = '-'
    marker: str = 'o'
    markersize: float = 3
    markevery: Optional[int] = None
    alpha: float = 1.0
    color: Optional[str] = None


DEFAULT_FIELDS = {
    'water_table': FieldConfig(
        field_name='water_table',
        label='Water Table Elevation',
        units='m',
        colormap='Blues'
    ),
    'saturation': FieldConfig(
        field_name='saturation',
        label='Saturation',
        units='-',
        colormap='Blues',
        vmin=0,
        vmax=1
    ),
    'concentration': FieldConfig(
        field_name='concentration',
        label='Concentration',
        units='mg/L',
        colormap='Spectral_r',
        vmin=0,
        transform_func=lambda x: np.array(x) * 1e3
    ),
    'pressure': FieldConfig(
        field_name='pressure',
        label='Pressure Head',
        units='m',
        colormap='RdBu_r'
    ),
}


DEFAULT_STYLES = {
    'simulation': TimeSeriesStyle(
        linewidth=2.5,
        linestyle='-',
        marker='o',
        markersize=3,
        alpha=1.0
    ),
    'comsol': TimeSeriesStyle(
        linewidth=2.0,
        linestyle='--',
        marker='s',
        markersize=4,
        alpha=0.8
    ),
    'measured': TimeSeriesStyle(
        linewidth=1.5,
        linestyle=':',
        marker='^',
        markersize=3,
        alpha=0.7
    ),
}