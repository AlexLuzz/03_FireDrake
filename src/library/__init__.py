"""
Material and property library
"""
from .soils import SoilParameters
from .contaminants import ContaminantProperties
from .soils_from_curve import HydraulicCurves

__all__ = ['SoilParameters', 'ContaminantProperties', 'HydraulicCurves']
