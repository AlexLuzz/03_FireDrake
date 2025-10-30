"""
Soil hydraulic curves library
Empirical curve data for various soil types
"""
import numpy as np
from ..physics.curve_tools import CurveData


class HydraulicCurves:
    """Library of predefined hydraulic curves for various soil types"""
    
    @staticmethod
    def till_theta() -> CurveData:
        """Till: pressure → water content"""
        pressure_MPa = np.array([-9.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.050, -0.020, -0.010, 0.000])
        pressure_m = pressure_MPa * 1e6 / 9810
        theta = np.array([0.011, 0.014, 0.019, 0.023, 0.027, 0.034, 0.039, 0.045, 0.059, 0.083, 0.140])
        
        return CurveData(pressure_m, theta, x_name="pressure", y_name="theta", 
                        units_x="m", units_y="m³/m³")
    
    @staticmethod
    def till_kr() -> CurveData:
        """Till: pressure → relative permeability"""
        pressure_MPa = np.array([-9.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.050, -0.020, -0.010, 0.000])
        pressure_m = pressure_MPa * 1e6 / 9810
        kr = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.010, 0.010, 0.070, 1.000])
        
        return CurveData(pressure_m, kr, x_name="pressure", y_name="kr", 
                        units_x="m", units_y="-")
    
    @staticmethod
    def terreau_theta() -> CurveData:
        """Terreau: pressure → water content"""
        pressure_MPa = np.array([-9.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.050, -0.020, -0.010, 0.000])
        pressure_m = pressure_MPa * 1e6 / 9810
        theta = np.array([0.017, 0.021, 0.028, 0.035, 0.042, 0.052, 0.061, 0.076, 0.138, 0.283, 0.430])
        
        return CurveData(pressure_m, theta, x_name="pressure", y_name="theta",
                        units_x="m", units_y="m³/m³")
    
    @staticmethod
    def terreau_kr() -> CurveData:
        """Terreau: pressure → relative permeability"""
        pressure_MPa = np.array([-9.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.050, -0.020, -0.010, 0.000])
        pressure_m = pressure_MPa * 1e6 / 9810
        kr = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.010, 0.100, 1.000])
        
        return CurveData(pressure_m, kr, x_name="pressure", y_name="kr",
                        units_x="m", units_y="-")
