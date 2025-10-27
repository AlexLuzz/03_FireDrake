"""
Generic curve interpolation system for ANY property relationship
Can be used for hydraulic, transport, thermal, geophysical properties
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Literal

# ==============================================
# GENERIC CURVE DATA
# ==============================================

@dataclass
class CurveData:
    """
    Generic empirical curve data - represents ANY x→y relationship
    Automatically sorts and validates data
    """
    x_values: np.ndarray
    y_values: np.ndarray
    x_name: str = "x"          # e.g., "pressure", "saturation", "temperature"
    y_name: str = "y"          # e.g., "theta", "kr", "D_eff"
    units_x: str = ""          # e.g., "m", "°C", "-"
    units_y: str = ""          # e.g., "m³/m³", "m²/s"
    
    def __post_init__(self):
        """Convert to numpy arrays and sort by x values"""
        self.x_values = np.array(self.x_values, dtype=float)
        self.y_values = np.array(self.y_values, dtype=float)
        
        if len(self.x_values) != len(self.y_values):
            raise ValueError("x_values and y_values must have same length")
        
        # Sort by x
        sort_idx = np.argsort(self.x_values)
        self.x_values = self.x_values[sort_idx]
        self.y_values = self.y_values[sort_idx]
    
    @property
    def x_min(self) -> float:
        return float(self.x_values[0])
    
    @property
    def x_max(self) -> float:
        return float(self.x_values[-1])
    
    @property
    def y_min(self) -> float:
        return float(self.y_values.min())
    
    @property
    def y_max(self) -> float:
        return float(self.y_values.max())
    
    def smooth(self, window_size: int = 1) -> 'CurveData':
        """
        Apply moving average smoothing
        
        Parameters:
        -----------
        window_size : int
            Window size for smoothing (1 = no smoothing)
        
        Returns:
        --------
        New CurveData with smoothed y values
        """
        if window_size <= 1:
            return self
        
        smoothed_y = np.zeros_like(self.y_values)
        for i in range(len(self.y_values)):
            start = max(0, i - window_size // 2)
            end = min(len(self.y_values), i + window_size // 2 + 1)
            smoothed_y[i] = np.mean(self.y_values[start:end])
        
        return CurveData(
            self.x_values.copy(),
            smoothed_y,
            self.x_name,
            self.y_name,
            self.units_x,
            self.units_y
        )
    
    def resample(self, n_points: int = 100, method: str = 'linear') -> 'CurveData':
        """
        Resample curve to different number of points
        Useful for creating smoother curves
        """
        x_new = np.linspace(self.x_min, self.x_max, n_points)
        interp = interp1d(self.x_values, self.y_values, kind=method, bounds_error=False, fill_value='extrapolate')
        y_new = interp(x_new)
        
        return CurveData(x_new, y_new, self.x_name, self.y_name, self.units_x, self.units_y)


# ==============================================
# GENERIC CURVE INTERPOLATOR
# ==============================================

class CurveInterpolator:
    """
    Generic 1D curve interpolation with smart extrapolation
    Works for ANY property relationship
    """
    
    def __init__(self, 
                 curve_data: CurveData,
                 method: Literal['linear', 'cubic'] = 'linear',
                 extrapolate_mode: Literal['constant', 'linear', 'clip'] = 'linear',
                 bounds: Optional[tuple] = None,
                 fill_value_below: Optional[float] = None,
                 fill_value_above: Optional[float] = None):
        """
        Parameters:
        -----------
        curve_data : CurveData
            The empirical data to interpolate
        method : str
            'linear' or 'cubic' spline
        extrapolate_mode : str
            - 'constant': Use edge values outside range
            - 'linear': Linear extrapolation
            - 'clip': Clip to [y_min, y_max]
        bounds : tuple, optional
            Hard bounds (y_min, y_max) to enforce
        fill_value_below : float, optional
            Fixed value to use below x_min (overrides extrapolate_mode)
        fill_value_above : float, optional
            Fixed value to use above x_max (overrides extrapolate_mode)
        """
        self.curve = curve_data
        self.method = method
        self.extrapolate_mode = extrapolate_mode
        self.bounds = bounds
        self.fill_value_below = fill_value_below
        self.fill_value_above = fill_value_above
        
        # Build interpolator
        if extrapolate_mode == 'linear':
            self._interp = interp1d(
                curve_data.x_values, 
                curve_data.y_values,
                kind=method,
                fill_value='extrapolate',
                bounds_error=False
            )
        else:  # constant or clip
            self._interp = interp1d(
                curve_data.x_values, 
                curve_data.y_values,
                kind=method,
                fill_value=(curve_data.y_values[0], curve_data.y_values[-1]),
                bounds_error=False
            )
    
    def __call__(self, x):
        """
        Evaluate curve at x (can be scalar or array)
        
        Parameters:
        -----------
        x : float or np.ndarray
            Point(s) at which to evaluate
        
        Returns:
        --------
        y : float or np.ndarray
            Interpolated value(s)
        """
        is_scalar = np.isscalar(x)
        x_array = np.atleast_1d(x)
        
        # Handle custom fill values
        y = np.zeros_like(x_array, dtype=float)
        
        # Mask for different regions
        below = x_array < self.curve.x_min
        above = x_array > self.curve.x_max
        inside = ~(below | above)
        
        # Interpolate inside range
        if np.any(inside):
            y[inside] = self._interp(x_array[inside])
        
        # Handle below range
        if np.any(below):
            if self.fill_value_below is not None:
                y[below] = self.fill_value_below
            else:
                y[below] = self._interp(x_array[below])
        
        # Handle above range
        if np.any(above):
            if self.fill_value_above is not None:
                y[above] = self.fill_value_above
            else:
                y[above] = self._interp(x_array[above])
        
        # Apply bounds if specified
        if self.bounds is not None:
            y = np.clip(y, self.bounds[0], self.bounds[1])
        elif self.extrapolate_mode == 'clip':
            y = np.clip(y, self.curve.y_min, self.curve.y_max)
        
        # Return scalar if input was scalar
        if is_scalar:
            return float(y[0])
        return y
    
    def derivative(self, x, dx: Optional[float] = None):
        """
        Compute dy/dx at x using numerical differentiation
        
        Parameters:
        -----------
        x : float or np.ndarray
            Point(s) at which to compute derivative
        dx : float, optional
            Step size (adaptive if None)
        
        Returns:
        --------
        dydx : float or np.ndarray
            Derivative value(s)
        """
        is_scalar = np.isscalar(x)
        x_array = np.atleast_1d(x)
        
        if dx is None:
            # Adaptive step size
            dx = 1e-6 * max(1e-10, abs(self.curve.x_max - self.curve.x_min))
        
        y_minus = self(x_array - dx)
        y_plus = self(x_array + dx)
        dydx = (y_plus - y_minus) / (2 * dx)
        
        if is_scalar:
            return float(dydx[0])
        return dydx
    
    def integrate(self, x_start: float, x_end: float, n_points: int = 1000) -> float:
        """
        Numerical integration using trapezoidal rule
        """
        x_vals = np.linspace(x_start, x_end, n_points)
        y_vals = self(x_vals)
        return np.trapz(y_vals, x_vals)


# ==============================================
# CURVE LIBRARIES
# ==============================================

class HydraulicCurves:
    """Library of predefined hydraulic curves"""
    
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

