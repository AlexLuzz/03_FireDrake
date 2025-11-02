import sys
import os
# Add the parent directory to Python path to import the main package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *


theta_curve = HydraulicCurves.till_theta()
kr_curve = HydraulicCurves.till_kr()

hydro_model = CurveBasedHydraulicModel(theta_curve=theta_curve, kr_curve=kr_curve).fit_van_genuchten()

""""Fit Van Genuchten parameters to till curves and report goodness of fit

Fitted parameters:
  alpha = 0.9399 [1/m]
  n = 2.3579 [-]
  m = 0.5759 [-]
  l = 0.0000 [-]

Goodness of fit (theta):
  RMSE = 0.01085 [m³/m³]
  MAE  = 0.00770 [m³/m³]
  R²   = 0.9510

Goodness of fit (kr):
  RMSE = 0.02420 [-]
  MAE  = 0.01322 [-]
  R²   = 0.9964
"""

theta_curve = HydraulicCurves.terreau_theta()
kr_curve = HydraulicCurves.terreau_kr()

hydro_model = CurveBasedHydraulicModel(theta_curve=theta_curve, kr_curve=kr_curve).fit_van_genuchten()

"""
Fitted parameters:
  alpha = 1.1670 [1/m]
  n = 2.1052 [-]
  m = 0.5250 [-]
  l = 0.0000 [-]

Goodness of fit (theta):
  RMSE = 0.01519 [m³/m³]
  MAE  = 0.01253 [m³/m³]
  R²   = 0.9610

Goodness of fit (kr):
  RMSE = 0.01322 [-]
  MAE  = 0.00460 [-]
  R²   = 0.8035
"""
