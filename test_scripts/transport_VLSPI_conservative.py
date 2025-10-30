"""
Script to verify transport solver and evaluate errors, using analytical solutions from Sethi di Molfetta 2019.

Test 1: Vertical line source, Pulse Input (VLSPI) in 2D
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *

def C(x, y, t):
    """Analytical solution for VLSPI - Pulse Input from Sethi di Molfetta 2019"""
    
    left_part = m / (4 * np.pi * b * n_e * t * np.sqrt(D_xx * D_yy))
    right_part = np.exp(
                        - (x - v_x * t)**2 / (4 * D_xx * t) + 
                        - (y - v_y * t)**2 / (4 * D_yy * t)
                        )


    return left_part * right_part

def C_max(t):
    """Maximum concentration for VLSPI - Pulse Input from Sethi di Molfetta 2019"""
    return m / (4 * np.pi * b * n_e * t * np.sqrt(D_xx * D_yy) )

m = 10 # mass of solute (kg)
b = 1.0 # thickness of the aquifer (m)
n_e = 0.3 # effective porosity (-)
D_xx = 0.1 # longitudinal dispersion coefficient (m2/day)
D_yy = 0.01 # transverse dispersion coefficient (m2/day)
v_x = 1.0 # pore water velocity in x direction (m/day)
v_y = 0.0 # pore water velocity in y direction (m/day)



if __name__ == "__main__":
    main_VLSPI_conservative()