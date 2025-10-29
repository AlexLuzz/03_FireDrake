import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools import *

if __name__ == "__main__":
    # Simple test
    comsol = load_comsol_data()
    measured = load_measured_data(time_converter=TimeConverter(DEFAULT_COMSOL_REF_DATE))
    fig = preview_data(comsol, measured, 
                       time_converter=TimeConverter(DEFAULT_COMSOL_REF_DATE)
                       )