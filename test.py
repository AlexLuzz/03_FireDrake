from datetime import datetime, timedelta
from firedrake import FunctionSpace
from src import *
from src.physics import domain


F = FunctionSpace(domain.mesh, "DG", 1)