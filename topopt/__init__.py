"""topopt - a small, documented 2D SIMP topology optimiser in Python."""

from .cases import CASES, Problem, make_case
from .fem import FEModel, element_stiffness
from .optimizer import Result, Settings, optimize

__version__ = "1.0.0"
__all__ = ["CASES", "Problem", "make_case", "FEModel", "element_stiffness",
           "Result", "Settings", "optimize", "__version__"]
