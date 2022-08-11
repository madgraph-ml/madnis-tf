""" Mappings for the MadNIS framework """

# Import the base class first
from .base import *

# Then all inheriting modules
from .cauchy import *
from .cauchy_2d import *
from .cauchy_nD import *
from .gaussian import *
from .flow import *

__all__ = [
    "Mapping",
    "CauchyDistribution",
    "CauchyRingMap",
    "CauchyLineMap",
    "MultiDimCauchy",
    "GaussianMap",
    "Flow",
]
