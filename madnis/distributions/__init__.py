""" Distributions for the MadNIS framework """

# Import the base class first
from .base import *

# Then all inheriting modules
from .camel import *
from .normal import *
from .uniform import *
from .gaussians_2d import *

__all__ = [
    "Distribution",
    "Camel",
    "CuttedCamel",
    "MultiDimCamel",
    "GaussianRing",
    "GaussianLine",
    "TwoChannelLineRing",
    "StandardNormal",
    "Normal",
    "DiagonalNormal",
    "ConditionalMeanNormal",
    "ConditionalDiagonalNormal",
    "StandardUniform",
]
