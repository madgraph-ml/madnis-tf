"""
Implementation of distributions for sampling and for importance sampling
"""


from typing import Tuple
import numpy as np
import tensorflow as tf

from ..distributions.uniform import StandardUniform
from .base import Mapping


class Identity(Mapping):
    """Identity mapping"""

    def __init__(
        self,
        shape: Tuple[int],
        **kwargs
    ):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
                ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
        """
        super().__init__(StandardUniform(shape), **kwargs)
        self._shape = tf.TensorShape(shape)

    def _forward(self, x, condition):
        # Note: the condition is ignored.
        del condition
        return x, 0.

    def _inverse(self, z, condition):
        # Note: the condition is ignored.
        del condition
        return z, 0.

    def _det(self, x_or_z, condition=None, inverse=False):
        # Note: the condition is ignored.
        del condition, x_or_z, inverse
        return 1.
    
    def _log_det(self, x_or_z, condition=None, inverse=False):
        # Note: the condition is ignored.
        del condition, x_or_z, inverse
        return 0.

    def _sample(self, num_samples, condition):
        sample = self.base_dist.sample(num_samples, condition)
        return sample
