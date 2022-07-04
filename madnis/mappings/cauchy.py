"""
Implementation of distributions for sampling and for importance sampling
"""

import numpy as np
import tensorflow as tf

from .base import Mapping
from ..utils import tfutils


class CauchyMap(Mapping):
    """Cauchy mapping with given mean and gamma (FWHMW)."""

    def __init__(self, mean=0.0, gamma=1.0):
        """
        Args:
            mean: float, location of the peak
            gamma: float, scale parameter (FWHM)
        """
        super().__init__()
        self._shape = tf.TensorShape([1])

        self.mean = tf.constant(mean, dtype=self._dtype)
        self.gamma = tf.constant(gamma, dtype=self._dtype)
        self.tf_pi = tf.constant(np.pi, dtype=self._dtype)
        self.prefactor = tf.constant(1 / (np.pi * self.gamma), dtype=self._dtype)

    def _forward(self, x, condition):
        """In this 1-dimensional case
        the forward pass of the mapping (distribution)
        coincides with the cumulative distribution function (cdf).
        """
        # Note: the condition is ignored.
        cumulative_fn = tf.math.atan((x - self.mean) / self.gamma)
        return 1 / self.tf_pi * cumulative_fn + 0.5

    def _inverse(self, z, condition):
        """In this 1-dimensional case
        the inverse pass of the mapping (distribution)
        coincides with the quantile function.
        """
        # Note: the condition is ignored.
        quantile_fn = tf.math.tan(self.tf_pi * (z - 0.5))
        return self.mean + self.gamma * quantile_fn

    def _det(self, x_or_z, condition=None, inverse=False):
        # Note: the condition is ignored.
        if inverse:
            # the derivative of the quantile function (dF^{-1}/dz)
            det_fn = self.tf_pi / (tf.math.sin(self.tf_pi * x_or_z) ** 2)
            return self.gamma * det_fn
        else:
            # the derivative of the cdf (dF/dx)
            det_fn = self.gamma / ((x_or_z - self.mean) ** 2 + self.gamma ** 2)
            return 1/(self.tf_pi) * det_fn

    def _sample(self, num_samples, condition):
        # Sample from quantile
        if condition is None:
            z_values = tf.random.uniform((num_samples, *self._shape), dtype=self._dtype)
            return self._inverse(z_values, condition)
        else:
            # The value of the context is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            z_values = tf.random.uniform((condition_size * num_samples, *self._shape), dtype=self._dtype)
            samples = self._inverse(z_values, condition)
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])
