"""
Implementation of distributions for sampling and for importance sampling
"""


import numpy as np
import tensorflow as tf

from .base import Mapping
from ..utils import tfutils, special_math


class GaussianMap(Mapping):
    """A 1-dimensional Normal with given mean and std."""

    def __init__(self, mean=0.0, std=1.0):
        """
        Args:
            mean: float, location of the peak
            std: float, standard deviation
        """
        super().__init__()
        self._shape = tf.TensorShape([1])

        self.mean = tf.constant(mean, dtype=self._dtype)

        if std < 0:
            raise TypeError("Standard deviation must be positive.")

        self.log_std = tf.constant(tf.math.log(std), dtype=self._dtype)

    def _standardize(self, x):
        """Standardize input `x` to a unit normal."""
        return (x - self.mean) * tf.math.exp(-1 * self.log_std)

    def _forward(self, x, condition):
        """In this 1-dimensional case
        the forward pass of the mapping (distribution)
        coincides with the cumulative distribution function (cdf).
        """
        # Note: the condition is ignored.
        return special_math.ndtr(self._standardize(x))

    def _inverse(self, z, condition):
        """In this 1-dimensional case
        the inverse pass of the mapping (distribution)
        coincides with the quantile function.
        """
        # Note: the condition is ignored.
        return tf.math.ndtri(z) * tf.math.exp(self.log_std) + self.mean

    def _log_det(self, x_or_z, condition, inverse=False):
        # Note: the condition is ignored.
        if inverse:
            raise NotImplementedError()
        else:
            # the log derivative of the cdf (dF/dx)
            log_norm = tf.constant(- 0.5 * np.log(2 * np.pi) - self.log_std, dtype=self.mean.dtype)
            log_inner = - 0.5 * tf.math.exp(-2 * self.log_std) * ((x_or_z - self.mean) ** 2)
            log_det_fn = log_inner - log_norm
            return log_det_fn

    def _sample(self, num_samples, condition):
        if condition is None:
            eps = tf.random.normal((num_samples, *self._shape), dtype=self._dtype)
            return tf.math.exp(self.log_std) * eps + self.mean
        else:
            # The value of the context is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            eps = tf.random.normal(condition_size * num_samples, *self._shape, dtype=self._dtype)
            samples = tf.math.exp(self.log_std) * eps + self.mean
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])
