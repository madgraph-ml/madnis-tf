"""
Implementation of distributions for sampling and for importance sampling
"""


import numpy as np
import tensorflow as tf

from .base import Mapping
from ..distributions.uniform import StandardUniform
from ..utils import tfutils, special_math


class GaussianMap(Mapping):
    """A 1-dimensional Normal with given mean and std."""

    def __init__(self, mean: float=0.0, std: float=1.0, **kwargs):
        """
        Args:
            mean: float, location of the peak
            std: float, standard deviation
        """
        super().__init__(StandardUniform([1]), **kwargs)
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
        del condition
        return special_math.ndtr(self._standardize(x))

    def _inverse(self, z, condition):
        """In this 1-dimensional case
        the inverse pass of the mapping (distribution)
        coincides with the quantile function.
        """
        # Note: the condition is ignored.
        del condition
        return tf.math.ndtri(z) * tf.math.exp(self.log_std) + self.mean
    
    @tf.function
    def _quantile_jacdet(self, z):
        """
        Calculat the jacobian determinant of the quantile function, i.e.
        of the inverse mapping
        """
        with tf.GradientTape() as tape:
            tape.watch(z)
            x = tf.math.ndtri(z) * tf.math.exp(self.log_std) + self.mean

        # get the determinant of the contour deform
        jac = tape.batch_jacobian(x, z)
        det = tf.linalg.det(jac)
        return det

    def _log_det(self, x_or_z, condition, inverse=False):
        # Note: the condition is ignored.
        del condition
        if inverse:
            det = self._quantile_jacdet(x_or_z)
            return tf.math.log(det)
        else:
            # the log derivative of the cdf (dF/dx)
            log_norm = tf.constant(- 0.5 * np.log(2 * np.pi) - self.log_std, dtype=self.mean.dtype)
            log_inner = - 0.5 * tf.math.exp(-2 * self.log_std) * ((x_or_z - self.mean) ** 2)
            log_det_fn = log_inner - log_norm
            return log_det_fn

    def _sample(self, num_samples, condition):
        del condition
        eps = tf.random.normal((num_samples, *self._shape), dtype=self._dtype)
        return tf.math.exp(self.log_std) * eps + self.mean
