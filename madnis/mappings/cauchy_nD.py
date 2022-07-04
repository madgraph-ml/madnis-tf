"""
Implementation of distributions for sampling and for importance sampling
"""

import numpy as np
import tensorflow as tf

from .base import Mapping
from ..utils import tfutils


class MultiDimCauchy(Mapping):
    """Multi-dimensional Cauchy mapping."""

    def __init__(
        self,
        mean: tf.Tensor,
        gamma: float,
        dims: int,
    ):
        """
        Args:
            means (List[tf.Tensor]): peak locations.
            gammas (List[float]): scale parameters (FWHM)
            dims (int): dimensionality of the distributions
        """
        super().__init__()
        self._shape = tf.TensorShape([dims])

        # check that its a proper tensor
        if not isinstance(mean, tf.Tensor):
            raise NotImplementedError("Mean is not a proper tensors")

        # check that its a float
        if isinstance(gamma, int):
            gamma = float(gamma)

        self.mean = mean
        self.gamma = tf.constant(gamma, dtype=self._dtype)
        self.D = tf.constant(dims, dtype=self._dtype)

        self.tf_pi = tf.constant(np.pi, dtype=self._dtype)
        self.prefactor = tf.constant(1 / (np.pi * self.gamma), dtype=self._dtype)

    def _forward(self, x, condition):
        """The forward pass of the mapping"""
        # Note: the condition is ignored.
        z = 1 / self.tf_pi * tf.math.atan((x - self.mean) / self.gamma) + 0.5
        return z

    def _inverse(self, z, condition):
        """The inverse pass of the mapping"""
        # Note: the condition is ignored.
        x = self.mean + self.gamma * tf.math.tan(self.tf_pi * (z - 0.5))
        return x

    def _det(self, x_or_z, condition, inverse=False):
        # Note: the condition is ignored.
        if inverse:
            # the derivative of the inverse pass (dF^{-1}/dz)
            deltas = self.tf_pi * self.gamma * 1 / (tf.math.sin(self.tf_pi * x_or_z) ** 2)
            return tf.reduce_prod(deltas, axis=-1)
        else:
            # the derivative of the forward pass (dF/dx)
            deltas = self.gamma/self.tf_pi * 1 / ((x_or_z - self.mean) ** 2 + self.gamma ** 2)
            return tf.reduce_prod(deltas, axis=-1)

    def _sample(self, num_samples, condition):
        # Sample from quantile
        if condition is None:
            z_values = tf.random.uniform((num_samples, *self._shape), dtype=self._dtype)
            return self._inverse(z_values, condition)
        else:
            # The value of the context is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            z_values = tf.random.uniform(
                (condition_size * num_samples, *self._shape), dtype=self._dtype
            )
            samples = self._inverse(z_values, condition)
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])
