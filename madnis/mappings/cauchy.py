"""
Implementation of distributions for sampling and for importance sampling
"""

import numpy as np
import tensorflow as tf

from madnis.distributions.base import Distribution

from ..distributions.uniform import StandardUniform
from .base import Mapping
from ..utils import tfutils


class CauchyDistribution(Mapping):
    """Cauchy mapping with given mean and gamma (FWHMW)."""

    def __init__(
        self,
        base_dist: Distribution = StandardUniform([1]),
        mean: float = 0.0,
        gamma: float = 1.0,
        **kwargs
    ):
        """
        Args:
            base_dist: base distribution to start from.
                Defaults to Standard Uniform.
            mean: float, location of the peak.
            gamma: float, scale parameter (FWHM).
        """
        super().__init__(base_dist, **kwargs)
        self._shape = tf.TensorShape([1])

        self.mean = tf.constant(mean, dtype=self._dtype)
        self.gamma = tf.constant(gamma, dtype=self._dtype)
        self.tf_pi = tf.constant(np.pi, dtype=self._dtype)
        self.prefactor = tf.constant(1 / (np.pi * self.gamma), dtype=self._dtype)

    def _forward(self, x, condition):
        # Note: the condition is ignored.
        del condition

        cumulative_fn = tf.math.atan((x - self.mean) / self.gamma)
        logdet = self.log_det(x, inverse=False)
        return 1 / self.tf_pi * cumulative_fn + 0.5, logdet

    def _inverse(self, z, condition):
        # Note: the condition is ignored.
        del condition

        quantile_fn = tf.math.tan(self.tf_pi * (z - 0.5))
        logdet = self.log_det(z, inverse=True)
        return self.mean + self.gamma * quantile_fn, logdet

    def _det(self, x_or_z, condition=None, inverse=False):
        # Note: the condition is ignored.
        del condition

        if inverse:
            # the derivative of the quantile function (dF^{-1}/dz)
            det_fn = self.tf_pi / (tf.math.sin(self.tf_pi * x_or_z) ** 2)
            return tf.reduce_prod(self.gamma * det_fn, axis=-1)
        else:
            # the derivative of the cdf (dF/dx)
            det_fn = self.gamma / ((x_or_z - self.mean) ** 2 + self.gamma ** 2)
            return tf.reduce_prod(1 / (self.tf_pi) * det_fn, axis=-1)

    def _sample(self, num_samples, condition):
        z_values = self.base_dist.sample(num_samples, condition)

        # Sample from quantile
        if condition is not None:
            # Merge the condition dimension with sample dimension in order to call log_prob.
            z_values = tfutils.merge_leading_dims(z_values, num_dims=2)
            condition = tfutils.repeat_rows(condition, num_reps=num_samples)
            assert z_values.shape[0] == condition.shape[0]

        sample, _ = self.inverse(z_values, condition)

        if condition is not None:
            # Split the context dimension from sample dimension.
            sample = tfutils.split_leading_dim(sample, shape=[-1, num_samples])

        return sample
