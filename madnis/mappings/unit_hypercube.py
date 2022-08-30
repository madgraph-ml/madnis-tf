"""
Implementation of distributions for sampling and for importance sampling
"""


from typing import Tuple
import numpy as np
import tensorflow as tf

from ..distributions.uniform import StandardUniform
from ..distributions.normal import StandardNormal
from ..utils.tfutils import sum_except_batch
from .base import Mapping


class RealsToUnit(Mapping):
    """Map reals to unit hypercube"""

    def __init__(self, shape: Tuple[int], temperature=1.0, epsilon=1e-8, **kwargs):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
                ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
        """
        super().__init__(StandardUniform(shape), **kwargs)
        self.epsilon = epsilon
        self._dtype = tf.keras.backend.floatx()
        self.temperature = tf.constant(temperature, dtype=self._dtype)
        self._shape = tf.TensorShape(shape)

    def _forward(self, x, condition):
        # Note: the condition is ignored.
        del condition
        x = self.temperature * x
        z = tf.sigmoid(x)
        logdet = sum_except_batch(
            tf.math.log(self.temperature) - tf.math.softplus(-x) - tf.math.softplus(x)
        )
        return z, logdet

    def _inverse(self, z, condition):
        # Note: the condition is ignored.
        del condition
        z = tf.clip_by_value(
            z, clip_value_min=self.epsilon, clip_value_max=1 - self.epsilon
        )
        x = (1 / self.temperature) * (tf.math.log(z) - tf.math.log1p(-z))
        logdet = -sum_except_batch(
            tf.math.log(self.temperature)
            - tf.math.softplus(-self.temperature * x)
            - tf.math.softplus(self.temperature * x)
        )
        return x, logdet

    def _log_det(self, x_or_z, condition=None, inverse=False):
        # Note: the condition is ignored.
        del condition
        if inverse:
            z = tf.clip_by_value(
                x_or_z, clip_value_min=self.epsilon, clip_value_max=1 - self.epsilon
            )
            x = (1 / self.temperature) * (tf.math.log(z) - tf.math.log1p(-z))
            log_det = -sum_except_batch(
                tf.math.log(self.temperature)
                - tf.math.softplus(-self.temperature * x)
                - tf.math.softplus(self.temperature * x)
            )
            return log_det
        else:
            x = self.temperature * x_or_z
            log_det = sum_except_batch(
                tf.math.log(self.temperature)
                - tf.math.softplus(-x)
                - tf.math.softplus(x)
            )
            return log_det

    def _sample(self, num_samples, condition):
        z_values = self.base_dist.sample(num_samples, condition)
        # Sample from quantile
        sample, _ = self.inverse(z_values, condition)
        return sample
    
class UnitToReals(Mapping):
    """Map unit hypercube to reals"""

    def __init__(self, shape: Tuple[int], temperature=1.0, epsilon=1e-8, **kwargs):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
                ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
        """
        super().__init__(StandardNormal(shape), **kwargs)
        self.epsilon = epsilon
        self._dtype = tf.keras.backend.floatx()
        self.temperature = tf.constant(temperature, dtype=self._dtype)
        self._shape = tf.TensorShape(shape)

    def _forward(self, z, condition):
        # Note: the condition is ignored.
        del condition
        z = tf.clip_by_value(
            z, clip_value_min=self.epsilon, clip_value_max=1 - self.epsilon
        )
        x = (1 / self.temperature) * (tf.math.log(z) - tf.math.log1p(-z))
        logdet = -sum_except_batch(
            tf.math.log(self.temperature)
            - tf.math.softplus(-self.temperature * x)
            - tf.math.softplus(self.temperature * x)
        )
        return x, logdet
    
    def _inverse(self, x, condition):
        # Note: the condition is ignored.
        del condition
        x = self.temperature * x
        z = tf.sigmoid(x)
        logdet = sum_except_batch(
            tf.math.log(self.temperature) - tf.math.softplus(-x) - tf.math.softplus(x)
        )
        return z, logdet

    def _log_det(self, x_or_z, condition=None, inverse=False):
        # Note: the condition is ignored.
        del condition
        if not inverse:
            z = tf.clip_by_value(
                x_or_z, clip_value_min=self.epsilon, clip_value_max=1 - self.epsilon
            )
            x = (1 / self.temperature) * (tf.math.log(z) - tf.math.log1p(-z))
            log_det = -sum_except_batch(
                tf.math.log(self.temperature)
                - tf.math.softplus(-self.temperature * x)
                - tf.math.softplus(self.temperature * x)
            )
            return log_det
        else:
            x = self.temperature * x_or_z
            log_det = sum_except_batch(
                tf.math.log(self.temperature)
                - tf.math.softplus(-x)
                - tf.math.softplus(x)
            )
            return log_det

    def _sample(self, num_samples, condition):
        z_values = self.base_dist.sample(num_samples, condition)
        # Sample from quantile
        sample, _ = self.inverse(z_values, condition)
        return sample
