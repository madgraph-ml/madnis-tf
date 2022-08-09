"""
Implementation of mapping for sampling and for importance sampling
"""

from typing import Union, Tuple
import numpy as np
import tensorflow as tf

from madnis.distributions.base import Distribution
from ..distributions.uniform import StandardUniform

from .base import Mapping
from ..utils import tfutils


class MultiDimCauchy(Mapping):
    """Multi-dimensional Cauchy distribution."""

    def __init__(
        self,
        shape: Tuple[int],
        mean: Union[tf.Tensor, float] = 0.0,
        gamma: Union[tf.Tensor, float] = 1.0,
        base_dist: Distribution=None,
        **kwargs
    ):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
                ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
            base_dist (Distribution, optional): base distribution to start from.
                Defaults to None.
            means (List[tf.Tensor]): peak locations.
            gammas (List[float]): scale parameters (FWHM)
        """
        if base_dist is None:
            base_dist = StandardUniform(shape)
        
        super().__init__(base_dist, **kwargs)
        self._shape = tf.TensorShape(shape)

        # Define mean
        if isinstance(mean, (int, float)):
            self.mean = tf.constant(mean, dtype=self._dtype) * tf.ones(
                (1, *self._shape), dtype=self._dtype
            )
        elif isinstance(mean, tf.Tensor):
            assert mean.shape[1:] == self._shape
            assert mean.shape[0] == 1
            self.mean = tf.constant(mean, dtype=self._dtype)
        else:
            raise ValueError()

        # Define gamma
        if isinstance(gamma, (int, float)):
            self.gamma = tf.constant(gamma, dtype=self._dtype) * tf.ones(
                (1, *self._shape), dtype=self._dtype
            )
        elif isinstance(gamma, tf.Tensor):
            assert gamma.shape[1:] == self._shape
            assert gamma.shape[0] == 1
            self.gamma = tf.constant(gamma, dtype=self._dtype)
        else:
            raise ValueError()

        self.tf_pi = tf.constant(np.pi, dtype=self._dtype)

    def _forward(self, x, condition):
        """The forward pass of the mapping"""
        # Note: the condition is ignored.
        del condition
        
        z = 1 / self.tf_pi * tf.math.atan((x - self.mean) / self.gamma) + 0.5
        logdet = self.log_det(x)
        return z, logdet

    def _inverse(self, z, condition):
        """The inverse pass of the mapping"""
        # Note: the condition is ignored.
        del condition
        
        x = self.mean + self.gamma * tf.math.tan(self.tf_pi * (z - 0.5))
        logdet = self.log_det(z, inverse=True)
        return x, logdet

    def _det(self, x_or_z, condition, inverse=False):
        # Note: the condition is ignored.
        del condition
        
        if inverse:
            # the derivative of the inverse pass (dF^{-1}/dz)
            deltas = self.tf_pi * self.gamma * 1 / (tf.math.sin(self.tf_pi * x_or_z) ** 2)
            return tf.reduce_prod(deltas, axis=-1)
        else:
            # the derivative of the forward pass (dF/dx)
            deltas = self.gamma/self.tf_pi * 1 / ((x_or_z - self.mean) ** 2 + self.gamma ** 2)
            return tf.reduce_prod(deltas, axis=-1)

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
