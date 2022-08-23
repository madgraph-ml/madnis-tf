"""
Implementation of distributions for sampling and for importance sampling
"""

import tensorflow as tf
from typing import Tuple

from .base import Distribution
from ..utils import tfutils


class StandardUniform(Distribution):
    """A multivariate Uniform with boundaries (0,1)."""

    def __init__(self, shape: Tuple[int], **kwargs):
        """
        Args:
            shape: list, tuple or tf.TensorShape, containing the dimension
                with shape (dim_1, dim_2,...) without the batch dimension 'dim_0'.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape(shape)
        self._zero = tf.zeros(self._shape, dtype=self._dtype)
        self._one = tf.ones(self._shape, dtype=self._dtype)

    def _log_prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        if x.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(self._shape, x.shape[1:])
            )

        lb = tfutils.mean_except_batch(
            tf.cast(tf.math.greater_equal(x, self._zero), dtype=self._dtype)
        )
        ub = tfutils.mean_except_batch(
            tf.cast(tf.math.less_equal(x, self._one), dtype=self._dtype)
        )
        return tf.math.log(lb * ub)

    def _sample(self, num_samples, condition):
        del condition
        return tf.random.uniform((num_samples, *self._shape), dtype=self._dtype)
