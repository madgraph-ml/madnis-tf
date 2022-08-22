"""
Implementation of distributions for sampling and for importance sampling
"""


import numpy as np
import tensorflow as tf
from typing import Tuple, Callable, Union

from .base import Distribution
from ..utils import tfutils


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape: Tuple[int], **kwargs):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
            ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape(shape)

    def _log_prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        if x.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(self._shape, x.shape[1:])
            )
        log_norm = tf.convert_to_tensor(-0.5 * np.log(2 * np.pi), dtype=self._dtype)
        log_inner = -0.5 * x ** 2
        return tfutils.sum_except_batch(log_inner + log_norm, num_batch_dims=1)

    def _sample(self, num_samples, condition):
        if condition is None:
            return tf.random.normal((num_samples, *self._shape), dtype=self._dtype)
        else:
            # The value of the context is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            samples = tf.random.normal(
                condition_size * num_samples, *self._shape, dtype=self._dtype
            )
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])


class Normal(Distribution):
    """A multivariate diagonal Normal with given mean and log_std."""

    def __init__(
        self,
        shape: Tuple[int],
        mean: Union[tf.Tensor, float] = 0.0,
        log_std: Union[tf.Tensor, float] = 0.0,
        **kwargs
    ):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
            ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
            mean (float, optional): location of the peak. Defaults to 0.
            log_std (float, optional): log of standard deviation. Defaults to 0.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape(shape)

        # Define mean
        if isinstance(mean, (int, float)):
            self.mean = tf.constant(mean, dtype=self._dtype) * tf.ones(
                (1, *self._shape), dtype=self._dtype
            )
        elif isinstance(mean, tf.Tensor):
            assert mean.shape[1:] == self._shape
            assert mean.shape[0] == 1
            self.mean = tf.cast(mean, dtype=self._dtype)
        else:
            raise ValueError()

        # Define log_std
        if isinstance(log_std, (int, float)):
            self.log_std = tf.constant(log_std, dtype=self._dtype) * tf.ones(
                (1, *self._shape), dtype=self._dtype
            )
        elif isinstance(log_std, tf.Tensor):
            assert log_std.shape[1:] == self._shape
            assert log_std.shape[0] == 1
            self.log_std = tf.cast(log_std, dtype=self._dtype)
        else:
            raise ValueError()

    def _log_prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        if x.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(self._shape, x.shape[1:])
            )
        log_norm = (
            tf.convert_to_tensor(-0.5 * np.log(2 * np.pi), dtype=self._dtype) - self.log_std
        )
        log_inner = -0.5 * tf.math.exp(-2 * self.log_std) * ((x - self.mean) ** 2)
        return tfutils.sum_except_batch(log_inner + log_norm, num_batch_dims=1)

    def _sample(self, num_samples, condition):
        if condition is None:
            eps = tf.random.normal((num_samples, *self._shape), dtype=self._dtype)
            return tf.math.exp(self.log_std) * eps + self.mean
        else:
            # The value of the context is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            eps = tf.random.normal(
                condition_size * num_samples, *self._shape, dtype=self._dtype
            )
            samples = tf.math.exp(self.log_std) * eps + self.mean
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])


class DiagonalNormal(Distribution):
    """A diagonal multivariate Normal with trainable mean and log_std."""

    def __init__(self, shape: Tuple[int], **kwargs):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
            ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape(shape)

        # trainable loc and log_scale
        mean_init = tf.zeros_initializer()
        self.mean = tf.Variable(
            initial_value=mean_init(shape=(1, *self._shape), dtype=self._dtype),
            trainable=True,
        )

        log_std_init = tf.zeros_initializer()
        self.log_std = tf.Variable(
            initial_value=log_std_init(shape=(1, *self._shape), dtype=self._dtype),
            trainable=True,
        )

    def _log_prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        if x.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(self._shape, x.shape[1:])
            )
        log_norm = (
            tf.convert_to_tensor(-0.5 * np.log(2 * np.pi), dtype=self._dtype) - self.log_std
        )
        log_inner = -0.5 * tf.math.exp(-2 * self.log_std) * ((x - self.mean) ** 2)
        return tfutils.sum_except_batch(log_inner + log_norm, num_batch_dims=1)

    def _sample(self, num_samples, condition):
        if condition is None:
            eps = tf.random.normal((num_samples, *self._shape), dtype=self._dtype)
            return tf.math.exp(self.log_std) * eps + self.mean
        else:
            # The value of the context is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            eps = tf.random.normal(
                condition_size * num_samples, *self._shape, dtype=self._dtype
            )
            samples = tf.math.exp(self.log_std) * eps + self.mean
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])


class ConditionalMeanNormal(Distribution):
    """A multivariate Normal with conditional mean and fixed std."""

    def __init__(
        self,
        shape: Tuple[int],
        log_std: float = 0.0,
        embedding_net: Callable = None,
        **kwargs
    ):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
            ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
            log_std: float or Tensor, log of standard deviation. Defaults to 0.
            embedding_net: callable or None, embedded the condition to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape(shape)

        # Allow for an encoding net
        if embedding_net is None:
            self.embedding_net = lambda x: x
        else:
            self.embedding_net = embedding_net

        self.log_std = tf.constant(log_std, dtype=self._dtype)

    def _compute_mean(self, condition):
        """Compute the mean from the condition."""
        if condition is None:
            raise ValueError("Condition can't be None.")

        mean = self.embedding_net(condition)
        return mean

    def _log_prob(self, x, condition):
        if x.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(self._shape, x.shape[1:])
            )

        # compute parameters
        mean = self._compute_mean(condition)
        assert mean.shape == x.shape

        log_norm = (
            tf.convert_to_tensor(-0.5 * np.log(2 * np.pi), dtype=self._dtype) - self.log_std
        )
        log_inner = -0.5 * tf.math.exp(-2 * self.log_std) * ((x - mean) ** 2)
        return tfutils.sum_except_batch(log_inner + log_norm, num_batch_dims=1)

    def _sample(self, num_samples, condition):
        if condition is None:
            raise ValueError("Condition can't be None.")
        else:
            # compute parameters
            mean = self._compute_mean(condition)
            log_std = self.log_std * tf.ones_like(mean)
            mean = tfutils.repeat_rows(mean, num_samples)
            log_std = tfutils.repeat_rows(log_std, num_samples)

            # generate samples
            condition_size = condition.shape[0]
            eps = tf.random.normal(
                condition_size * num_samples, *self._shape, dtype=mean.dtype
            )
            samples = tf.math.exp(log_std) * eps + mean
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])


class ConditionalDiagonalNormal(Distribution):
    """A diagonal multivariate Normal with conditional mean and log_std.."""

    def __init__(self, shape: Tuple[int], embedding_net: Callable = None, **kwargs):
        """
        Args:
            shape (Tuple[int]): containing the dimensions with
            ``shape = (dim_1, dim_2,...)``, excluding the batch dimension 'dim_0'.
            embedding_net: callable or None, encodes the condition to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape(shape)

        # Allow for an encoding net
        if embedding_net is None:
            self.embedding_net = lambda x: x
        else:
            self.embedding_net = embedding_net

    def _compute_params(self, condition):
        """Compute the means and log_stds from the condition."""
        if condition is None:
            raise ValueError("Condition can't be None.")

        params = self.embedding_net(condition)
        if params.shape[-1] % 2 != 0:
            raise RuntimeError(
                "The embedding net must return a tensor which last dimension is even."
            )
        if params.shape[0] != condition.shape[0]:
            raise RuntimeError(
                "The batch dimension of the parameters is inconsistent with the input."
            )

        mean, log_std = tf.split(params, 2, -1)
        return mean, log_std

    def _log_prob(self, x, condition):
        if x.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(self._shape, x.shape[1:])
            )

        # compute parameters
        mean, log_std = self._compute_params(condition)
        assert mean.shape == x.shape and log_std.shape == x.shape

        log_norm = tf.convert_to_tensor(-0.5 * np.log(2 * np.pi), dtype=self._dtype) - log_std
        log_inner = -0.5 * tf.math.exp(-2 * log_std) * ((x - mean) ** 2)
        return tfutils.sum_except_batch(log_inner - log_norm, num_batch_dims=1)

    def _sample(self, num_samples, condition):
        if condition is None:
            raise ValueError("Condition can't be None.")
        else:
            # compute parameters
            mean, log_std = self._compute_params(condition)
            mean = tfutils.repeat_rows(mean, num_samples)
            log_std = tfutils.repeat_rows(log_std, num_samples)

            # generate samples
            condition_size = condition.shape[0]
            eps = tf.random.normal.randn(
                condition_size * num_samples, *self._shape, dtype=self.dtype
            )
            samples = tf.math.exp(log_std) * eps + mean
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])
