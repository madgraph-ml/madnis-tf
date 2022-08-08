"""Implementations of Base mapping."""

import tensorflow as tf
from typing import Any, Callable

from ..utils import typechecks
from ..utils import tfutils


class Mapping(tf.Module):
    """Base class for all mapping objects."""

    def __init__(self):
        super().__init__()
        # Define the right floating point precision
        self._dtype = tf.keras.backend.floatx()

    def forward(self, x: tf.Tensor, condition: tf.Tensor=None):
        """
        Forward pass of the mapping ``f``.
        Conventionally in MC, this is the pass from the
        momenta/data ``x`` to the random numbers ``z``
        living on the unit hypercube ``U[0,1]^d``, i.e.

        ..math::
            f(x) = z.

        Args:
            x: Tensor with shape (batch_size, n_features).
            condition: None or Tensor with shape (batch_size, n_features).
                If None, the condition is ignored. Defaults to None.

        Returns:
            z: Tensor with shape (batch_size, n_features).
        """
        x = tf.convert_to_tensor(x, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            if x.shape[0] != condition.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )
        return self._forward(x, condition)

    def _forward(self, x, condition):
        """Should be overridden by all subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _forward(...) method"
        )
        
    __call__: Callable[..., Any] = forward

    def inverse(self, z, condition=None):
        """
        Inverse pass ``f^{-1}`` of the mapping. Conventionally, this is the pass
        from the random numbers ``z` to the momenta/data ``x``, i.e.

        ..math::
            f^{-1}(z) = x.

        Args:
            z: Tensor with shape (batch_size, n_features).
            condition: None or Tensor with shape (batch_size, n_features).
                If None, the condition is ignored. Defaults to None.

        Returns:
            x: Tensor with shape (batch_size, n_features).
        """
        z = tf.convert_to_tensor(z, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            if z.shape[0] != condition.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )

        return self._inverse(z, condition)

    def _inverse(self, z, condition):
        """Should be overridden by all subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _inverse(...) method"
        )

    def log_det(self, x_or_z, condition=None, inverse=False):
        """Calculates the logarithm of the jacobian determinant
        of the mapping:

        ...math::
            log_det = log(|J|), with
            J = dz/dx = df(x)/dx

        or for the inverse mapping:

        ...math::
            log_det = log(|J_inv|), with
            J_inv = dx/dz = df^{-1}(z)/dz

        Args:
            x_or_z: Tensor with shape (batch_size, n_features).
            condition (optional): None or Tensor with shape (batch_size, n_features).
                If None, the condition is ignored. Defaults to None.
            inverse (optional): bool, whether to return the log_det of
                inverse or forward mapping. Defaults to False.

        Returns:
            log_det: Tensor of shape (batch_size,).
        """
        x_or_z = tf.convert_to_tensor(x_or_z, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            if x_or_z.shape[0] != condition.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )
        return self._call_log_det(x_or_z, condition, inverse)

    def _call_log_det(self, x, condition, inverse):
        """Wrapper around _log_det."""
        if hasattr(self, "_log_det"):
            return self._log_det(x, condition, inverse)
        if hasattr(self, "_det"):
            return tf.math.log(self._det(x, condition, inverse))
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide log_det(...) method"
        )

    def det(self, x_or_z, condition=None, inverse=False):
        """Calculates the jacobian determinant of the mapping:

        ...math::
            det = |J|, with
            J = dz/dx = df(x)/dx

        or for the inverse mapping:

        ...math::
            det = |J_inv|, with
            J_inv = dx/dz = df^{-1}(z)/dz

        Args:
            x_or_z: Tensor with shape (batch_size, n_features).
            condition (optional): None or Tensor with shape (batch_size, n_features).
                If None, the condition is ignored. Defaults to None.
            inverse (optional): bool, whether to return the log_det of
                inverse or forward mapping. Defaults to False.

        Returns:
            det: Tensor of shape (batch_size,).
        """
        x = tf.convert_to_tensor(x_or_z, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            if x.shape[0] != condition.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )
        return self._call_det(x_or_z, condition, inverse)

    def _call_det(self, x, condition, inverse):
        """Wrapper around _prob."""
        if hasattr(self, "_det"):
            return self._det(x, condition, inverse)
        if hasattr(self, "_log_det"):
            return tf.math.exp(self._log_det(x, condition, inverse))
        raise NotImplementedError("det is not implemented")

    def sample(self, num_samples, condition=None, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.
        Args:
            num_samples: int, number of samples to generate.
            condition: Tensor or None, conditioning variables of shape (condition_size, ...).
                If None, the condition is ignored.
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.
        Returns:
            samples: Tensor with shape (num_samples, ...) if condition is None, or
            (condition_size, num_samples, ...) if condition is given.
        """
        if not typechecks.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if condition is not None:
            condition = tf.convert_to_tensor(condition)

        if batch_size is None:
            return self._sample(num_samples, condition)

        else:
            if not typechecks.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, condition) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, condition))

            # Might not be needed in most cases
            if condition is not None:
                return tf.concat(samples, axis=1)

            return tf.concat(samples, axis=0)

    def _sample(self, num_samples, condition):
        raise NotImplementedError()

    def sample_and_log_det(self, num_samples, condition=None):
        """Generates samples from the distribution together with with their log probability.

        Args:
            num_samples: int, number of samples to generate.
            condition: Tensor or None, conditioning variables of shape (condition_size, ...).
                If None, the condition is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape (num_samples, ...) if condition is None,
                  or (condition_size, num_samples, ...) if condition is given.
                * A Tensor containing the log probabilities of the samples, with shape
                  (num_samples,) if condition is None, or (condition_size, num_samples) if
                  condition is given.
        """
        samples = self.sample(num_samples, condition)

        if condition is not None:
            # Merge the condition dimension with sample dimension in order to call log_prob.
            samples = tfutils.merge_leading_dims(samples, num_dims=2)
            condition = tfutils.repeat_rows(condition, num_reps=num_samples)
            assert samples.shape[0] == condition.shape[0]

        log_det = self.log_det(samples, condition)

        if condition is not None:
            # Split the context dimension from sample dimension.
            samples = tfutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_det = tfutils.split_leading_dim(log_det, shape=[-1, num_samples])

        return samples, log_det
