"""Implementations of Base mapping."""

import tensorflow as tf
from typing import Any, Callable, Tuple

from ..distributions.base import Distribution


class Mapping(Distribution):
    """Base class for all mapping objects.

    In contrast to normal distributions, mappings have
    forward and inverse passes and their log probabilities
    are dependent on the direction of the pass.
    """

    def __init__(self, base_dist: Distribution, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(base_dist, Distribution)
        self.base_dist = base_dist

    def call(
        self,
        x: tf.Tensor,
        condition: tf.Tensor = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass of the mapping ``f``.
        Conventionally, this is the pass from the
        momenta/data ``x`` to the latent space ``z``, i.e.

        ..math::
            f(x) = z.

        Args:
            x: Tensor with shape (batch_size, n_features).
            condition: None or Tensor with shape (batch_size, n_features).
                If None, the condition is ignored. Defaults to None.

        Returns:
            z: Tensor with shape (batch_size, n_features).
            logdet: Tensor of shape (batch_size,), the logdet of the mapping.
        """
        x = tf.convert_to_tensor(x, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            tf.debugging.assert_equal(
                tf.shape(x)[0], tf.shape(condition)[0],
                "Number of input items must be equal to number of condition items."
            )
        return self._forward(x, condition)

    def _forward(self, x, condition):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _forward(...) method"
        )

    def inverse(
        self,
        z: tf.Tensor,
        condition: tf.Tensor = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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
            logdet: Tensor of shape (batch_size,), the logdet of the mapping.
        """
        z = tf.convert_to_tensor(z, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            if tf.shape(z)[0] != tf.shape(condition)[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )

        return self._inverse(z, condition)

    def _inverse(self, z, condition):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _inverse(...) method"
        )

    def log_prob(
        self,
        x_or_z: tf.Tensor,
        condition: tf.Tensor = None,
        inverse: bool = False,
    ) -> tf.Tensor:
        """
        Calculate log probability of the mapping combined
        with the base distribution.

        **Remark:**
        Note the different relative signs between base_log_prob
        and log_det depending on the direction of the mapping.

        Args:
            x: Tensor, shape (batch_size, ...).
            condition (optional): None or Tensor, shape (batch_size, ...).
                Must have the same number or rows as input.
                If None, the condition is ignored.
            inverse (bool, optional): return log_prob of inverse pass. Defaults to False.

        Returns:
            log_prob: Tensor of shape (batch_size,), the log probability of the inputs.
        """
        if inverse:
            base_log_prob = self.base_dist.log_prob(x_or_z, condition)
            _, logdet = self.inverse(x_or_z, condition)
            return base_log_prob - logdet
        else:
            noise, logdet = self.call(x_or_z, condition)
            base_log_prob = self.base_dist.log_prob(noise, condition)
            return base_log_prob + logdet

    def prob(
        self,
        x_or_z: tf.Tensor,
        condition: tf.Tensor = None,
        inverse: bool = False,
    ) -> tf.Tensor:
        """Calculate full probability of the mapping"""
        return tf.math.exp(self.log_prob(x_or_z, condition, inverse))

    def log_det(
        self,
        x_or_z: tf.Tensor,
        condition: tf.Tensor = None,
        inverse: bool = False,
    ) -> tf.Tensor:
        """Calculate log det of the mapping only:

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
            inverse (bool, optional): return logdet of inverse pass. Defaults to False.

        Returns:
            log_det: Tensor of shape (batch_size,).
        """
        x_or_z = tf.convert_to_tensor(x_or_z, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            if tf.shape(x_or_z)[0] != tf.shape(condition)[0]:
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

    def det(
        self,
        x_or_z: tf.Tensor,
        condition: tf.Tensor = None,
        inverse: bool = False,
    ) -> tf.Tensor:
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
            inverse (bool, optional): return det of inverse pass. Defaults to False.

        Returns:
            det: Tensor of shape (batch_size,).
        """
        x = tf.convert_to_tensor(x_or_z, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            if tf.shape(x)[0] != tf.shape(condition)[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )
        return self._call_det(x_or_z, condition, inverse)

    def _call_det(self, x, condition, inverse):
        """Wrapper around _det."""
        if hasattr(self, "_det"):
            return self._det(x, condition, inverse)
        if hasattr(self, "_log_det"):
            return tf.math.exp(self._log_det(x, condition, inverse))
        raise NotImplementedError("det is not implemented")
