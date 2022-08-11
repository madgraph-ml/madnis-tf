"""Implementation of distribution class."""

import tensorflow as tf
from typing import Tuple

from ..utils import typechecks
from ..utils import tfutils


class Distribution(tf.keras.Model):
    """Base class for all distribution objects."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define the right floating point precision
        self._dtype = tf.keras.backend.floatx()

    def call(self, *args, **kwargs):
        "Call method just calls the prob function"
        return self.prob(*args, **kwargs)

    def log_prob(self, x: tf.Tensor, condition: tf.Tensor = None) -> tf.Tensor:
        """Calculate log probability of the distribution.

        Args:
            x: Tensor, shape (batch_size, ...).
            condition (optional): None or Tensor, shape (batch_size, ...).
                Must have the same number or rows as input.
                If None, the condition is ignored.

        Returns:
            log_prob: Tensor of shape (batch_size,), the log probability of the inputs.
        """
        x = tf.convert_to_tensor(x, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            if x.shape[0] != condition.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )
        return self._call_log_prob(x, condition)

    def _call_log_prob(self, x, condition):
        """Wrapper around _log_prob."""
        if hasattr(self, "_log_prob"):
            return self._log_prob(x, condition)
        if hasattr(self, "_prob"):
            return tf.math.log(self._prob(x, condition))
        raise NotImplementedError("log_prob is not implemented")

    def prob(self, x: tf.Tensor, condition: tf.Tensor = None) -> tf.Tensor:
        """Calculate probability of the distribution.

        Args:
            x: Tensor, shape (batch_size, ...).
            condition: None or Tensor, shape (batch_size, ...).
                Must have the same number or rows as input.
                If None, the condition is ignored.

        Returns:
            prob: Tensor of shape (batch_size,), the probability of the inputs.
        """
        x = tf.convert_to_tensor(x, dtype=self._dtype)
        if condition is not None:
            condition = tf.convert_to_tensor(condition, dtype=self._dtype)
            if x.shape[0] != condition.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )
        return self._call_prob(x, condition)

    def _call_prob(self, x, condition):
        """Wrapper around _prob."""
        if hasattr(self, "_prob"):
            return self._prob(x, condition)
        if hasattr(self, "_log_prob"):
            return tf.math.exp(self._log_prob(x, condition))
        raise NotImplementedError("prob is not implemented")

    def sample(
        self,
        num_samples: int,
        condition: tf.Tensor = None,
        batch_size: int = None,
    ) -> tf.Tensor:
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
        raise NotImplementedError("sampling is not implemented")

    def sample_and_log_prob(
        self,
        num_samples: int,
        condition: tf.Tensor = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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

        log_prob = self.log_prob(samples, condition)

        if condition is not None:
            # Split the context dimension from sample dimension.
            samples = tfutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = tfutils.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob

    def sample_and_prob(
        self,
        num_samples: int,
        condition: tf.Tensor = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generates samples from the distribution together with with their probability.

        Args:
            num_samples: int, number of samples to generate.
            condition: Tensor or None, conditioning variables of shape (condition_size, ...).
                If None, the condition is ignored.

        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape (num_samples, ...) if condition is None,
                  or (condition_size, num_samples, ...) if condition is given.
                * A Tensor containing the probabilities of the samples, with shape
                  (num_samples,) if condition is None, or (condition_size, num_samples) if
                  condition is given.
        """
        samples = self.sample(num_samples, condition)

        if condition is not None:
            # Merge the condition dimension with sample dimension in order to call log_prob.
            samples = tfutils.merge_leading_dims(samples, num_dims=2)
            condition = tfutils.repeat_rows(condition, num_reps=num_samples)
            assert samples.shape[0] == condition.shape[0]

        prob = self.prob(samples, condition)

        if condition is not None:
            # Split the context dimension from sample dimension.
            samples = tfutils.split_leading_dim(samples, shape=[-1, num_samples])
            prob = tfutils.split_leading_dim(prob, shape=[-1, num_samples])

        return samples, prob
