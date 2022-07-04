"""Basic definitions for the flows module."""


import tensorflow as tf

from ..distributions.base import Distribution
from ..utils import tfutils


class Flow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution: Distribution, embedding_net=None):
        """Constructor.
        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `tf.keras.Model` which has trainable parameters to encode the
                condition. It is trained jointly with the flow.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        if embedding_net is not None:
            assert isinstance(embedding_net, tf.keras.Model), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self._embedding_net = embedding_net
        else:
            self._embedding_net = lambda x: x

    def _log_prob(self, inputs, condition):
        embedded_condition = self._embedding_net(condition)
        noise, logabsdet = self._transform(inputs, context=embedded_condition)
        log_prob = self._distribution.log_prob(noise, context=embedded_condition)
        return log_prob + logabsdet

    def _sample(self, num_samples, condition):
        embedded_condition = self._embedding_net(condition)
        noise = self._distribution.sample(num_samples, context=embedded_condition)

        if embedded_condition is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = tfutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = tfutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = tfutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, condition=None):
        """Generates samples from the flow, together with their log probabilities.
        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_condition = self._embedding_net(condition)
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=embedded_condition
        )

        if embedded_condition is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = tfutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = tfutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_condition is not None:
            # Split the context dimension from sample dimension.
            samples = tfutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = tfutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, condition=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.
        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.
        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(condition))
        return noise
