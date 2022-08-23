"""Basic definition for the flow module."""

from typing import Iterable, List, Union
import tensorflow as tf

from ..distributions.base import Distribution
from ..mappings.base import Mapping
from ..transforms.base import Transform
from ..utils import tfutils


class Flow(Mapping):
    def __init__(
        self,
        base_dist: Distribution,
        transforms: Union[Transform, List[Transform]],
        embedding_net: tf.keras.Model = None,
        **kwargs
    ):
        super().__init__(base_dist, **kwargs)

        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform):
            transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.transforms = transforms

        if embedding_net is not None:
            assert isinstance(embedding_net, tf.keras.Model), (
                "embedding_net is not a keras.Model "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self.embedding_net = embedding_net
        else:
            self.embedding_net = lambda x: x

    def _forward(self, x: tf.Tensor, condition: tf.Tensor = None):
        log_det = 0
        for transform in self.transforms:
            x, ljd = transform(x, c=[condition])
            log_det += ljd
        return x, log_det

    def _inverse(self, z: tf.Tensor, condition: tf.Tensor = None):
        log_det = 0
        for transform in reversed(self.transforms):
            z, ljd = transform.inverse(z, c=[condition])
            log_det += ljd
        return z, log_det

    def _log_det(
        self,
        x_or_z: tf.Tensor,
        condition: tf.Tensor = None,
        inverse: bool = False,
    ):
        embedded_condition = self.embedding_net(condition)
        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            _, logdet = self._inverse(x_or_z, embedded_condition)
            return logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            _, logdet = self._forward(x_or_z, embedded_condition)
            return logdet

    def _sample(self, num_samples: int, condition: tf.Tensor = None):
        """Generates samples from the flow."""

        embedded_condition = self.embedding_net(condition)
        z = self.base_dist.sample(num_samples, condition=embedded_condition)

        if embedded_condition is not None:
            z = tfutils.merge_leading_dims(z, num_dims=2)
            embedded_condition = tfutils.repeat_rows(
                embedded_condition, num_reps=num_samples
            )

        sample, _ = self._inverse(z, embedded_condition)

        if embedded_condition is not None:
            # Split the context dimension from sample dimension.
            sample = tfutils.split_leading_dim(sample, shape=[-1, num_samples])

        return sample

    def log_prob(
        self,
        x_or_z: tf.Tensor,
        condition: tf.Tensor = None,
        inverse: bool = False,
    ):
        """
        Calculate log probability of the mapping combined
        with the flow distribution. We override the base class
        as we have to handle the embedding of the condition, i.e
        ``condition -> embedded_condition``.

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
        embedded_condition = self.embedding_net(condition)
        if inverse:
            base_log_prob = self.base_dist.log_prob(x_or_z, embedded_condition)
            _, logabsdet = self._inverse(x_or_z, embedded_condition)
            return base_log_prob - logabsdet
        else:
            noise, logabsdet = self._forward(x_or_z, embedded_condition)
            base_log_prob = self.base_dist.log_prob(noise, embedded_condition)
            return base_log_prob + logabsdet

    def sample_and_log_prob(self, num_samples, condition=None):
        """Generates samples from the flow, together with their log probabilities.
        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """

        embedded_condition = self.embedding_net(condition)
        z, log_prob = self.base_dist.sample_and_log_prob(
            num_samples, condition=embedded_condition
        )

        if embedded_condition is not None:
            # Merge the condition dimension with sample dimension in order to call log_prob.
            z = tfutils.merge_leading_dims(z, num_dims=2)
            embedded_condition = tfutils.repeat_rows(
                embedded_condition, num_reps=num_samples
            )
            assert z.shape[0] == embedded_condition.shape[0]

        samples, logabsdet = self._inverse(z, embedded_condition)

        if embedded_condition is not None:
            # Split the context dimension from sample dimension.
            samples = tfutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = tfutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def sample_and_prob(self, num_samples, condition=None):
        """Generates samples from the flow, together with their probabilities.
        For flows, this is more efficient that calling `sample` and `prob` separately.
        """

        embedded_condition = self.embedding_net(condition)
        z, log_prob = self.base_dist.sample_and_log_prob(
            num_samples, condition=embedded_condition
        )

        if embedded_condition is not None:
            # Merge the condition dimension with sample dimension in order to call log_prob.
            z = tfutils.merge_leading_dims(z, num_dims=2)
            embedded_condition = tfutils.repeat_rows(
                embedded_condition, num_reps=num_samples
            )
            assert z.shape[0] == embedded_condition.shape[0]

        samples, logabsdet = self._inverse(z, embedded_condition)

        if embedded_condition is not None:
            # Split the context dimension from sample dimension.
            samples = tfutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = tfutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, tf.math.exp(log_prob - logabsdet)
