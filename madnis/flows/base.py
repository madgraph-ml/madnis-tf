"""Basic definition for the flow module."""

from typing import Iterable, List, Union, Callable
import tensorflow as tf

from ..distributions.base import Distribution
from ..transforms.base import Transform
from ..utils import tfutils
from ..utils import typechecks


class Flow(tf.keras.Model):
    def __init__(
        self,
        base_dist: Distribution,
        transforms: Union[Transform, List[Transform]],
        embedding_net: tf.keras.Model = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform):
            transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = transforms
        if embedding_net is not None:
            assert isinstance(embedding_net, tf.keras.Model), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self.embedding_net = embedding_net
        else:
            self.embedding_net = lambda x: x

    def call(self, x: tf.Tensor, condition: tf.Tensor = None):
        """
        Forward pass of the flow ``f``.
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
        """
        z = x
        log_det = 0
        for transform in self.transforms:
            z, ljd = transform(z, c=[condition])
            log_det += ljd

        return z, log_det

    def inverse(self, z: tf.Tensor, condition: tf.Tensor = None):
        """
        Inverse pass ``f^{-1}`` of the flow. Conventionally, this is the pass
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
        x = z
        log_det = 0
        for transform in reversed(self.transforms):
            x, ljd = transform.inverse(x, c=[condition])
            log_det += ljd
        return x, log_det

    def log_prob(
        self,
        x_or_z: tf.Tensor,
        condition: tf.Tensor = None,
        inverse: bool = False,
    ):
        """
        Calculate log probability of the flow including the base distribution.

        Remark: Note the different relative signs between base_log_prob
        and log_det depending on the direction of the mapping.

        Args:
            x_or_z (tf.Tensor): Input, latent space (z) or data (x).
            condition (tf.Tensor, optional): conditioning variables of shape (condition_size, ...).
                If None, the condition is ignored. Defaults to None.
            inverse (bool, optional): perform inverse pass. Defaults to False.

        Returns:
            log_prob: Tensor of shape (batch_size,), the log probability of the inputs.
        """

        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            base_log_prob = self.base_dist.log_prob(x_or_z, condition)
            _, logdet = self.inverse(x_or_z, condition)
            return base_log_prob - logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            z, logdet = self.call(x_or_z, condition)
            base_log_prob = self.base_dist.log_prob(z, condition)
            return base_log_prob + logdet

    def prob(
        self,
        x_or_z: tf.Tensor,
        condition: tf.Tensor = None,
        inverse: bool = False,
    ):
        """Calculate probability of the flow"""
        return tf.math.exp(self.log_prob(x_or_z, condition, inverse))

    def sample(self, num_samples: int, condition: tf.Tensor = None):
        """Generates samples from the flow.
        Args:
            num_samples: int, number of samples to generate.
            condition: Tensor or None, conditioning variables of shape (condition_size, ...).
                If None, the condition is ignored.
        Returns:
            samples: Tensor with shape (num_samples, ...) if condition is None, or
            (condition_size, num_samples, ...) if condition is given.
        """
        if not typechecks.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if condition is not None:
            condition = tf.convert_to_tensor(condition)

        embedded_condition = self.embedding_net(condition)
        z = self.base_dist.sample(num_samples, condition=embedded_condition)

        if embedded_condition is not None:
            z = tfutils.merge_leading_dims(z, num_dims=2)
            embedded_condition = tfutils.repeat_rows(
                embedded_condition, num_reps=num_samples
            )

        for transform in reversed(self.transforms):
            z, _ = transform.inverse(z, c=embedded_condition)

        if embedded_condition is not None:
            # Split the context dimension from sample dimension.
            z = tfutils.split_leading_dim(z, shape=[-1, num_samples])

        return z

    def sample_and_log_prob(self, num_samples: int, condition: tf.Tensor = None):
        """Generates samples from the flow together with with their log probability.

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

        log_prob = self.log_prob(samples, condition, inverse=True)

        if condition is not None:
            # Split the context dimension from sample dimension.
            samples = tfutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = tfutils.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob
