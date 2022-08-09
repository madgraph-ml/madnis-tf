"""Basic definition for the flow module."""

from typing import Iterable, List, Union, Callable
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
        embedding_net: Callable = None,
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
            assert isinstance(embedding_net, Callable), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self.embedding_net = embedding_net
        else:
            self.embedding_net = lambda x: x

    def _forward(self, x: tf.Tensor, condition: tf.Tensor = None):
        log_det = tf.zeros(x.shape[0], dtype=self._dtype)
        for transform in self.transforms:
            x, ljd = transform(x, c=[condition])
            log_det += ljd
        return x, log_det

    def _inverse(self, z: tf.Tensor, condition: tf.Tensor = None):
        log_det = tf.zeros(z.shape[0], dtype=self._dtype)
        for transform in reversed(self.transforms):
            z, ljd = transform.inverse(z, c=[condition])
            log_det += ljd
        return z, log_det
    
    def _log_det(self,
        x_or_z: tf.Tensor,
        condition: tf.Tensor = None,
        inverse: bool = False
    ):
        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            _, logdet = self.inverse(x_or_z, condition)
            return logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            _, logdet = self.call(x_or_z, condition)
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
        
        sample, _ = self.inverse(z, condition=embedded_condition)

        if embedded_condition is not None:
            # Split the context dimension from sample dimension.
            sample = tfutils.split_leading_dim(sample, shape=[-1, num_samples])

        return sample
    
    def sample_and_log_prob(self, num_samples, condition=None):
        """Generates samples from the flow, together with their log probabilities.
        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        
        embedded_condition = self.embedding_net(condition)
        z, log_prob = self.base_dist.sample_and_log_prob(num_samples, condition=embedded_condition)

        if embedded_condition is not None:
            # Merge the condition dimension with sample dimension in order to call log_prob.
            z = tfutils.merge_leading_dims(z, num_dims=2)
            embedded_condition = tfutils.repeat_rows(embedded_condition, num_reps=num_samples)
            assert z.shape[0] == embedded_condition.shape[0]
            
        samples, logabsdet = self.inverse(z, condition=embedded_condition)

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
        z, log_prob = self.base_dist.sample_and_log_prob(num_samples, condition=embedded_condition)

        if embedded_condition is not None:
            # Merge the condition dimension with sample dimension in order to call log_prob.
            z = tfutils.merge_leading_dims(z, num_dims=2)
            embedded_condition = tfutils.repeat_rows(embedded_condition, num_reps=num_samples)
            assert z.shape[0] == embedded_condition.shape[0]
            
        samples, logabsdet = self.inverse(z, condition=embedded_condition)

        if embedded_condition is not None:
            # Split the context dimension from sample dimension.
            samples = tfutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = tfutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, tf.math.exp(log_prob - logabsdet)
