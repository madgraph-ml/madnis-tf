""" Implement the flow integrator. """


import numpy as np
import tensorflow as tf
from typing import Tuple, Union, Callable

from ..utils.divergences import Divergence
from ..mappings.flow import Flow
from ..distributions.base import Distribution

_EPSILON = 1e-16


class Integrator:
    """Class implementing a normalizing flow integrator."""

    def __init__(
        self,
        func: Union[Callable, Distribution],
        flow: Flow,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_func: Union[str, Callable] = "chi2",
        **kwargs,
    ):
        """
        Args:
            func (Union[Callable, Distribution]):
                Function to be integrated
            flow (Flow):
                Trainable flow model to match the function
            optimizer (tf.keras.optimizers.Optimizer):
                An optimizer to train the flow
            loss_func (Union[str, Callable], optional):
                The loss function to be minimized. Defaults to "chi2".
            kwargs: Additional arguments that need to be passed to the loss
        """

        self._func = func
        self.flow = flow
        self.optimizer = optimizer

        if isinstance(loss_func, str):
            self.divergence = Divergence(**kwargs)
            self.loss_func = self.divergence(loss_func)
        elif isinstance(loss_func, Callable):
            self.loss_func = loss_func
        else:
            raise ValueError(f"Loss function needs to be a string or a callable")

    @tf.function
    def train_one_step(self, nsamples: int, integral: bool = False):
        """Perform one step of integration and improve the sampling.

        Args:
            nsamples (int): Number of samples to be taken in a training step
            integral (bool, optional): return the integral value. Defaults to False.

        Returns:
            loss: Value of the loss function for this step
            integral (optional): Estimate of the integral value
            uncertainty (optional): Integral statistical uncertainty

        Args:
            nsamples (int): _description_
            integral (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        samples = self.flow.sample(nsamples)
        true = tf.abs(self._func(samples))
        with tf.GradientTape() as tape:
            q_test = self.flow.prob(samples)
            logq = self.flow.log_prob(samples)
            mean, var = tf.nn.moments(true / q_test, axes=[0])
            p_true = tf.stop_gradient(true / mean)
            logp = tf.where(
                p_true > _EPSILON, tf.math.log(p_true), tf.math.log(p_true + _EPSILON)
            )
            loss = self.loss_func(p_true, q_test, logp, logq)

        grads = tape.gradient(loss, self.flow.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.flow.trainable_weights))

        if integral:
            return loss, mean, tf.sqrt(var / (nsamples - 1.0))

        return loss

    @tf.function
    def sample(self, nsamples: int):
        """Sample from the trained distribution.

        Args:
            nsamples (int): Number of points to be sampled.

        Returns:
            tf.tensor of size (nsamples, ndim) of sampled points.

        """
        return self.flow.sample(nsamples)

    @tf.function
    def integrate(self, nsamples: int):
        """Integrate the function with trained distribution.

        This method estimates the value of the integral based on
        Monte Carlo importance sampling. It returns a tuple of two
        tf.tensors. The first one is the mean, i.e. the estimate of
        the integral. The second one gives the variance of the integrand.
        To get the variance of the estimated mean, the returned variance
        needs to be divided by (nsamples -1).

        Args:
            nsamples (int): Number of points on which the estimate is based on.

        Returns:
            tuple of 2 tf.tensors: mean and mc error

        """
        samples = self.flow.sample(nsamples)
        q_test = self.flow.prob(samples)
        true = self._func(samples)
        mean, var = tf.nn.moments(true / q_test, axes=[0])
        return mean, tf.sqrt(var / (nsamples - 1.0))

    @tf.function
    def sample_weights(self, nsamples: int, yield_samples: bool = False):
        """Sample from the trained distribution and return their weights.

        This method samples 'nsamples' points from the trained distribution
        and computes their weights, defined as the functional value of the
        point divided by the probability of the trained distribution of
        that point.

        Optionally, the drawn samples can be returned, too.

        Args:
            nsamples (int): Number of samples to be drawn.
            yield_samples (bool, optional): return samples. Defaults to False.

        Returns:
            true/test: tf.tensor of size (nsamples, 1) of sampled weights
            (samples: tf.tensor of size (nsamples, ndims) of sampled points)

        """
        samples = self.flow.sample(nsamples)
        test = self.flow.prob(samples)
        true = self._func(samples)
        weight = true / test

        if yield_samples:
            return weight[..., None], samples

        return weight[..., None]

    def acceptance(self, nopt: int, npool: int = 50, nreplica: int = 1000):
        """Calculate the acceptance, i.e. the unweighting
            efficiency as discussed in arXiv:2001.10028 [hep-ph]

        Args:
            nopt (int): Number of points on which the optimization was based on.
            npool (int, optional): called n in the reference. Defaults to 50.
            nreplica (int, optional): called m in the reference. Defaults to 1000.

        Returns:
            (float): unweighting efficiency

        """

        weights = []
        for _ in range(npool):
            wgt = self.sample_weights(nopt)
            weights.append(wgt)
        weights = np.concatenate(weights)

        sample = np.random.choice(weights, (nreplica, nopt))
        s_max = np.max(sample, axis=1)
        s_mean = np.mean(sample, axis=1)
        s_acc = np.mean(s_mean) / np.median(s_max)

        return s_acc

    def save_weights(self, path: str):
        """Save the network."""
        self.flow.save_weights(path)

    def load_weights(self, path: str):
        """Load the network."""
        self.flow.load_weights(path)
        print("Model loaded successfully")
