""" Implement the multi-channel flow integrator. """


import numpy as np
import tensorflow as tf
from typing import List, Union, Callable

from ..utils.divergences import Divergence
from ..mappings.flow import Flow
from ..distributions.base import Distribution

_EPSILON = 1e-16


class MultiChannelIntegrator:
    """Class implementing a conditional normalizing flow
    multi-channel integrator.
    """

    def __init__(
        self,
        func: Union[Callable, Distribution],
        flow: Flow,
        mcw_model: tf.keras.Model,
        optimizer: List[tf.keras.optimizers.Optimizer],
        use_weight_init: bool = True,
        n_channels: int = 2,
        loss_func: str = "chi2",
        **kwargs,
    ):
        """
        Args:
            func (Union[Callable, Distribution]):
                Function to be integrated
            flow (Flow):
                Trainable flow model to match the function
            mcw_model (tf.keras.Model):
                Model which learns the multi-channel weights
            optimizer (List[tf.keras.optimizers.Optimizer]):
                A list of optimizers for each of the two models
            n_channels (int): number of channels. Defaults to 2.
            loss_func (str, optional):
                The loss function to be minimized. Defaults to "chi2".
            kwargs: Additional arguments that need to be passed to the loss
        """
        self._dtype = tf.keras.backend.floatx()

        self._func = func
        self.flow = flow
        self.mcw_model = mcw_model
        self.flow_optimizer = optimizer[0]
        self.mcw_optimizer = optimizer[1]

        self.use_weight_init = use_weight_init
        
        if n_channels > 1:
            self.n_channels = n_channels
        else:
            raise ValueError(f"More than 1 channel expected. Use Integrator instead.")

        # Define the loss functions
        self.flow_divergence = Divergence(**kwargs)
        self.flow_loss_func = self.flow_divergence(loss_func)

        self.mcw_divergence = Divergence(train_mcw=True, **kwargs)
        self.mcw_loss_func = self.mcw_divergence(loss_func)

    def _get_channel_condition(self, nsamples: int):
        # creates ones with shape (nsamples, nc)
        cond = tf.ones((nsamples, self.n_channels), dtype=self._dtype)

        # creates shape (b, nc, nc) with b unit-matrices (nc x nc)
        c_cond = tf.linalg.diag(cond)
        return c_cond

    @tf.function
    def _get_samples(self, nsamples: int, one_hot_channels: tf.Tensor):
        xs = []
        fs = []
        qs = []

        for i in range(self.n_channels):
            # Channel dependent flow sampling
            xi, qi = self.flow.sample_and_prob(nsamples, condition=one_hot_channels[:, :, i])
            xs.append(xi)
            qs.append(qi)
            fs.append(self._func(xi))

        # Get concatenated stuff all in shape (nsamples, n_channels)
        return (
            tf.stack(xs, axis=-1),
            tf.stack(qs, axis=-1),
            tf.stack(fs, axis=-1)
        )

    @tf.function
    def _get_probs(self, samples: tf.Tensor, func_vals: tf.Tensor, one_hot_channels: tf.Tensor):
        ps = []
        qs = []
        logqs = []
        means = []
        vars = []
        for i in range(self.n_channels):
            samples_i = samples[:, :, i]

            # Flow density estimation
            logqi = self.flow.log_prob(samples_i, condition=one_hot_channels[:, :, i])
            qi = tf.math.exp(logqi)
            logqs.append(logqi)
            qs.append(qi)

            # Get multi-channel weights
            if self.use_weight_init:
                init_weights = (
                    1 / self.n_channels
                    * tf.ones((len(nsamples), self.n_channels), dtype=self._dtype)
                )
                alphas = self.mcw_model([samples_i, init_weights])
            else:
                alphas = self.mcw_model(samples_i)

            # Get true integrand
            pi = alphas[:, i] * tf.abs(func_vals[:, i])
            meani, vari = tf.nn.moments(pi / qi, axes=[0])
            pi = pi / meani
            ps.append(pi)
            means.append(meani)
            vars.append(vari)

        # Get concatenated stuff all in shape (nsamples, n_channels)
        q_test = tf.stack(qs, axis=-1)
        logq = tf.stack(logqs, axis=-1)
        p_true = tf.stack(ps, axis=-1)
        logp = tf.where(
            p_true > _EPSILON, tf.math.log(p_true), tf.math.log(p_true + _EPSILON)
        )
        
        return p_true, q_test, logp, logq, sum(means), sum(vars)

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
        loss = 0

        # Sample from flow
        one_hot_channels = self._get_channel_condition(nsamples)
        samples, q_sample, func_vals = self._get_samples(nsamples, one_hot_channels)
        
        # Optimize the Flow
        with tf.GradientTape() as tape:
            p_true, q_test, logp, logq, mean, var = self._get_probs(
                samples, func_vals, one_hot_channels
            )
            flow_loss = self.flow_loss_func(p_true, q_test, logp, logq, q_sample=q_sample)

        grads = tape.gradient(flow_loss, self.flow.trainable_weights)
        self.flow_optimizer.apply_gradients(zip(grads, self.flow.trainable_weights))
        loss += flow_loss
        
        # Optimize the channel weight
        with tf.GradientTape() as tape:
            p_true, q_test, logp, logq, mean, var = self._get_probs(
                samples, func_vals, one_hot_channels
            )
            mcw_loss = self.mcw_loss_func(p_true, q_test, logp, logq, q_sample=q_sample)

        grads = tape.gradient(mcw_loss, self.mcw_model.trainable_weights)
        self.mcw_optimizer.apply_gradients(zip(grads, self.mcw_model.trainable_weights))
        loss = 0.5 * (loss + mcw_loss)

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
