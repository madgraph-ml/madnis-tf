""" Implement the multi-channel flow integrator. """


import numpy as np
import tensorflow as tf
from typing import List, Union, Callable

from ..utils.divergences import Divergence
from ..mappings.flow import Flow
from ..mappings.base import Mapping
from ..distributions.base import Distribution

_EPSILON = 1e-16


class MultiChannelIntegrator:
    """Class implementing a conditional normalizing flow
    multi-channel integrator.
    """

    def __init__(
        self,
        func: Union[Callable, Distribution],
        dist: Distribution,
        optimizer: List[tf.keras.optimizers.Optimizer],
        mcw_model: tf.keras.Model = None,
        mappings: List[Mapping] = None,
        use_weight_init: bool = True,
        n_channels: int = 2,
        loss_func: str = "chi2",
        sample_capacity: int = 0,
        uniform_channel_ratio: float = 1.0,
        variance_history_length: int = 20,
        **kwargs,
    ):
        """
        Args:
            func (Union[Callable, Distribution]):
                Function to be integrated
            dist (Distribution):
                Trainable flow distribution to match the function
                or a fixed base distribution.
            optimizer (List[tf.keras.optimizers.Optimizer]):
                A list of optimizers for each of the two models
            mcw_model (tf.keras.Model, optional):
                Model which learns the multi-channel weights. Defaults to None.
            mappings (List(Mapping), optional): A list of analytic mappings applied
                to each channel of integration. Defaults to None.
            n_channels (int, optional): number of channels. Defaults to 2.
            loss_func (str, optional):
                The loss function to be minimized. Defaults to "chi2".
            sample_capacity (int, optional):
                Number of samples to be stored by the integrator for later reuse.
                Defaults to 0.
            uniform_channel_ratio (float, optional):
                Part of samples in each batch that will be distributed equally between
                all channels. Between 0 and 1. Defaults to 1.
            variance_history_length (int, optional):
                How many of the previous batches to take into account to compute the variances
                for weighting the channels of the next batch
            kwargs: Additional arguments that need to be passed to the loss
        """
        self._dtype = tf.keras.backend.floatx()

        self._func = func
        self.dist = dist

        # Define flow or base mapping
        if isinstance(dist, Flow):
            self.train_flow = True
        else:
            self.train_flow = False

        # Define mcw model if given
        self.mcw_model = mcw_model
        if self.mcw_model is None:
            self.train_mcw = False
        else:
            self.train_mcw = True

        # Define optimizers
        if len(optimizer) > 1:
            assert self.mcw_model is not None
            assert isinstance(self.dist, Flow)
            self.flow_optimizer = optimizer[0]
            self.mcw_optimizer = optimizer[1]
        elif len(optimizer) == 1:
            if self.mcw_model is not None:
                self.flow_optimizer = None
                self.mcw_optimizer = optimizer[0]
            else:
                self.flow_optimizer = optimizer[0]
                self.mcw_optimizer = None
        else:
            raise ValueError(
                f"Number of given optimziers: {len(optimizer)}, must be either 2 or 1."
            )

        self.use_weight_init = use_weight_init

        self.n_channels = n_channels
        if self.n_channels < 2:
            self.train_mcw = False

        # Check and define analytic mappings
        self.mappings = mappings
        if self.mappings is not None:
            assert n_channels == len(self.mappings)
            self.use_analytic_mappings = True
        else:
            self.use_analytic_mappings = False

        # Define the loss functions
        self.flow_divergence = Divergence(**kwargs)
        self.flow_loss_func = self.flow_divergence(loss_func)

        self.mcw_divergence = Divergence(train_mcw=True, **kwargs)
        self.mcw_loss_func = self.mcw_divergence(loss_func)

        self.uniform_channel_ratio = uniform_channel_ratio
        self.variance_history_length = variance_history_length
        self.variance_history = []
        self.count_history = []

        self.sample_capacity = sample_capacity
        if sample_capacity > 0:
            self.stored_samples = []
            self.stored_q_sample = []
            self.stored_func_vals = []
            self.stored_channels = []
            self.stored_dataset = None

    def _store_samples(
        self,
        samples: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        channels: tf.Tensor,
    ):
        if self.sample_capacity == 0:
            return

        self.stored_samples.append(samples)
        self.stored_q_sample.append(q_sample)
        self.stored_func_vals.append(func_vals)
        self.stored_channels.append(channels)
        del self.stored_samples[: -self.sample_capacity]
        del self.stored_q_sample[: -self.sample_capacity]
        del self.stored_func_vals[: -self.sample_capacity]
        del self.stored_channels[: -self.sample_capacity]
        self.stored_dataset = None

    @tf.function
    def _compute_analytic_mappings(self, x, logq, channels):
        if not self.use_analytic_mappings:
            return x, logq

        xs = tf.dynamic_partition(x, channels, self.n_channels)
        idx = tf.dynamic_partition(tf.range(x.shape[0]), channels, self.n_channels)
        ys = []
        jacs = []
        for i, xi in enumerate(xs):
            yi, _ = self.mappings[i].inverse(xi)
            ys.append(yi)
            jacs.append(self.mappings[i].log_prob(yi))

        y = tf.dynamic_stitch(idx, ys)
        jac = tf.dynamic_stitch(idx, jacs)
        return y, logq + jac

    @tf.function
    def _get_samples(
        self, nsamples: int, channel_weights: tf.Tensor, uniform_channel_ratio: float
    ):
        assert channel_weights.shape == (self.n_channels,)
        # Split up nsamples * uniform_channel_ratio equally among all the channels
        n_uniform = int(nsamples * uniform_channel_ratio)
        uniform_channels = tf.tile(
            tf.range(self.n_channels), (n_uniform // self.n_channels + 1,)
        )[:n_uniform]
        # Sample the rest of the events from the distribution given by channel_weights
        # after correcting for the uniformly distributed samples
        normed_weights = channel_weights / tf.reduce_sum(channel_weights)
        probs = tf.maximum(
            normed_weights - uniform_channel_ratio / self.n_channels, 1e-15
        )
        sampled_channels = tf.random.categorical(
            tf.math.log(probs)[None, :], nsamples - n_uniform, dtype=tf.int32
        )[0]
        channels = tf.concat((uniform_channels, sampled_channels), axis=0)

        one_hot_channels = tf.one_hot(channels, self.n_channels, dtype=self._dtype)
        x, logq = self.dist.sample_and_log_prob(nsamples, condition=one_hot_channels)
        y, logq = self._compute_analytic_mappings(x, logq, channels)
        return x, tf.math.exp(logq), self._func(y), channels

    @tf.function
    def _get_probs(
        self,
        samples: tf.Tensor,
        func_vals: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable = None,
        return_integrand: bool = False,
    ):
        nsamples = samples.shape[0]
        one_hot_channels = tf.one_hot(channels, self.n_channels, dtype=self._dtype)
        logq = self.dist.log_prob(samples, condition=one_hot_channels)
        y, logq = self._compute_analytic_mappings(samples, logq, channels)
        q_test = tf.math.exp(logq)

        if self.train_mcw:
            if self.use_weight_init:
                if weight_prior is not None:
                    init_weights = weight_prior(y)
                    assert init_weights.shape[1] == self.n_channels
                else:
                    init_weights = (
                        1
                        / self.n_channels
                        * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                    )
                alphas = self.mcw_model([y, init_weights])
            else:
                alphas = self.mcw_model(y)
        else:
            if weight_prior is not None:
                alphas = weight_prior(y)
                assert alphas.shape[1] == self.n_channels
            else:
                alphas = (
                    1
                    / self.n_channels
                    * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                )
        alphas = tf.gather(alphas, channels, batch_dims=1)

        if return_integrand:
            return alphas * func_vals / q_test

        p_unnormed = alphas * tf.abs(func_vals)
        p_trues = []
        means = []
        vars = []
        counts = []
        ps = tf.dynamic_partition(p_unnormed, channels, self.n_channels)
        qs = tf.dynamic_partition(q_test, channels, self.n_channels)
        idx = tf.dynamic_partition(tf.range(nsamples), channels, self.n_channels)
        for pi, qi in zip(ps, qs):
            meani, vari = tf.nn.moments(pi / qi, axes=[0])
            p_trues.append(pi / meani)
            means.append(meani)
            vars.append(vari)
            counts.append(tf.cast(tf.shape(pi)[0], self._dtype))
        p_true = tf.dynamic_stitch(idx, p_trues)
        logp = tf.math.log(p_true + _EPSILON)

        return (
            p_true,
            q_test,
            logp,
            logq,
            tf.convert_to_tensor(means),
            tf.convert_to_tensor(vars),
            tf.convert_to_tensor(counts),
        )

    @tf.function
    def _optimization_step(
        self,
        samples: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable,
    ):
        loss = 0
        if not self.train_flow and not self.train_mcw:
            raise ValueError("No network defined which can be optimized")

        # Optimize the Flow
        if self.train_flow:
            with tf.GradientTape() as tape:
                p_true, q_test, logp, logq, means, vars, counts = self._get_probs(
                    samples, func_vals, channels, weight_prior
                )
                flow_loss = self.flow_loss_func(
                    p_true, q_test, logp, logq, q_sample=q_sample
                )

            grads = tape.gradient(flow_loss, self.dist.trainable_weights)
            self.flow_optimizer.apply_gradients(zip(grads, self.dist.trainable_weights))
            loss += flow_loss

        # Optimize the channel weight
        if self.train_mcw:
            with tf.GradientTape() as tape:
                p_true, q_test, logp, logq, means, vars, counts = self._get_probs(
                    samples, func_vals, channels
                )
                mcw_loss = self.mcw_loss_func(
                    p_true, q_test, logp, logq, q_sample=q_sample
                )

            grads = tape.gradient(mcw_loss, self.mcw_model.trainable_weights)
            self.mcw_optimizer.apply_gradients(
                zip(grads, self.mcw_model.trainable_weights)
            )
            loss += mcw_loss

        return loss, means, vars, counts

    def _get_variance_weights(self):
        if len(self.variance_history) < self.variance_history_length:
            return tf.fill((self.n_channels,), 1.0)

        count_hist = tf.convert_to_tensor(self.count_history, dtype=self._dtype)
        var_hist = tf.convert_to_tensor(self.variance_history, dtype=self._dtype)
        hist_weights = count_hist / tf.reduce_sum(count_hist)
        w = tf.reduce_sum(hist_weights * var_hist, axis=0)
        return w

    def delete_samples(self):
        """Delete all stored samples."""
        del self.stored_samples[:]
        del self.stored_q_sample[:]
        del self.stored_func_vals[:]
        del self.stored_channels[:]
        self.stored_dataset = None

    def train_one_step(
        self, nsamples: int, weight_prior: Callable = None, integral: bool = False
    ):
        """Perform one step of integration and improve the sampling.

        Args:
            nsamples (int): Number of samples to be taken in a training step
            weight_prior (Callable, optional): returns the prior weights. Defaults to None.
            integral (bool, optional): return the integral value. Defaults to False.

        Returns:
            loss: Value of the loss function for this step
            integral (optional): Estimate of the integral value
            uncertainty (optional): Integral statistical uncertainty
        """

        # Sample from flow
        samples, q_sample, func_vals, channels = self._get_samples(
            nsamples, self._get_variance_weights(), self.uniform_channel_ratio
        )
        self._store_samples(samples, q_sample, func_vals, channels)

        loss, means, vars, counts = self._optimization_step(
            samples, q_sample, func_vals, channels, weight_prior
        )
        self.variance_history.append(vars)
        self.count_history.append(counts)
        del self.variance_history[: -self.variance_history_length]
        del self.count_history[: -self.variance_history_length]

        if integral:
            return (
                loss,
                tf.reduce_sum(means),
                tf.sqrt(tf.reduce_sum(vars / (counts - 1.0))),
            )

        return loss

    def train_on_stored_samples(self, batch_size: int, weight_prior: Callable = None):
        """Train the network on all saved samples.

        Args:
            batch_size (int): batch size
            weight_prior (Callable, optional): returns the prior weights. Defaults to None.

        Returns:
            loss: Value of the loss function for this step
        """
        if self.stored_dataset is None:
            samples = tf.concat(self.stored_samples, axis=0)
            q_sample = tf.concat(self.stored_q_sample, axis=0)
            func_vals = tf.concat(self.stored_func_vals, axis=0)
            channels = tf.concat(self.stored_channels, axis=0)
            self.stored_dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (samples, q_sample, func_vals, channels)
                )
                .shuffle(samples.shape[0])
                .batch(batch_size, drop_remainder=True)
            )

        losses = []
        for ys, qs, fs, cs in self.stored_dataset:
            loss, _, _, _ = self._optimization_step(ys, qs, fs, cs, weight_prior)
            losses.append(loss)

        return tf.reduce_mean(losses)

    @tf.function
    def integrate(self, nsamples: int, weight_prior: Callable = None):
        """Integrate the function with trained distribution.

        This method estimates the value of the integral based on
        Monte Carlo importance sampling. It returns a tuple of two
        tf.tensors. The first one is the mean, i.e. the estimate of
        the integral. The second one gives the variance of the integrand.
        To get the variance of the estimated mean, the returned variance
        needs to be divided by (nsamples - 1).

        Args:
            nsamples (int): Number of points on which the estimate is based on.
            weight_prior (Callable, optional): returns the prior weights. Defaults to None.

        Returns:
            tuple of 2 tf.tensors: mean and mc error

        """
        samples, _, func_vals, channels = self._get_samples(
            nsamples, self._get_variance_weights(), uniform_channel_ratio=0.0
        )
        integrands = self._get_probs(
            samples, func_vals, channels, weight_prior, return_integrand=True
        )
        mean = 0.0
        var = 0.0
        integs = tf.dynamic_partition(integrands, channels, self.n_channels)
        for integ in integs:
            meani, vari = tf.nn.moments(integ, axes=[0])
            mean += meani
            var += vari / (tf.cast(tf.shape(integ)[0], self._dtype) - 1.0)
        return mean, tf.sqrt(var)

    @tf.function
    def sample_per_channel(
        self, nsamples: int, channel: int, weight_prior: Callable = None
    ):
        """Sample from the trained distribution and return their weights
        for a single channel.

        This method samples 'nsamples' points from the trained distribution
        and computes their weights, defined as the functional value of the
        point divided by the probability of the trained distribution of
        that point.

        Args:
            nsamples (int): Number of samples to be drawn.
            channel (int): the channel of the sampling.
            weight_prior (Callable, optional): returns the prior weights. Defaults to None.

        Returns:
            samples: tf.tensor of size (nsamples, ndims) of sampled points
            true/test: tf.tensor of size (nsamples, ) of sampled weights

        """
        channels = [channel] * nsamples
        one_hot_channels = tf.one_hot(channels, self.n_channels, dtype=self._dtype)
        samples, logq = self.dist.sample_and_log_prob(
            nsamples, condition=one_hot_channels
        )
        y, logq = self._compute_analytic_mappings(samples, logq, channels)

        weight = self._get_probs(
            samples, self._func(y), channels, weight_prior, return_integrand=True
        )

        return samples, weight

    @tf.function
    def sample_weights(
        self, nsamples: int, yield_samples: bool = False, weight_prior: Callable = None
    ):
        """Sample from the trained distribution and return their weights.

        This method samples 'nsamples' points from the trained distribution
        and computes their weights, defined as the functional value of the
        point divided by the probability of the trained distribution of
        that point.

        Optionally, the drawn samples can be returned, too.

        Args:
            nsamples (int): Number of samples to be drawn.
            yield_samples (bool, optional): return samples. Defaults to False.
            weight_prior (Callable, optional): returns the prior weights. Defaults to None.

        Returns:
            true/test: tf.tensor of size (nsamples, 1) of sampled weights
            (samples: tf.tensor of size (nsamples, ndims) of sampled points)

        """
        samples, _, func_vals, channels = self._get_samples(
            nsamples, self._get_variance_weights(), uniform_channel_ratio=0.0
        )
        weight = self._get_probs(
            samples, func_vals, channels, weight_prior, return_integrand=True
        )

        if yield_samples:
            return weight, samples

        return weight

    def acceptance(self, nopt: int, npool: int = 50, nreplica: int = 1000):
        """Calculate the acceptance, i.e. the unweighting
            efficiency as discussed in arXiv:2001.10028

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
        """Save the networks."""
        if self.train_flow:
            self.dist.save_weights(path + "flow")
        if self.train_mcw:
            self.mcw_model.save_weights(path + "mcw")

    def load_weights(self, path: str):
        """Load the networks."""
        if self.train_flow:
            self.dist.load_weights(path + "flow")
        if self.train_mcw:
            self.mcw_model.load_weight(path + "mcw")
        print("Models loaded successfully")
