""" Implement the multi-channel flow integrator. """


import numpy as np
import tensorflow as tf
from typing import List, Union, Callable

from ..utils.divergences import Divergence
from ..utils.bayesian import BayesianHelper
from ..mappings.flow import Flow
from ..mappings.multi_flow import MultiFlow
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
        optimizer: tf.keras.optimizers.Optimizer,
        mcw_model: tf.keras.Model = None,
        mappings: List[Mapping] = None,
        use_weight_init: bool = True,
        n_channels: int = 2,
        loss_func: str = "chi2",
        sample_capacity: int = 0,
        uniform_channel_ratio: float = 1.0,
        variance_history_length: int = 20,
        integrand_has_channels: bool = False,
        bayesian_helper: BayesianHelper = None,
        single_pass_opt: bool = False,
        second_order_opt: bool = False,
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
            integrand_has_channels (bool, optional):
                Whether to pass the channel number to the integrand as the second argument.
                Defaults to False.
            flow_bayesian_helper (BayesianHelper, optional):
                If present, the training of the flow is done in a bayesian way
            mcw_bayesian_helper (BayesianHelper, optional):
                If present, the training of the MC weights is done in a bayesian way
            kwargs: Additional arguments that need to be passed to the loss
        """
        self._dtype = tf.keras.backend.floatx()

        if integrand_has_channels:
            self._func = func
        else:
            self._func = lambda x, c: func(x)
        self.dist = dist

        # Define flow or base mapping
        self.trainable_weights = []
        if isinstance(dist, Flow) or isinstance(dist, MultiFlow):
            self.train_flow = True
        else:
            self.train_flow = False

        # Define mcw model if given
        self.mcw_model = mcw_model
        if self.mcw_model is None:
            self.train_mcw = False
        else:
            self.train_mcw = True

        # Define optimizer
        self.optimizer = optimizer

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
        self.single_pass_opt = single_pass_opt
        self.second_order_opt = second_order_opt
        self.divergence = Divergence(
            n_channels=self.n_channels, single_pass_opt=single_pass_opt, **kwargs
        )
        self.loss_func = self.divergence(loss_func)
        self.class_loss = tf.keras.losses.categorical_crossentropy

        self.uniform_channel_ratio = tf.constant(
            uniform_channel_ratio, dtype=self._dtype
        )
        self.variance_history_length = variance_history_length
        self.variance_history = []
        self.count_history = []

        self.sample_capacity = sample_capacity
        if sample_capacity > 0:
            self.stored_samples = []
            self.stored_q_sample = []
            self.stored_func_vals = []
            self.stored_channels = []

        self.bayesian_helper = bayesian_helper

    def _store_samples(
        self,
        samples: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        channels: tf.Tensor,
    ):
        """
        Stores the generated samples and probabilites
        to re-use for the two-stage training.

        Args:
            samples (tf.Tensor): Samples generated either uniformly or by the flow.
            q_sample (tf.Tensor): Test probability of all mappings (analytic + flow)
            func_vals (tf.Tensor): True probability
            channels (tf.Tensor): Tensor encoding which channel to use with shape (nsamples,).
        """
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

    @tf.function
    def _compute_analytic_mappings(
        self,
        x: tf.Tensor,
        logq: tf.Tensor,
        channels: tf.Tensor,
    ):
        """Computes the remapped output and log determinant of possible
        fixed analytic mappings used.

        Args:
            x (tf.Tensor: Input coming either from an uniform distribution or from a flow.
            logq (tf.Tensor: Log probability of the uniform distribution or the flow.
            channels (tf.Tensor): Tensor encoding which channel to use with shape (nsamples,).

        Returns:
            y (tf.Tensor): final sample output after all mappings.
            logq (tf.Tensor): combined log probability of all mappings.
        """
        if not self.use_analytic_mappings:
            return x, logq

        xs = tf.dynamic_partition(x, channels, self.n_channels)
        idx = tf.dynamic_partition(tf.range(tf.shape(x)[0]), channels, self.n_channels)
        ys = []
        jacs = []
        for i, xi in enumerate(xs):
            yi, _ = self.mappings[i].inverse(xi)
            ys.append(yi)
            jacs.append(self.mappings[i].log_prob(yi))

        y = tf.dynamic_stitch(idx, ys)
        jac = tf.dynamic_stitch(idx, jacs)
        return y, logq + jac

    @tf.function(reduce_retracing=True)
    def _get_samples_fixed(self, nsamples: tf.Tensor, channels: tf.Tensor):
        one_hot_channels = tf.one_hot(channels, self.n_channels, dtype=self._dtype)
        x, logq = self.dist.sample_and_log_prob(nsamples, condition=one_hot_channels)
        y, logq = self._compute_analytic_mappings(x, logq, channels)
        return x, y, logq

    @tf.function(reduce_retracing=True)
    def _get_channels(
        self,
        nsamples: tf.Tensor,
        channel_weights: tf.Tensor,
        uniform_channel_ratio: tf.Tensor,
    ):
        assert channel_weights.shape == (self.n_channels,)
        # Split up nsamples * uniform_channel_ratio equally among all the channels
        n_uniform = tf.cast(
            tf.cast(nsamples, self._dtype) * uniform_channel_ratio, tf.dtypes.int32
        )
        uniform_channels = tf.tile(
            tf.range(self.n_channels), (n_uniform // self.n_channels + 1,)
        )[:n_uniform]
        # Sample the rest of the events from the distribution given by channel_weights
        # after correcting for the uniformly distributed samples
        normed_weights = channel_weights / tf.reduce_sum(channel_weights)
        probs = tf.maximum(
            normed_weights
            - uniform_channel_ratio / tf.constant(self.n_channels, dtype=self._dtype),
            1e-15,
        )
        sampled_channels = tf.random.categorical(
            tf.math.log(probs)[None, :], nsamples - n_uniform, dtype=tf.int32
        )[0]
        channels = tf.concat((uniform_channels, sampled_channels), axis=0)
        return channels

    @tf.function(reduce_retracing=True)
    def _get_samples(
        self,
        nsamples: tf.Tensor,
        channel_weights: tf.Tensor,
        uniform_channel_ratio: tf.Tensor,
        return_mom: bool = False,
    ):
        """
        Args:
            nsamples (tf.Tensor): Numper of samples to be generated.
            channel_weights (tf.Tensor): Importance of each channel with shape (n_channels,1)
            uniform_channel_ratio (tf.Tensor): ratio of samples which are distributed
                uniformly across all channels. If > 0, this guarantees that no channel is empty.

        Returns:
            x (tf.Tensor): output generated either uniformly or by the flow.
            q (tf.Tensor): test probability of all mappings (analytic + flow)
            p (tf.Tensor): true probability/function.
            channels (tf.Tensor): tensor encoding which channel to use with shape (nsamples,).
        """
        channels = self._get_channels(nsamples, channel_weights, uniform_channel_ratio)
        x, y, logq = self._get_samples_fixed(nsamples, channels)

        if return_mom:
            return x, y, tf.math.exp(logq), self._func(y, channels), channels

        return x, tf.math.exp(logq), self._func(y, channels), channels

    @tf.function(reduce_retracing=True)
    def _get_single_pass_probs(
        self,
        nsamples: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable = None,
        return_integrand: bool = False,
        return_alphas: bool = False,
    ):
        x, y, q_test, logq, alphas, alphas_prior = self._get_single_pass_probs_alphas(
            nsamples, channels, weight_prior
        )

        if return_integrand:
            if return_alphas:
                if alphas_prior is not None:
                    alphas_prior = tf.gather(alphas_prior, channels, batch_dims=1)
                return alphas, alphas_prior, alphas * self._func(y, channels) / q_test
            else:
                return alphas * self._func(y, channels) / q_test

        p_true, logp, means, vars, counts = self._get_probs_integral(
            nsamples, alphas, q_test, self._func(y, channels), channels
        )
        return (
            p_true,
            q_test,
            logp,
            logq,
            means,
            vars,
            counts,
            x,
            self._func(y, channels),
        )

    @tf.function(reduce_retracing=True)
    def _get_probs(
        self,
        samples: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable = None,
        return_integrand: bool = False,
        return_alphas: bool = False,
    ):
        nsamples, q_test, logq, alphas, alphas_prior = self._get_probs_alphas(
            samples, channels, weight_prior
        )

        if return_integrand:
            if return_alphas:
                if alphas_prior is not None:
                    alphas_prior = tf.gather(alphas_prior, channels, batch_dims=1)
                return alphas, alphas_prior, alphas * func_vals / q_sample
            else:
                return alphas * func_vals / q_sample

        p_true, logp, means, vars, counts = self._get_probs_integral(
            nsamples, alphas, q_sample, func_vals, channels
        )
        return p_true, q_test, logp, logq, means, vars, counts

    @tf.function
    def _get_alphas(
        self,
        y: tf.Tensor,
        weight_prior: Callable = None,
    ):
        nsamples = tf.shape(y)[0]
        alphas_prior = None
        if self.train_mcw:
            if self.use_weight_init:
                if weight_prior is not None:
                    init_weights = weight_prior(y)
                    alphas_prior = init_weights
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
                alphas_prior = alphas
                assert alphas.shape[1] == self.n_channels
            else:
                alphas = (
                    1
                    / self.n_channels
                    * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                )

        return alphas, alphas_prior

    @tf.function
    def _get_single_pass_probs_alphas(
        self,
        nsamples: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable = None,
    ):
        one_hot_channels = tf.one_hot(channels, self.n_channels, dtype=self._dtype)
        x, logq = self.dist.sample_and_log_prob(nsamples, condition=one_hot_channels)
        y, logq = self._compute_analytic_mappings(x, logq, channels)
        q_test = tf.math.exp(logq)

        alphas, alphas_prior = self._get_alphas(y, weight_prior)
        alphas = tf.gather(alphas, channels, batch_dims=1)

        return x, y, q_test, logq, alphas, alphas_prior

    @tf.function
    def _get_probs_alphas(
        self,
        samples: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable = None,
    ):
        nsamples = tf.shape(samples)[0]
        one_hot_channels = tf.one_hot(channels, self.n_channels, dtype=self._dtype)
        logq = self.dist.log_prob(samples, condition=one_hot_channels)
        y, logq = self._compute_analytic_mappings(samples, logq, channels)
        q_test = tf.math.exp(logq)

        alphas, alphas_prior = self._get_alphas(y, weight_prior)
        alphas = tf.gather(alphas, channels, batch_dims=1)

        return nsamples, q_test, logq, alphas, alphas_prior

    @tf.function
    def _get_probs_integral(
        self,
        nsamples: tf.Tensor,
        alphas: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        channels: tf.Tensor,
    ):
        p_unnormed = alphas * tf.abs(func_vals)
        p_trues = []
        means = []
        vars = []
        counts = []
        ps = tf.dynamic_partition(p_unnormed, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
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
            logp,
            tf.convert_to_tensor(means),
            tf.convert_to_tensor(vars),
            tf.convert_to_tensor(counts),
        )

    @tf.function(reduce_retracing=True)
    def _optimization_step(
        self,
        samples: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable,
    ):
        if not self.train_flow and not self.train_mcw:
            raise ValueError("No network defined which can be optimized")

        with tf.GradientTape() as tape:
            p_true, q_test, logp, logq, means, vars, counts = self._get_probs(
                samples, q_sample, func_vals, channels, weight_prior
            )
            loss = self.loss_func(
                p_true, q_test, logp, logq, channels, q_sample=q_sample
            )
            if self.bayesian_helper is not None:
                loss += self.bayesian_helper.kl_loss()

        trainable_weights = []
        if self.train_flow:
            trainable_weights.extend(self.dist.trainable_weights)
        if self.train_mcw:
            trainable_weights.extend(self.mcw_model.trainable_weights)
        grads = tape.gradient(loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        return loss, means, vars, counts

    @tf.function
    def _single_pass_optimization_step(
        self,
        nsamples: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable,
    ):
        if not self.train_flow and not self.train_mcw:
            raise ValueError("No network defined which can be optimized")

        with tf.GradientTape() as tape:
            (
                p_true,
                q_test,
                logp,
                logq,
                means,
                vars,
                counts,
                samples,
                func_vals,
            ) = self._get_single_pass_probs(nsamples, channels, weight_prior)
            loss = self.loss_func(p_true, q_test, logp, logq, channels, q_sample=q_test)

            if self.bayesian_helper is not None:
                loss += self.bayesian_helper.kl_loss()

        trainable_weights = []
        if self.train_flow:
            trainable_weights.extend(self.dist.trainable_weights)
        if self.train_mcw:
            trainable_weights.extend(self.mcw_model.trainable_weights)
        grads = tape.gradient(loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        return (
            loss,
            means,
            vars,
            counts,
            tf.stop_gradient(samples),
            tf.stop_gradient(q_test),
            tf.stop_gradient(func_vals),
        )

    @tf.function
    def _classification_optimization_step(
        self,
        samples: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable,
    ):
        if not self.train_mcw:
            raise ValueError("No network defined which can be optimized")

        with tf.GradientTape() as tape:
            alpha_vector, _ = self._get_alphas(samples, weight_prior, x_samples=True)
            loss = tf.reduce_mean(self.class_loss(channels, alpha_vector), axis=0)

        grads = tape.gradient(loss, self.mcw_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.mcw_model.trainable_weights))

        return loss

    def _get_variance_weights(self):
        if len(self.variance_history) < self.variance_history_length:
            return tf.fill((self.n_channels,), tf.constant(1.0, dtype=self._dtype))

        count_hist = tf.convert_to_tensor(self.count_history, dtype=self._dtype)
        var_hist = tf.convert_to_tensor(self.variance_history, dtype=self._dtype)
        hist_weights = count_hist / tf.reduce_sum(count_hist, axis=0)
        w = tf.reduce_sum(hist_weights * var_hist, axis=0)
        return w

    def delete_samples(self):
        """Delete all stored samples."""
        del self.stored_samples[:]
        del self.stored_q_sample[:]
        del self.stored_func_vals[:]
        del self.stored_channels[:]

    def train_one_classification_step(
        self,
        nsamples: int,
        weight_prior: Callable = None,
    ):
        """Perform one step of alpha optimization using classification

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
        samples, _, _, channels = self._get_samples(
            tf.constant(nsamples),
            self._get_variance_weights(),
            self.uniform_channel_ratio,
        )

        loss = self._classification_optimization_step(samples, channels, weight_prior)

        return loss

    def train_one_step(
        self,
        nsamples: int,
        weight_prior: Callable = None,
        integral: bool = False,
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

        # Sample from flow and update
        if self.single_pass_opt:
            channels = self._get_channels(
                tf.constant(nsamples),
                self._get_variance_weights(),
                self.uniform_channel_ratio,
            )

            (
                loss,
                means,
                vars,
                counts,
                samples,
                q_sample,
                func_vals,
            ) = self._single_pass_optimization_step(
                tf.constant(nsamples), channels, weight_prior
            )

        else:
            samples, q_sample, func_vals, channels = self._get_samples(
                tf.constant(nsamples),
                self._get_variance_weights(),
                self.uniform_channel_ratio,
            )

            loss, means, vars, counts = self._optimization_step(
                samples, q_sample, func_vals, channels, weight_prior
            )

        self._store_samples(samples, q_sample, func_vals, channels)

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
        samples = tf.concat(self.stored_samples, axis=0)
        q_sample = tf.concat(self.stored_q_sample, axis=0)
        func_vals = tf.concat(self.stored_func_vals, axis=0)
        channels = tf.concat(self.stored_channels, axis=0)
        perm = tf.random.shuffle(tf.range(int(tf.shape(samples)[0])))

        dataset = (
            tf.data.Dataset.from_tensor_slices((
                tf.gather(samples, perm, axis=0),
                tf.gather(q_sample, perm, axis=0),
                tf.gather(func_vals, perm, axis=0),
                tf.gather(channels, perm, axis=0)
            ))
            .batch(batch_size, drop_remainder=True)
        )

        losses = []
        for ys, qs, fs, cs in dataset:
            loss, _, _, _ = self._optimization_step(ys, qs, fs, cs, weight_prior)
            losses.append(loss)
        return tf.reduce_mean(losses)

    def integrate(self, nsamples: int, weight_prior: Callable = None):
        """Integrate the function with trained distribution.

        This method estimates the value of the integral based on
        Monte Carlo importance sampling. It returns a tuple of two
        tf.tensors. The first one is the mean, i.e. the estimate of
        the integral. The second one is the variance of the estimated mean,
        i.e. the square root of (the variance divided by (nsamples - 1)).

        Args:
            nsamples (int): Number of points on which the estimate is based on.
            weight_prior (Callable, optional): returns the prior weights. Defaults to None.

        Returns:
            tuple of 2 tf.tensors: mean and mc error

        """
        return self._integrate(tf.constant(nsamples), weight_prior)

    @tf.function
    def _integrate(self, nsamples, weight_prior):
        samples, q_sample, func_vals, channels = self._get_samples(
            nsamples,
            self._get_variance_weights(),
            uniform_channel_ratio=tf.constant(0.0, dtype=self._dtype),
        )
        integrands = self._get_probs(
            samples, q_sample, func_vals, channels, weight_prior, return_integrand=True
        )
        mean = 0.0
        var = 0.0
        integs = tf.dynamic_partition(integrands, channels, self.n_channels)
        for integ in integs:
            meani, vari = tf.nn.moments(integ, axes=[0])
            mean += meani
            var += vari / (tf.cast(tf.shape(integ)[0], self._dtype) - 1.0)
        return mean, tf.sqrt(var)

    def sample_per_channel(
        self,
        nsamples: int,
        channel: int,
        weight_prior: Callable = None,
        return_alphas: bool = False,
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
            return_alphas (bool): if True, also return channel weights

        Returns:
            samples: tf.tensor of size (nsamples, ndims) of sampled points
            true/test: tf.tensor of size (nsamples, ) of sampled weights

        """
        y, weight, alphas, alphas_prior = self._sample_per_channel(
            tf.constant(nsamples), tf.constant(channel), weight_prior
        )
        if return_alphas:
            return y, weight, alphas, alphas_prior
        else:
            return y, weight

    @tf.function
    def _sample_per_channel(self, nsamples, channel, weight_prior):
        channels = tf.fill((nsamples,), channel)
        x, y, logq = self._get_samples_fixed(nsamples, channels)
        q_sample = tf.math.exp(logq)
        alphas, alphas_prior, weight = self._get_probs(
            x,
            q_sample,
            self._func(y, channels),
            channels,
            weight_prior,
            return_integrand=True,
            return_alphas=True,
        )
        return y, weight, alphas, alphas_prior

    def sample_weights(
        self,
        nsamples: int,
        yield_samples: bool = False,
        weight_prior: Callable = None,
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
        weight, samples = self._sample_weights(nsamples, weight_prior)
        if yield_samples:
            return weight, samples
        else:
            return weight

    @tf.function
    def _sample_weights(self, nsamples, weight_prior):
        samples, p_samples, q_sample, func_vals, channels = self._get_samples(
            tf.constant(nsamples),
            self._get_variance_weights(),
            uniform_channel_ratio=tf.constant(0.0, dtype=self._dtype),
            return_mom=True,
        )
        weight = self._get_probs(
            samples, q_sample, func_vals, channels, weight_prior, return_integrand=True
        )

        return weight, p_samples

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

        s_acc = np.mean(s_mean / np.median(s_max))
        s_acc_partial = np.mean(np.minimum(sample / np.median(s_max), 1))

        # Get accuracy without overweights
        over_weight_rate = np.mean(sample > np.median(s_max))

        return s_acc, s_acc_partial, over_weight_rate

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
            self.mcw_model.load_weights(path + "mcw")
        print("Models loaded successfully")
