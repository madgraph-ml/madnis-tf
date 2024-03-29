""" Implement the multi-channel flow integrator. """


import numpy as np
import tensorflow as tf
from typing import List, Union, Callable, Optional

from ..utils.divergences import Divergence
from ..mappings.flow import Flow
from ..mappings.multi_flow import MultiFlow
from ..mappings.base import Mapping
from ..distributions.base import Distribution

_EPSILON = 1e-16


class MultiChannelIntegrator:
    """Class implementing a (conditional) normalizing flow
    multi-channel integrator.
    """

    def __init__(
        self,
        func: Union[Callable, Distribution],
        dist: Distribution,
        optimizer: tf.keras.optimizers.Optimizer,
        mcw_model: Optional[tf.keras.Model] = None,
        mappings: Optional[List[Mapping]] = None,
        use_weight_init: bool = True,
        n_channels: int = 2,
        loss_func: str = "chi2",
        sample_capacity: int = 0,
        uniform_channel_ratio: float = 1.0,
        variance_history_length: int = 20,
        integrand_has_channels: bool = False,
        weight_prior: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Args:
            func (Union[Callable, Distribution]):
                Function to be integrated
            dist (Distribution):
                Trainable flow distribution to match the function
                or a fixed base distribution.
            optimizer (tf.keras.optimizers.Optimizer):
                Optimizer for the two models
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
                If active, the integrand is expected to return a tuple
                (weights, momenta, alphas). Defaults to False.
            weight_prior (Callable, optional):
                If integrand_has_channels is False and a function is given for weight_prior,
                this is used as the prior for the channel weights.
            kwargs: Additional arguments that need to be passed to the loss
        """
        self._dtype = tf.keras.backend.floatx()

        self.integrand_has_channels = integrand_has_channels
        self._func = func
        self.weight_prior = weight_prior
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
        self.divergence = Divergence(n_channels=self.n_channels, **kwargs)
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

    def _store_samples(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        alphas_prior: tf.Tensor,
        channels: tf.Tensor,
    ):
        """Stores the generated samples and probabilites to re-use for the buffered training.

        Args:
            x (tf.Tensor):
                Samples generated either uniformly or by the flow with shape (nsamples, ndim)
            y (tf.Tensor):
                Samples after transformation through analytic mappings or from the call to
                the integrand (if integrand_has_channels is True) with shape (nsamples, ndim)
            q_sample (tf.Tensor):
                Test probability of all mappings (analytic + flow) with shape (nsamples,)
            func_vals (tf.Tensor):
                True probability with shape (nsamples,)
            alphas_prior (tf.Tensor):
                Prior for the channel weights with shape (nsamples, nchannels)
            channels (tf.Tensor):
                Tensor encoding which channel to use with shape (nsamples,)
        """
        if self.sample_capacity == 0:
            return

        self.stored_samples.append((x, y, q_sample, func_vals, alphas_prior, channels))
        del self.stored_samples[: -self.sample_capacity]

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
            x (tf.Tensor):
                Samples generated either uniformly or by the flow with shape (nsamples, ndim)
            logq (tf.Tensor):
                Log probability of the uniform distribution or the flow with shape (nsamles,)
            channels (tf.Tensor):
                Tensor encoding which channel to use with shape (nsamples,)

        Returns:
            y (tf.Tensor):
                final sample output after all mappings with shape (nsamples, ndim)
            logq (tf.Tensor):
                combined log probability of all mappings with shape (nsamples,)
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
            jacs.append(self.mappings[i].log_det(yi))

        y = tf.dynamic_stitch(idx, ys)
        jac = tf.dynamic_stitch(idx, jacs)
        return y, logq + jac

    @tf.function(reduce_retracing=True)
    def _get_channels(
        self,
        nsamples: tf.Tensor,
        channel_weights: tf.Tensor,
        uniform_channel_ratio: tf.Tensor,
    ):
        """Sample a tensor of channel numbers in two steps:
        1. Split up nsamples * uniform_channel_ratio equally among all the channels
        2. Sample the rest of the events from the distribution given by channel_weights
           after correcting for the uniformly distributed samples
        This allows stratified sampling by variance weighting while ensuring stable training
        because there are events in every channel.

        Args:
            nsamples (tf.Tensor):
                Number of samples as scalar integer tensor
            channel_weights (tf.Tensor):
                Weights of the channels (not normalized) with shape (nchannels,)
            uniform_channel_ratio (tf.Tensor):
                Scalar tensor between 0.0 and 1.0 to determine the ratio of samples that
                will be distributed uniformly first

        Returns:
            channels (tf.Tensor):
                Tensor of channel numbers with shape (nsamples,)
        """
        tf.debugging.assert_equal(tf.shape(channel_weights), (self.n_channels,))
        n_uniform = tf.cast(
            tf.cast(nsamples, self._dtype) * uniform_channel_ratio, tf.dtypes.int32
        )
        uniform_channels = tf.tile(
            tf.range(self.n_channels), (n_uniform // self.n_channels + 1,)
        )[:n_uniform]

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
        channels: tf.Tensor
    ):
        """Draws samples from the flow for the given channels and then computes the analytic
        mappings, evaluates the integrand and the channel weight prior.

        Args:
            channels (tf.Tensor):
                Tensor encoding which channel to use with shape (nsamples,)

        Returns:
            x (tf.Tensor):
                Samples generated either uniformly or by the flow with shape (nsamples, ndim)
            y (tf.Tensor):
                Samples after transformation through analytic mappings or from the call to
                the integrand (if integrand_has_channels is True) with shape (nsamples, ndim)
            q_sample (tf.Tensor):
                Test probability of all mappings (analytic + flow) with shape (nsamples,)
            weight (tf.Tensor):
                True probability with shape (nsamples,)
            alphas_prior (tf.Tensor):
                Prior for the channel weights with shape (nsamples, nchannels)
        """
        nsamples = tf.shape(channels)[0]
        one_hot_channels = tf.one_hot(channels, self.n_channels, dtype=self._dtype)
        x, logq = self.dist.sample_and_log_prob(nsamples, condition=one_hot_channels)
        y, logq = self._compute_analytic_mappings(x, logq, channels)

        if self.integrand_has_channels:
            weight, y, alphas_prior = self._func(y, channels)
        else:
            weight = self._func(y)
            if self.weight_prior is not None:
                alphas_prior = self.weight_prior(y)
                tf.debugging.assert_equal(tf.shape(alphas_prior)[1], self.n_channels)
            else:
                alphas_prior = None

        return x, y, tf.math.exp(logq), weight, alphas_prior

    @tf.function(reduce_retracing=True)
    def _get_integral_and_alphas(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        alphas_prior: tf.Tensor,
        channels: tf.Tensor,
    ):
        """Return the weighted integrand and the channel weights

        Args:
            x (tf.Tensor):
                Samples generated either uniformly or by the flow with shape (nsamples, ndim)
            y (tf.Tensor):
                Samples after transformation through analytic mappings or from the call to
                the integrand (if integrand_has_channels is True) with shape (nsamples, ndim)
            q_sample (tf.Tensor):
                Test probability of all mappings (analytic + flow) with shape (nsamples,)
            func_vals (tf.Tensor):
                True probability with shape (nsamples,)
            alphas_prior (tf.Tensor):
                Prior for the channel weights with shape (nsamples, nchannels)
            channels (tf.Tensor):
                Tensor encoding which channel to use with shape (nsamples,)

        Returns:
            integrand (tf.Tensor):
                True probability weighted by alpha / qsample (nsamples,)
            alphas (tf.Tensor):
                Channel weights with shape (nsamples, nchannels)
        """
        q_test, logq, alphas = self._get_probs_alphas(
            x, y, alphas_prior, channels
        )
        return tf.gather(alphas, channels, batch_dims=1) * func_vals / q_sample, alphas

    @tf.function
    def _get_probs_alphas(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        alphas_prior: tf.Tensor,
        channels: tf.Tensor,
    ):
        """Run the flow to extract the probability and its logarithm for the given samples x.
        Compute the corresponding channel weights using y.

        Args:
            x (tf.Tensor):
                Samples generated either uniformly or by the flow with shape (nsamples, ndim)
            y (tf.Tensor):
                Samples after transformation through analytic mappings or from the call to
                the integrand (if integrand_has_channels is True) with shape (nsamples, ndim)
            alphas_prior (tf.Tensor):
                Prior for the channel weights with shape (nsamples, nchannels)
            channels (tf.Tensor):
                Tensor encoding which channel to use with shape (nsamples,)

        Returns:
            q_test (tf.tensor):
                true probability weighted by alpha / qsample (nsamples,)
            logq (tf.tensor):
                true probability weighted by alpha / qsample (nsamples,)
            alphas (tf.Tensor):
                Channel weights with shape (nsamples, nchannels)
        """
        nsamples = tf.shape(x)[0]
        one_hot_channels = tf.one_hot(channels, self.n_channels, dtype=self._dtype)
        logq = self.dist.log_prob(x, condition=one_hot_channels)
        y, logq = self._compute_analytic_mappings(x, logq, channels)
        q_test = tf.math.exp(logq)
        alphas = self._get_alphas(y, alphas_prior)
        return q_test, logq, alphas

    @tf.function
    def _get_probs_integral(
        self,
        alphas: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        channels: tf.Tensor,
    ):
        """Compute the channel wise means and variances of the integrand and use it to
        normalize the alpha-weighted true probability.

        Args:
            alphas (tf.Tensor):
                Channel weights with shape (nsamples, nchannels)
            q_sample (tf.Tensor):
                Test probability of all mappings (analytic + flow) with shape (nsamples,)
            func_vals (tf.Tensor):
                True probability with shape (nsamples,)
            channels (tf.Tensor):
                Tensor encoding which channel to use with shape (nsamples,)

        Returns:
            p_true (tf.Tensor):
                Normalized true probability with shape (nsamples,)
            logp (tf.Tensor):
                Log of the normalized true probability with shape (nsamples,)
            means (tf.Tensor):
                Channel-wise means of the integrand with shape (nchannels,)
            vars (tf.Tensor):
                Channel-wise variances of the integrand with shape (nchannels,)
            counts (tf.Tensor):
                Channel-wise number of samples with shape (nchannels,)
        """
        nsamples = tf.shape(q_sample)[0]
        alphas = tf.gather(alphas, channels, batch_dims=1)
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

    @tf.function
    def _get_alphas(
        self,
        y: tf.Tensor,
        alphas_prior: Optional[tf.Tensor]
    ):
        """Run the MCW network to get the channel weights. If no MCW network is given,
        the prior (if given) or uniform channel weights are returned.

        Args:
            y (tf.Tensor):
                Samples after transformation through analytic mappings or from the call to
                the integrand (if integrand_has_channels is True) with shape (nsamples, ndim)
            alphas_prior (tf.Tensor):
                Prior for the channel weights with shape (nsamples, nchannels)

        Returns:
            alphas (tf.Tensor):
                Channel weights with shape (nsamples, nchannels)
        """
        nsamples = tf.shape(y)[0]
        if self.train_mcw:
            if self.use_weight_init:
                if alphas_prior is None:
                    init_weights = (
                        1
                        / self.n_channels
                        * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                    )
                else:
                    init_weights = alphas_prior
                alphas = self.mcw_model([y, init_weights])
            else:
                alphas = self.mcw_model(y)
        else:
            if alphas_prior is not None:
                alphas = alphas_prior
            else:
                alphas = (
                    1
                    / self.n_channels
                    * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                )

        return alphas

    @tf.function(reduce_retracing=True)
    def _optimization_step(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor,
        alphas_prior: tf.Tensor,
        channels: tf.Tensor,
    ):
        """Perform one optimization step of the networks with the given samples

        Args:
            x (tf.Tensor):
                Samples generated either uniformly or by the flow with shape (nsamples, ndim)
            y (tf.Tensor):
                Samples after transformation through analytic mappings or from the call to
                the integrand (if integrand_has_channels is True) with shape (nsamples, ndim)
            q_sample (tf.Tensor):
                Test probability of all mappings (analytic + flow) with shape (nsamples,)
            func_vals (tf.Tensor):
                True probability with shape (nsamples,)
            alphas_prior (tf.Tensor):
                Prior for the channel weights with shape (nsamples, nchannels)
            channels (tf.Tensor):
                Tensor encoding which channel to use with shape (nsamples,)

        Returns:
            loss (tf.Tensor):
                Result for the loss (scalar)
            means (tf.Tensor):
                Channel-wise means of the integrand with shape (nchannels,)
            vars (tf.Tensor):
                Channel-wise variances of the integrand with shape (nchannels,)
            counts (tf.Tensor):
                Channel-wise number of samples with shape (nchannels,)
        """
        if not self.train_flow and not self.train_mcw:
            raise ValueError("No network defined which can be optimized")

        with tf.GradientTape() as tape:
            q_test, logq, alphas = self._get_probs_alphas(
                x, y, alphas_prior, channels
            )
            p_true, logp, means, vars, counts = self._get_probs_integral(
                alphas, q_sample, func_vals, channels
            )
            loss = self.loss_func(
                p_true, q_test, logp, logq, channels, q_sample=q_sample
            )

        trainable_weights = []
        if self.train_flow:
            trainable_weights.extend(self.dist.trainable_weights)
        if self.train_mcw:
            trainable_weights.extend(self.mcw_model.trainable_weights)
        grads = tape.gradient(loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        return loss, means, vars, counts

    def _get_variance_weights(self):
        """Use the list of saved variances to compute weights for sampling the
        channels to allow for stratified sampling.

        Returns:
            w (tf.Tensor):
                Weights for sampling the channels with shape (nchannels,)
        """
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

    def train_one_step(
        self,
        nsamples: int,
        integral: bool = False,
    ):
        """Perform one step of integration and improve the sampling.

        Args:
            nsamples (int):
                Number of samples to be taken in a training step
            integral (bool, optional):
                return the integral value. Defaults to False.

        Returns:
            loss:
                Value of the loss function for this step
            integral (optional):
                Estimate of the integral value
            uncertainty (optional):
                Integral statistical uncertainty
        """

        # Sample from flow and update
        channels = self._get_channels(
            tf.constant(nsamples),
            self._get_variance_weights(),
            self.uniform_channel_ratio,
        )
        x, y, q_sample, func_vals, alphas_prior = self._get_samples(channels)
        loss, means, vars, counts = self._optimization_step(
            x, y, q_sample, func_vals, alphas_prior, channels
        )

        self._store_samples(x, y, q_sample, func_vals, alphas_prior, channels)

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

    def train_on_stored_samples(self, batch_size: int):
        """Train the network on all saved samples.

        Args:
            batch_size (int):
                Size of the batches that the saved samples are split up into

        Returns:
            loss:
                Averaged value of the loss function
        """
        sample_count = sum(int(tf.shape(item[0])[0]) for item in self.stored_samples)
        perm = tf.random.shuffle(tf.range(sample_count))

        dataset = (
            tf.data.Dataset.from_tensor_slices(tuple(
                tf.gather(tf.concat(item, axis=0), perm, axis=0)
                for item in zip(*self.stored_samples)
            ))
            .batch(batch_size, drop_remainder=True)
        )

        losses = []
        for xs, ys, qs, fs, alphas_prior, cs in dataset:
            loss, _, _, _ = self._optimization_step(xs, ys, qs, fs, alphas_prior, cs)
            losses.append(loss)
        return tf.reduce_mean(losses)

    def integrate(self, nsamples: int, return_channels: bool = False):
        """Integrate the function with trained distribution.

        This method estimates the value of the integral based on
        Monte Carlo importance sampling. It returns a tuple of two
        tf.tensors. The first one is the mean, i.e. the estimate of
        the integral. The second one is the variance of the estimated mean,
        i.e. the square root of (the variance divided by (nsamples - 1)).

        Args:
            nsamples (int):
                Number of points on which the estimate is based on
            return_channels (bool, optional):
                If True, also return channel-wise means and variances. Defaults to False

        Returns:
            mean (tf.Tensor):
                integration result
            std (tf.Tensor):
                standard error
            chan_means (tf.Tensor, optional):
                channel-wise integration results
            chan_stds (tf.Tensor, optional):
                channel-wise standard errors
        """
        mean, std, chan_means, chan_stds = self._integrate(tf.constant(nsamples))
        if return_channels:
            return mean, std, chan_means, chan_stds
        else:
            return mean, std

    @tf.function
    def _integrate(self, nsamples):
        """Integrate the function with trained distribution.
        See function integrate for details.

        Args:
            nsamples (int):
                Number of points on which the estimate is based on

        Returns:
            mean (tf.Tensor):
                integration result
            std (tf.Tensor):
                standard error
            chan_means (tf.Tensor):
                channel-wise integration results
            chan_stds (tf.Tensor):
                channel-wise standard errors
        """
        channels = self._get_channels(
            nsamples,
            self._get_variance_weights(),
            uniform_channel_ratio=tf.constant(0.0, dtype=self._dtype),
        )
        x, y, q_sample, func_vals, alphas_prior = self._get_samples(channels)
        integrands, _ = self._get_integral_and_alphas(
            x, y, q_sample, func_vals, alphas_prior, channels
        )
        mean = 0.0
        chan_means = []
        var = 0.0
        chan_stds = []
        integs = tf.dynamic_partition(integrands, channels, self.n_channels)
        for integ in integs:
            meani, vari = tf.nn.moments(integ, axes=[0])
            var_scale = tf.cast(tf.shape(integ)[0], self._dtype) - 1.0
            chan_means.append(meani)
            chan_stds.append(tf.sqrt(vari / var_scale))
            mean += meani
            var += vari / var_scale
        return mean, tf.sqrt(var), chan_means, chan_stds

    @tf.function
    def _sample_weights_and_alphas(
        self,
        channels: tf.Tensor
    ):
        """Draw samples for the given channels and compute the integrands and channel weights.

        Args:
            channels (tf.Tensor):
                Tensor encoding which channel to use with shape (nsamples,)

        Returns:
            y (tf.Tensor):
                Samples after transformation through analytic mappings or from the call to
                the integrand (if integrand_has_channels is True) with shape (nsamples, ndim)
            weight (tf.Tensor):
                True probability weighted by alpha / qsample (nsamples,)
            alphas (tf.Tensor):
                Channel weights with shape (nsamples, nchannels)
            alphas_prior (tf.Tensor):
                Prior for the channel weights with shape (nsamples, nchannels)
        """
        x, y, q_sample, func_vals, alphas_prior = self._get_samples(channels)
        weight, alphas = self._get_integral_and_alphas(
            x, y, q_sample, func_vals, alphas_prior, channels
        )
        return y, weight, alphas, alphas_prior

    def sample_per_channel(
        self,
        nsamples: int,
        channel: int,
        return_alphas: bool = False,
    ):
        """Sample from the trained distribution and return their weights
        for a single channel.

        This method samples 'nsamples' points from the trained distribution
        and computes their weights, defined as the functional value of the
        point divided by the probability of the trained distribution of
        that point.

        Args:
            nsamples (int):
                Number of samples to be drawn.
            channel (int):
                the channel of the sampling.
            return_alphas (bool):
                if True, also return channel weights. Defaults to False

        Returns:
            y (tf.Tensor):
                Samples after transformation through analytic mappings or from the call to
                the integrand (if integrand_has_channels is True) with shape (nsamples, ndim)
            weight (tf.Tensor):
                True probability weighted by alpha / qsample (nsamples,)
            alphas (tf.Tensor, optional):
                Channel weights with shape (nsamples, nchannels)
            alphas_prior (tf.Tensor, optional):
                Prior for the channel weights with shape (nsamples, nchannels)
        """
        channels = tf.fill((nsamples,), channel)
        y, weight, alphas, alphas_prior = self._sample_weights_and_alphas(channels)
        if return_alphas:
            return y, weight, alphas, alphas_prior
        else:
            return y, weight

    def sample_weights(
        self,
        nsamples: int,
        yield_samples: bool = False,
    ):
        """Sample from the trained distribution and return their weights.

        This method samples 'nsamples' points from the trained distribution
        and computes their weights, defined as the functional value of the
        point divided by the probability of the trained distribution of
        that point.

        Optionally, the drawn samples can be returned, too.

        Args:
            nsamples (int):
                Number of samples to be drawn.
            yield_samples (bool, optional):
                If True, also return samples. Defaults to False.

        Returns:
            weight (tf.Tensor):
                True probability weighted by alpha / qsample (nsamples,)
            y (tf.Tensor):
                Samples after transformation through analytic mappings or from the call to
                the integrand (if integrand_has_channels is True) with shape (nsamples, ndim)
        """
        channels = self._get_channels( 
            tf.constant(nsamples),
            self._get_variance_weights(),
            uniform_channel_ratio=tf.constant(0.0, dtype=self._dtype)
        )
        y, weight, _, _ = self._sample_weights_and_alphas(channels)
        if yield_samples:
            return weight, y
        else:
            return weight

    def acceptance(self, nopt: int, npool: int = 50, nreplica: int = 1000):
        """Calculate the acceptance, i.e. the unweighting efficiency
        as discussed in arXiv:2001.10028

        Args:
            nopt (int):
                Number of points on which the optimization was based on.
            npool (int, optional):
                called n in the reference. Defaults to 50.
            nreplica (int, optional):
                called m in the reference. Defaults to 1000.

        Returns:
            s_acc (float):
                unweighting efficiency
            s_acc_partial (float):
                partial unweighting efficiency computed by setting over-weights to 1
            over_weight_rate (float):
                rate of over-weights
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
        """Save the networks.

        Args:
            path (str):
                Path to save the networks to
        """
        if self.train_flow:
            self.dist.save_weights(path + "flow")
        if self.train_mcw:
            self.mcw_model.save_weights(path + "mcw")

    def load_weights(self, path: str):
        """Load the networks.

        Args:
            path (str):
                Path to load the networks from
        """
        if self.train_flow:
            self.dist.load_weights(path + "flow")
        if self.train_mcw:
            self.mcw_model.load_weights(path + "mcw")
        print("Models loaded successfully")
