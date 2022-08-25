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
        flow: Flow,
        optimizer: List[tf.keras.optimizers.Optimizer],
        mcw_model: tf.keras.Model = None,
        mappings: List[Mapping] = None,
        use_weight_init: bool = True,
        n_channels: int = 2,
        loss_func: str = "chi2",
        sample_capacity: int = 0,
        **kwargs,
    ):
        """
        Args:
            func (Union[Callable, Distribution]):
                Function to be integrated
            flow (Flow):
                Trainable flow model to match the function
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
            kwargs: Additional arguments that need to be passed to the loss
        """
        self._dtype = tf.keras.backend.floatx()

        self._func = func
        self.flow = flow

        # Define mcw model if given
        self.mcw_model = mcw_model
        if self.mcw_model is None:
            self.train_mcw = False
        else:
            self.train_mcw = True

        # Define optimizers
        if len(optimizer) > 1:
            assert self.mcw_model is not None
            self.flow_optimizer = optimizer[0]
            self.mcw_optimizer = optimizer[1]
        else:
            self.flow_optimizer = optimizer[0]
            self.mcw_optimizer = None

        self.use_weight_init = use_weight_init
        self.n_channels = n_channels

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

        self.sample_capacity = sample_capacity
        if sample_capacity > 0:
            self.stored_samples = []
            self.stored_q_sample = []
            self.stored_func_vals = []
            self.stored_dataset = None

    def _store_samples(self, samples: tf.Tensor, q_sample: tf.Tensor, func_vals: tf.Tensor):
        if self.sample_capacity == 0:
            return

        self.stored_samples.append(samples)
        self.stored_q_sample.append(q_sample)
        self.stored_func_vals.append(func_vals)
        del self.stored_samples[:-self.sample_capacity]
        del self.stored_q_sample[:-self.sample_capacity]
        del self.stored_func_vals[:-self.sample_capacity]
        self.stored_dataset = None

    def _get_channel_condition(self, nsamples: int):
        # creates ones with shape (nsamples, nc)
        cond = tf.ones((nsamples, self.n_channels), dtype=self._dtype)

        # creates shape (b, nc, nc) with b unit-matrices (nc x nc)
        c_cond = tf.linalg.diag(cond)
        return c_cond

    @tf.function
    def _get_samples(self, nsamples: int, one_hot_channels: tf.Tensor):
        ys = []
        fs = []
        qs = []

        for i in range(self.n_channels):
            # Channel dependent flow sampling
            xi, logqi = self.flow.sample_and_log_prob(
                nsamples, condition=one_hot_channels[:, :, i]
            )

            # Check for analytic remappings
            if self.use_analytic_mappings:
                yi, _ = self.mappings[i].inverse(xi)
                logqi += self.mappings[i].log_prob(yi)
            else:
                yi = xi

            ys.append(yi)
            qs.append(tf.math.exp(logqi))
            fs.append(self._func(yi))

        # Get concatenated stuff all in shape (nsamples, n_channels)
        return tf.stack(ys, axis=-1), tf.stack(qs, axis=-1), tf.stack(fs, axis=-1)

    @tf.function
    def _get_probs(
        self,
        samples: tf.Tensor,
        func_vals: tf.Tensor,
        one_hot_channels: tf.Tensor,
        weight_prior: Callable = None,
    ):
        ps = []
        qs = []
        logqs = []
        means = []
        vars = []
        nsamples = samples.shape[0]

        for i in range(self.n_channels):
            yi = samples[:, :, i]

            # Flow density estimation
            if self.use_analytic_mappings:
                xi, _ = self.mappings[i](yi)
                remap_jac = self.mappings[i].log_prob(yi)
            else:
                xi = yi
                remap_jac = 0

            logqi = self.flow.log_prob(xi, condition=one_hot_channels[:, :, i])
            logqi += remap_jac

            qi = tf.math.exp(logqi)
            logqs.append(logqi)
            qs.append(qi)

            # Get multi-channel weights
            if self.train_mcw:
                if self.use_weight_init:
                    if weight_prior is not None:
                        init_weights = weight_prior(yi)
                        assert init_weights.shape[1] == self.n_channels
                    else:
                        init_weights = (
                            1
                            / self.n_channels
                            * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                        )
                    alphas = self.mcw_model([yi, init_weights])
                else:
                    alphas = self.mcw_model(yi)
            else:
                if weight_prior is not None:
                    alphas = weight_prior(yi)
                    assert alphas.shape[1] == self.n_channels
                else:
                    alphas = (
                        1
                        / self.n_channels
                        * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                    )

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
        logp = tf.math.log(p_true + _EPSILON)

        # TODO: Understand why this where returns an error.
        # cond = tf.less(p_true, _EPSILON)
        # logp = tf.where(
        #     cond, tf.math.log(p_true), tf.math.log(p_true + _EPSILON)
        # )

        return p_true, q_test, logp, logq, sum(means), sum(vars)

    @tf.function
    def _get_integrand(
        self, nsamples: int, one_hot_channels: tf.Tensor, weight_prior: Callable = None
    ):
        ps = []
        qs = []

        for i in range(self.n_channels):
            # Channel dependent flow sampling
            sample = self.flow.sample(nsamples, condition=one_hot_channels[:, :, i])
            logqi = self.flow.log_prob(sample, condition=one_hot_channels[:, :, i])

            # Check for analytic remappings
            if self.use_analytic_mappings:
                yi, _ = self.mappings[i].inverse(sample)
                logqi += self.mappings[i].log_det(yi)
            else:
                yi = sample

            # Define test probs
            qi = tf.math.exp(logqi)
            qs.append(qi[..., None])

            # Get multi-channel weights
            if self.train_mcw:
                if self.use_weight_init:
                    if weight_prior is not None:
                        init_weights = weight_prior(yi)
                        assert init_weights.shape[1] == self.n_channels
                    else:
                        init_weights = (
                            1
                            / self.n_channels
                            * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                        )
                    alphas = self.mcw_model([yi, init_weights])
                else:
                    alphas = self.mcw_model(yi)
            else:
                if weight_prior is not None:
                    alphas = weight_prior(yi)
                    assert alphas.shape[1] == self.n_channels
                else:
                    alphas = (
                        1
                        / self.n_channels
                        * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                    )

            # Get true integrand
            pi = alphas[:, i] * self._func(yi)
            ps.append(pi[..., None])

        # Get concatenated stuff all in shape (nsamples, n_channels)
        p_true = tf.concat(ps, axis=-1)
        q_test = tf.concat(qs, axis=-1)

        return p_true / q_test

    @tf.function
    def _optimization_step(
        self,
        samples: tf.Tensor,
        q_sample: tf.Tensor,
        func_vals: tf.Tensor, 
        one_hot_channels: tf.Tensor,
        weight_prior: Callable
    ):
        loss = 0

        # Optimize the Flow
        with tf.GradientTape() as tape:
            p_true, q_test, logp, logq, mean, var = self._get_probs(
                samples, func_vals, one_hot_channels, weight_prior
            )
            flow_loss = self.flow_loss_func(
                p_true, q_test, logp, logq, q_sample=q_sample
            )

        grads = tape.gradient(flow_loss, self.flow.trainable_weights)
        self.flow_optimizer.apply_gradients(zip(grads, self.flow.trainable_weights))
        loss += flow_loss

        # Optimize the channel weight
        if self.train_mcw:
            with tf.GradientTape() as tape:
                p_true, q_test, logp, logq, mean, var = self._get_probs(
                    samples, func_vals, one_hot_channels
                )
                mcw_loss = self.mcw_loss_func(
                    p_true, q_test, logp, logq, q_sample=q_sample
                )

            grads = tape.gradient(mcw_loss, self.mcw_model.trainable_weights)
            self.mcw_optimizer.apply_gradients(
                zip(grads, self.mcw_model.trainable_weights)
            )
            loss += mcw_loss

        return loss, mean, var

    def delete_samples(self):
        """ Delete all stored samples. """
        del self.stored_samples[:]
        del self.stored_q_sample[:]
        del self.stored_func_vals[:]
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

        Args:
            nsamples (int): _description_
            integral (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # Sample from flow
        one_hot_channels = self._get_channel_condition(nsamples)
        samples, q_sample, func_vals = self._get_samples(nsamples, one_hot_channels)
        self._store_samples(samples, q_sample, func_vals)

        loss, mean, var = self._optimization_step(
            samples, q_sample, func_vals, one_hot_channels, weight_prior
        )

        if integral:
            return loss, mean, tf.sqrt(var / (nsamples - 1.0))

        return loss

    def train_on_stored_samples(self, batch_size: int, weight_prior: Callable = None):
        if self.stored_dataset is None:
            samples = tf.concat(self.stored_samples, axis=0)
            q_sample = tf.concat(self.stored_q_sample, axis=0)
            func_vals = tf.concat(self.stored_func_vals, axis=0)
            self.stored_dataset = (
                tf.data.Dataset.from_tensor_slices((samples, q_sample, func_vals))
                .shuffle(samples.shape[0])
                .batch(batch_size, drop_remainder=True)
            )

        one_hot_channels = self._get_channel_condition(batch_size)
        losses = []
        for ys, qs, fs in self.stored_dataset:
            loss, _, _ = self._optimization_step(
                ys, qs, fs, one_hot_channels, weight_prior
            )
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
        needs to be divided by (nsamples -1).

        Args:
            nsamples (int): Number of points on which the estimate is based on.
            weight_prior (Callable, optional): returns the prior weights. Defaults to None.

        Returns:
            tuple of 2 tf.tensors: mean and mc error

        """
        one_hot_channels = self._get_channel_condition(nsamples)
        integrands = self._get_integrand(nsamples, one_hot_channels, weight_prior)
        means, vars = tf.nn.moments(integrands, axes=[0])
        mean, var = tf.reduce_sum(means), tf.reduce_sum(vars)
        return mean, tf.sqrt(var / (nsamples - 1.0))

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

        Returns:
            true/test: tf.tensor of size (nsamples, 1) of sampled weights
            (samples: tf.tensor of size (nsamples, ndims) of sampled points)

        """
        one_hot_channels = self._get_channel_condition(nsamples)
        weights = []
        samples = []

        for i in range(self.n_channels):
            # Channel dependent flow sampling
            sample = self.flow.sample(nsamples, condition=one_hot_channels[:, :, i])
            logqi = self.flow.log_prob(sample, condition=one_hot_channels[:, :, i])

            # Check for analytic remappings
            if self.use_analytic_mappings:
                yi = self.mappings[i].inverse(sample)
                logqi += self.mappings[i].log_prob(yi)
            else:
                yi = sample

            # Define test probs
            qi = tf.math.exp(logqi)

            # Get multi-channel weights
            if self.use_weight_init:
                if weight_prior is not None:
                    init_weights = weight_prior(yi)
                    assert init_weights.shape[1] == self.n_channels
                else:
                    init_weights = (
                        1
                        / self.n_channels
                        * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                    )
                alphas = self.mcw_model([yi, init_weights])
            else:
                alphas = self.mcw_model(yi)

            # Get true integrand
            fi = alphas[:, i] * self._func(yi)
            weighti = fi / qi
            samples.append(yi)
            weights.append(weighti[..., None])

        weight = tf.concat(weights, axis=0)
        sample = tf.concat(samples, axis=0)

        if yield_samples:
            return weight, sample

        return sample

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
        """Save the networks."""
        self.flow.save_weights(path + "flow")
        if self.train_mcw:
            self.mcw_model.save_weights(path + "mcw")

    def load_weights(self, path: str):
        """Load the networks."""
        self.flow.load_weights(path + "flow")
        if self.train_mcw:
            self.mcw_model.load_weight(path + "mcw")
        print("Models loaded successfully")
