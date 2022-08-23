""" Implement the multi-channel flow integrator. """


import numpy as np
import tensorflow as tf
from typing import List, Union, Callable
import sys

from ..utils.divergences import Divergence
from ..mappings.base import Mapping
from ..distributions.base import Distribution

_EPSILON = 1e-16


class MultiChannelWeight:
    """Class implementing a network that learns
    the multi-channel weight only.
    """

    def __init__(
        self,
        func: Union[Callable, Distribution],
        mcw_model: tf.keras.Model,
        mappings: List[Mapping],
        optimizer: tf.keras.optimizers.Optimizer,
        use_weight_init: bool = True,
        n_channels: int = 2,
        loss_func: str = "variance",
        **kwargs,
    ):
        """
        Args:
            func (Union[Callable, Distribution]):
                Function to be integrated
            mcw_model (tf.keras.Model):
                Model which learns the multi-channel weights
            mappings (List(Mapping)): A list of analytic mappings applied
                to each channel of integration.
            optimizer (List[tf.keras.optimizers.Optimizer]):
                A list of optimizers for each of the two models
            n_channels (int): number of channels. Defaults to 2.
            loss_func (str, optional):
                The loss function to be minimized. Defaults to "chi2".
            kwargs: Additional arguments that need to be passed to the loss
        """
        self._dtype = tf.keras.backend.floatx()

        self._func = func
        self.mcw_model = mcw_model
        self.mcw_optimizer = optimizer
        self.mappings = mappings

        self.use_weight_init = use_weight_init
        
        if n_channels > 1:
            self.n_channels = n_channels
        else:
            raise ValueError(f"More than 1 channel expected. Use Integrator instead.")

        # Define the loss functions
        self.divergence = Divergence(train_mcw=True, **kwargs)
        self.loss_func = self.divergence(loss_func)
    
    @tf.function
    def _get_probs(self, samples: List[tf.Tensor], weight_prior: Callable = None):
        ps = []
        qs = []
        logqs = []
        means = []
        vars = []
        nsamples = samples[0].shape[0]

        for i in range(self.n_channels):
            # Channel dependent mapping
            logqi = self.mappings[i].log_prob(samples[i])
            qi = tf.math.exp(logqi)
            qs.append(qi[..., None])
            logqs.append(logqi[..., None])

            # Get multi-channel weights
            if self.use_weight_init:
                if weight_prior is not None:
                    init_weights = weight_prior(samples[i])
                    assert init_weights.shape[1] == self.n_channels
                else:
                    init_weights = 1 / self.n_channels * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                alphas = self.mcw_model([samples[i], init_weights])
            else:
                alphas = self.mcw_model(samples[i])

            # Get true integrand
            pi = alphas[:, i] * tf.abs(self._func(samples[i]))
            meani, vari = tf.nn.moments(pi / qi, axes=[0])
            pi = pi / meani
            ps.append(pi[..., None])
            means.append(meani)
            vars.append(vari)

        # Get concatenated stuff all in shape (nsamples, n_channels)
        p_true = tf.concat(ps, axis=-1)
        q_test = tf.concat(qs, axis=-1)
        
        logq = tf.concat(logqs, axis=-1)
        logp = tf.math.log(p_true + _EPSILON)
        
        # TODO: Understand why this where returns an error.
        # cond = tf.less(p_true, _EPSILON)
        # logp = tf.where(
        #     cond, tf.math.log(p_true), tf.math.log(p_true + _EPSILON)
        # )
        
        return p_true, q_test, logp, logq, sum(means), sum(vars)
    
    @tf.function
    def _get_integrand(self, nsamples: int, weight_prior: Callable = None):
        ps = []
        qs = []
        
        for i in range(self.n_channels):
            # Channel dependent sampling
            samples = self.mappings[i].sample(nsamples)
            qi = self.mappings[i].prob(samples)
            qs.append(qi[..., None])

            # Get multi-channel weights
            if self.use_weight_init:
                if weight_prior is not None:
                    init_weights = weight_prior(samples)
                    assert init_weights.shape[1] == self.n_channels
                else:
                    init_weights = 1 / self.n_channels * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                alphas = self.mcw_model([samples, init_weights])
            else:
                alphas = self.mcw_model(samples)

            # Get true integrand
            pi = alphas[:, i] * self._func.prob(samples)
            ps.append(pi[..., None])

        # Get concatenated stuff all in shape (nsamples, n_channels)
        p_true = tf.concat(ps, axis=-1)
        q_test = tf.concat(qs, axis=-1)
        
        return p_true/q_test

    @tf.function
    def train_one_step(self, nsamples: int, weight_prior: Callable = None, integral: bool = False):
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
        
        # Get the samples (outside of gradientape! More important for flows)
        samples = []
        for i in range(self.n_channels):
            # Channel dependent flow sampling
            sample = self.mappings[i].sample(nsamples)
            samples.append(sample)
            
        # Optimize the channel weight 
        with tf.GradientTape() as tape:
            p_true, q_test, logp, logq, mean, var = self._get_probs(samples, weight_prior)
            loss = self.loss_func(p_true, q_test, logp, logq)

        grads = tape.gradient(loss, self.mcw_model.trainable_weights)
        self.mcw_optimizer.apply_gradients(zip(grads, self.mcw_model.trainable_weights))

        if integral:
            return loss, mean, tf.sqrt(var / (nsamples - 1.0))

        return loss

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
        integrands = self._get_integrand(nsamples, weight_prior)
        means, vars = tf.nn.moments(integrands, axes=[0])
        mean, var = tf.reduce_sum(means), tf.reduce_sum(vars)
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
        weights = []
        samples = []
        
        for i in range(self.n_channels):
            # Channel dependent flow sampling
            samplei = self.mappings[i].sample(nsamples)
            qi = self.mappings[i].prob(samplei)

            # Get multi-channel weights
            if self.use_weight_init:
                init_weights = 1 / self.n_channels * tf.ones((nsamples, self.n_channels), dtype=self._dtype)
                alphas = self.mcw_model([samplei, init_weights])
            else:
                alphas = self.mcw_model(samplei)

            # Get true integrand
            fi = alphas[:, i] * self._func(samplei)
            weighti = fi / qi
            samples.append(samplei)
            weights.append(weighti[...,None])
            
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
        """Save the network."""
        self.mcw_model.save_weights(path)

    def load_weights(self, path: str):
        """Load the network."""
        self.mcw_model.load_weights(path)
        print("Model loaded successfully")
