"""
Implementation of distributions for sampling and for importance sampling
"""

from typing import List

import numpy as np
import tensorflow as tf

from .base import Distribution
from ..utils import tfutils


class Camel(Distribution):
    """One-dimensional Camel distribution."""

    def __init__(
        self,
        means: List[float],
        stds: List[float],
        peak_ratios: List[float] = None,
        **kwargs
    ):
        """
        Args:
            means (List[float]): peak locations.
            stds (List[float]): standard deviations.
            peak_ratios (List[float], optional): relative peak contribution. Defaults to None.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape([1])

        # check that its floats
        if any(isinstance(mean, int) for mean in means):
            means = [float(i) for i in means]

        if any(isinstance(std, int) for std in stds):
            stds = [float(i) for i in stds]

        self.means = tf.constant(means, dtype=self._dtype)
        self.log_stds = tf.constant(np.log(stds), dtype=self._dtype)

        self.npeaks = len(means)

        if peak_ratios is None:
            fracs = 1 / self.npeaks
            self.ratios = tf.constant([fracs] * self.npeaks, dtype=self._dtype)
        else:
            if len(peak_ratios) != self.npeaks:
                raise ValueError("Length of lists do not match.")
            self.ratios = tf.constant(
                peak_ratios / tf.reduce_sum(peak_ratios), dtype=self._dtype
            )

    def _prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        prob = 0
        for i in range(self.npeaks):
            log_norm = tf.convert_to_tensor(
                -0.5 * np.log(2 * np.pi) - self.log_stds[i], dtype=self._dtype
            )
            log_base = (
                -0.5 * tf.math.exp(-2 * self.log_stds[i]) * ((x - self.means[i]) ** 2)
            )
            prob += self.ratios[i] * tf.math.exp(log_base + log_norm)
        return tfutils.sum_except_batch(prob, num_batch_dims=1)

    def _sample(self, num_samples, condition):
        if condition is None:
            rand_draw = np.random.binomial(num_samples, self.ratios[1:])
            n0 = num_samples - np.sum(rand_draw)
            n_samples = [n0] + [*rand_draw]
            samples = [
                self.means[i]
                + tf.math.exp(self.log_stds[i])
                * tf.random.normal((n_samples[i], *self._shape), dtype=self._dtype)
                for i in range(self.npeaks)
            ]
            return tf.concat(samples, axis=0)
        else:
            # The value of the condition is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            rand_draw = np.random.binomial(
                condition_size * num_samples, self.ratios[1:]
            )
            n0 = condition_size * num_samples - np.sum(rand_draw)
            n_samples = [n0] + [*rand_draw]
            samples = [
                self.means[i]
                + tf.math.exp(self.log_stds[i])
                * tf.random.normal((n_samples[i], *self._shape), dtype=self._dtype)
                for i in range(self.npeaks)
            ]
            samples = tf.concat(samples, axis=0)
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])


class CuttedCamel(Distribution):
    """One-dimensional cutted Camel distribution."""

    def __init__(
        self,
        means: List[float],
        stds: List[float],
        peak_ratios: List[float] = None,
        cut: float = None,
        **kwargs
    ):
        """
        Args:
            means: list containing the means, with shape [mean_0, mean_1,..].
            stds: list containing the standard deviations, with shape [var_0, var_1,..].
            peak_ratios: None, list or float, with the relative peak contribution, with shape [amp_0, amp_1,..]
            cut: None or float, use a lower cut on the distribution
        """
        super().__init__(**kwargs)

        # check that its floats
        if any(isinstance(mean, int) for mean in means):
            means = [float(i) for i in means]

        if any(isinstance(std, int) for std in stds):
            stds = [float(i) for i in stds]

        if cut is not None:
            self.cut = tf.constant(cut, dtype=self._dtype)
        else:
            self.cut = tf.constant(-np.inf, dtype=self._dtype)

        self.means = tf.constant(means, dtype=self._dtype)
        self.log_stds = tf.constant(np.log(stds), dtype=self._dtype)

        self.npeaks = len(means)

        if peak_ratios is None:
            fracs = 1 / self.npeaks
            self.ratios = tf.constant([fracs] * self.npeaks, dtype=self._dtype)
        else:
            if len(peak_ratios) != self.npeaks:
                raise ValueError("Length of lists do not match.")
            self.ratios = tf.constant(
                peak_ratios / tf.reduce_sum(peak_ratios), dtype=self._dtype
            )

    def _prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        prob = 0
        for i in range(self.npeaks):
            log_norm = tf.convert_to_tensor(
                -0.5 * np.log(2 * np.pi) - self.log_stds[i], dtype=self._dtype
            )
            log_base = (
                -0.5 * tf.math.exp(-2 * self.log_stds[i]) * ((x - self.means[i]) ** 2)
            )
            prob += self.ratios[i] * tf.math.exp(log_base + log_norm)
        prob = tf.where(x > self.cut, prob, 0)
        return tfutils.sum_except_batch(prob, num_batch_dims=1)

    def _get_samples(self, num_samples):
        rand_draw = np.random.binomial(num_samples, self.ratios[1:])
        len_sample = 0
        j = 0
        while len_sample < num_samples:
            if j == 0:
                n0 = num_samples - np.sum(rand_draw)
                n_samples = [n0] + [*rand_draw]
                samples = [
                    self.means[i]
                    + tf.math.exp(self.log_stds[i])
                    * tf.random.normal((n_samples[i], *self._shape), dtype=self._dtype)
                    for i in range(self.npeaks)
                ]
                sample = tf.concat(samples, axis=0)
                mask = sample > self._cut
                sample = tf.boolean_mask(sample, mask)
                len_sample = sample.shape[0]
                j += 1
            else:
                n0 = num_samples - np.sum(rand_draw)
                n_samples = [n0] + [*rand_draw]
                samples2 = [
                    self.means[i]
                    + tf.math.exp(self.log_stds[i])
                    * tf.random.normal((n_samples[i], *self._shape), dtype=self._dtype)
                    for i in range(self.npeaks)
                ]
                sample2 = tf.concat(samples2, axis=0)
                mask = sample2 > self._cut
                sample2 = tf.boolean_mask(sample2, mask)
                sample = tf.concat([sample, sample2], axis=0)
                len_sample = sample.shape[0]
        sample = tf.random.shuffle(sample)
        sample = sample[:num_samples]
        return sample

    def _sample(self, num_samples, condition):
        if condition is None:
            samples = self._get_samples(num_samples)
            return samples

        else:
            # The value of the condition is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            samples = self._get_samples(num_samples * condition_size)
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])


class MultiDimCamel(Distribution):
    """n-dimensional Camel distribution with m-peaks"""

    def __init__(
        self,
        means: List[tf.Tensor],
        stds: List[float],
        dims: int,
        peak_ratios: List[float] = None,
        **kwargs
    ):
        """
        Args:
            means (List[tf.Tensor]): peak locations.
            stds (List[float]): standard deviations.
            dims (int): dimensionality of the distributions
            peak_ratios (List[float], optional): relative peak contribution. Defaults to None.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape([dims])

        # check that its a proper tensor
        if not all(isinstance(mean, tf.Tensor) for mean in means):
            raise NotImplementedError("Means are not proper tensors")

        # check that its floats
        if any(isinstance(std, int) for std in stds):
            stds = [float(i) for i in stds]

        self.means = means
        self.log_stds = tf.constant(np.log(stds), dtype=self._dtype)

        self.npeaks = len(means)

        if peak_ratios is None:
            fracs = 1 / self.npeaks
            self.ratios = tf.constant([fracs] * self.npeaks, dtype=self._dtype)
        else:
            if len(peak_ratios) != self.npeaks:
                raise ValueError("Length of lists do not match.")
            self.ratios = tf.constant(
                peak_ratios / tf.reduce_sum(peak_ratios), dtype=self._dtype
            )

        # Define the norm for all peaks
        self.log_norms = tf.constant(
            -1 / 2 * np.log(2 * np.pi) - self.log_stds, dtype=self._dtype
        )

    def _prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        prob = 0
        for i in range(self.npeaks):
            log_norm = self.log_norms[i]
            log_base = (
                -0.5 * tf.math.exp(-2 * self.log_stds[i]) * ((x - self.means[i]) ** 2)
            )
            prob += self.ratios[i] * tf.math.exp(
                tfutils.sum_except_batch(log_base + log_norm, num_batch_dims=1)
            )
        return prob

    def _sample(self, num_samples, condition):
        if condition is None:
            rand_draw = np.random.binomial(num_samples, self.ratios[1:])
            n0 = num_samples - np.sum(rand_draw)
            n_samples = [n0] + [*rand_draw]
            samples = [
                self.means[i]
                + tf.math.exp(self.log_stds[i])
                * tf.random.normal((n_samples[i], *self._shape), dtype=self._dtype)
                for i in range(self.npeaks)
            ]
            return tf.concat(samples, axis=0)
        else:
            # The value of the context is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            rand_draw = np.random.binomial(
                condition_size * num_samples, self.ratios[1:]
            )
            n0 = condition_size * num_samples - np.sum(rand_draw)
            n_samples = [n0] + [*rand_draw]
            samples = [
                self.means[i]
                + tf.math.exp(self.log_stds[i])
                * tf.random.normal((n_samples[i], *self._shape), dtype=self._dtype)
                for i in range(self.npeaks)
            ]
            samples = tf.concat(samples, axis=0)
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])


class NormalizedMultiDimCamel(Distribution):
    """n-dimensional Camel distribution with m-peaks
    which is normalized to 1 on the unit-hypercube [0,1]^n
    """

    def __init__(
        self,
        means: List[tf.Tensor],
        stds: List[float],
        dims: int,
        peak_ratios: List[float] = None,
        **kwargs
    ):
        """
        Args:
            means (List[tf.Tensor]): peak locations.
            stds (List[float]): standard deviations.
            dims (int): dimensionality of the distributions
            peak_ratios (List[float], optional): relative peak contribution. Defaults to None.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape([dims])

        # check that its a proper tensor
        if not all(isinstance(mean, tf.Tensor) for mean in means):
            raise NotImplementedError("Means are not proper tensors")

        # check that its floats
        if any(isinstance(std, int) for std in stds):
            stds = [float(i) for i in stds]

        self.means = means
        self.stds = tf.constant(stds, dtype=self._dtype)
        self.log_stds = tf.constant(np.log(stds), dtype=self._dtype)

        self.npeaks = len(means)

        if peak_ratios is None:
            fracs = 1 / self.npeaks
            self.ratios = tf.constant([fracs] * self.npeaks, dtype=self._dtype)
        else:
            if len(peak_ratios) != self.npeaks:
                raise ValueError("Length of lists do not match.")
            self.ratios = tf.constant(
                peak_ratios / tf.reduce_sum(peak_ratios), dtype=self._dtype
            )

        # Define the norm for all peaks
        self.log_norms = tf.constant(
            -1 / 2 * np.log(2 * np.pi) - self.log_stds, dtype=self._dtype
        )

    def _norm(self):
        dims = self._shape[0]
        norms = tf.pow(
            0.5
            * (
                tf.math.erf((1 - self.means) / tf.sqrt(2) / self.stds)
                + tf.math.erf((self.means) / tf.sqrt(2) / self.stds)
            ),
            dims,
        )
        return tf.reduce_sum(self.ratios * norms)

    def _prob(self, x, condition):
        # Note: the condition is ignored.
        del condition

        prob = 0
        for i in range(self.npeaks):
            log_norm = self.log_norms[i]
            log_base = (
                -0.5 * tf.math.exp(-2 * self.log_stds[i]) * ((x - self.means[i]) ** 2)
            )
            prob += self.ratios[i] * tf.math.exp(
                tfutils.sum_except_batch(log_base + log_norm, num_batch_dims=1)
            )
        norm = self._norm()
        return prob / norm

    def _sample(self, num_samples, condition):
        if condition is None:
            rand_draw = np.random.binomial(num_samples, self.ratios[1:])
            n0 = num_samples - np.sum(rand_draw)
            n_samples = [n0] + [*rand_draw]
            samples = [
                self.means[i]
                + tf.math.exp(self.log_stds[i])
                * tf.random.normal((n_samples[i], *self._shape), dtype=self._dtype)
                for i in range(self.npeaks)
            ]
            return tf.concat(samples, axis=0)
        else:
            # The value of the context is ignored, only its size is taken into account.
            condition_size = condition.shape[0]
            rand_draw = np.random.binomial(
                condition_size * num_samples, self.ratios[1:]
            )
            n0 = condition_size * num_samples - np.sum(rand_draw)
            n_samples = [n0] + [*rand_draw]
            samples = [
                self.means[i]
                + tf.math.exp(self.log_stds[i])
                * tf.random.normal((n_samples[i], *self._shape), dtype=self._dtype)
                for i in range(self.npeaks)
            ]
            samples = tf.concat(samples, axis=0)
            return tfutils.split_leading_dim(samples, [condition_size, num_samples])
