"""
Implementation of distributions for sampling and for importance sampling
"""

from typing import List

import numpy as np
import tensorflow as tf

from .base import Distribution
from ..utils import tfutils


class GaussianRing(Distribution):
    r"""Two-dimensional ring distribution.

    The probability in x-y-space is given
    by:

    .. math::

        P(x,y) = N * e^{-1/(2 * \sigma^2) (\sqrt{(x-x0)^2 + (y-y0)^2} - r0)^2},

    with some normalisation constant N:

    .. math::

        N = (2 * \pi * e^{-\mu^2/(2 * \sigma^2)} * \sigma^2 +
            \sqrt{\pi/2} * \mu * \sigma * (1 + Erf{\mu/(\sqrt{2} \ \sigma)}))^{-1}

    """

    def __init__(
        self,
        r0: float,
        sigma: float,
        **kwargs
    ):
        """
        Args:
            r0 (float): peak location - radius of ring.
            sigma (float): standard deviation - width of the ring.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape([1])

        self.r0 = tf.constant(r0, dtype=self._dtype)
        self.sigma = tf.constant(sigma, dtype=self._dtype)

    def _prob(self, x, condition):
        # Note: the condition is ignored.
        del condition
        
        # Define the norm
        w = tf.convert_to_tensor(
            1 / np.sqrt(2.0) * self.r0 / self.sigma, dtype=self._dtype
        )
        inv_norm = tf.convert_to_tensor(
            2
            * np.pi
            * (
                self.sigma ** 2 * tf.math.exp(-(w ** 2))
                + np.sqrt(np.pi / 2) * self.r0 * self.sigma * (1 + tf.math.erf(w))
            ),
            dtype=self._dtype,
        )

        # Define the base with exponent
        r = tf.math.sqrt(tfutils.sum_except_batch(x ** 2, num_batch_dims=1))
        log_base = -0.5 / (self.sigma ** 2) * ((r - self.r0) ** 2)

        prob = 1 / inv_norm * tf.math.exp(log_base)
        return prob

    def _get_samples(self, num_samples):
        len_sample = 0
        j = 0
        while len_sample < num_samples:
            if j == 0:
                eps = tf.random.normal((num_samples, *self._shape), dtype=self._dtype)
                r = self.sigma * eps + self.r0
                z = tf.random.uniform((num_samples, *self._shape), dtype=self._dtype)
                theta = 2 * np.pi * z
                sample = tf.concat([r, theta], axis=1)
                mask = r > 0
                sample = tf.boolean_mask(sample, mask)
                len_sample = sample.shape[0]
                j += 1
            else:
                eps = tf.random.normal((num_samples, *self._shape), dtype=self._dtype)
                r = self.sigma * eps + self.r0
                z = tf.random.uniform((num_samples, *self._shape), dtype=self._dtype)
                theta = 2 * np.pi * z
                sample2 = tf.concat([r, theta], axis=1)
                mask = r > 0
                sample2 = tf.boolean_mask(sample2, mask)
                sample = tf.concat([sample, sample2], axis=0)
                len_sample = sample.shape[0]
        sample = tf.random.shuffle(sample)
        sample = sample[:num_samples]
        return sample

    def _sample(self, num_samples, condition):
        del condition
        samples = self._get_samples(num_samples)
        return samples


class GaussianLine(Distribution):
    r"""Two-dimensional line distribution.

    The probability in x-y-space is given
    by:

    .. math::

        P(x,y) = N * e^{-1/(2 * \sigma_1^2) (x- \mu_1)^2} * e^{-1/(2 * \sigma_2^2) (x- \mu_2)^2},

    with some normalisation constant N:

    .. math::

        N = 1/Sqrt(2 * \pi * \sigma_1^2) \times 1/Sqrt(2 * \pi * \sigma_2^2)

    """

    def __init__(
        self,
        means: List[float],
        sigmas: List[float],
        alpha: float,
        **kwargs
    ):
        """
        Args:
            means (List[float]): peak locations.
            sigmas (List[float]): standard deviations.
            alpha (float): rotation anlge.
        """
        super().__init__(**kwargs)
        self._shape = tf.TensorShape([1])

        # check that its floats
        if any(isinstance(mean, int) for mean in means):
            means = [float(i) for i in means]

        if any(isinstance(sigma, int) for sigma in sigmas):
            sigmas = [float(i) for i in sigmas]

        self.means = tf.constant(means, dtype=self._dtype)
        self.sigmas = tf.constant(sigmas, dtype=self._dtype)
        self.alpha = tf.constant(alpha, dtype=self._dtype)

    def _rotate(self, x, alpha, inverse=False):
        # Rotate the line to some diagonal
        if inverse:
            x1 = x[:, :1] * tf.math.cos(alpha) + x[:, 1:] * tf.math.sin(alpha)
            x2 = -x[:, :1] * tf.math.sin(alpha) + x[:, 1:] * tf.math.cos(alpha)
            return tf.concat([x1, x2], axis=-1)

        y1 = x[:, :1] * tf.math.cos(alpha) - x[:, 1:] * tf.math.sin(alpha)
        y2 = x[:, :1] * tf.math.sin(alpha) + x[:, 1:] * tf.math.cos(alpha)
        return tf.concat([y1, y2], axis=-1)

    def _prob(self, x, condition):
        # Note: the condition is ignored.
        del condition
        
        # Define the norm
        norm = tf.convert_to_tensor(
            1
            / tf.math.sqrt(2.0 * np.pi * self.sigmas[0] ** 2)
            * 1
            / tf.math.sqrt(2.0 * np.pi * self.sigmas[1] ** 2),
            dtype=self._dtype,
        )

        # Rotate and define log_base
        x = self._rotate(x, self.alpha)
        log_base1 = -0.5 / (self.sigmas[0] ** 2) * ((x[:, 0] - self.means[0]) ** 2)
        log_base2 = -0.5 / (self.sigmas[1] ** 2) * ((x[:, 1] - self.means[1]) ** 2)

        prob = norm * tf.math.exp(log_base1 + log_base2)
        return prob

    def _get_samples(self, num_samples):
        eps1 = tf.random.normal((num_samples, *self._shape), dtype=self._dtype)
        y1 = self.sigmas[0] * eps1 + self.means[0]

        eps2 = tf.random.normal((num_samples, *self._shape), dtype=self._dtype)
        y2 = self.sigmas[1] * eps2 + self.means[1]

        sample = tf.concat([y1, y2], axis=1)
        sample = self._rotate(self, sample, self.alpha, inverse=True)
        return sample

    def _sample(self, num_samples, condition):
        del condition
        samples = self._get_samples(num_samples)
        return samples


class TwoChannelLineRing(Distribution):
    r"""Two-dimensional distribution as sum of

    P(x,y) = A * P_ring(x,y) + (1-A) * P_line(x,y)

    """

    def __init__(
        self,
        r0: float,
        sigma: float,
        means: List[float],
        sigmas: List[float],
        alpha: float,
        ratio: float = 0.5,
        **kwargs
    ):
        """
        Args:
            r0 (float): peak location - radius of ring.
            sigma (float): standard deviation - width of the ring.
            means (List[float]): peak locations.
            sigmas (List[float]): standard deviations.
            alpha (float): rotation anlge.
            ratio (float, optional): relative contribution A (0<A<1). Default is 0.5.
        """
        super().__init__(**kwargs)

        self.ratio = tf.constant(ratio, dtype=self._dtype)
        self.ring = GaussianRing(r0, sigma)
        self.line = GaussianLine(means, sigmas, alpha)

    def _prob(self, x, condition):
        # Note: the condition is ignored.
        del condition
        
        p1 = self.ring.prob(x)
        p2 = self.line.prob(x)
        p_tot = self.ratio * p1 + (1. - self.ratio) * p2
        return p_tot

    def _get_samples(self, num_samples):
        n1 = np.random.binomial(num_samples, self.ratio)
        n2 = num_samples - n1
        sample1 = self.ring.sample(n1)
        sample2 = self.line.sample(n2)
        sample = tf.concat([sample1, sample2], axis=0)
        return sample

    def _sample(self, num_samples, condition):
        del condition
        samples = self._get_samples(num_samples)
        return samples
