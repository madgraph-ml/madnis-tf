"""
Implementation of distributions for sampling and for importance sampling
"""

from typing import List

import numpy as np
import tensorflow as tf

from .base import Mapping
from ..distributions.uniform import StandardUniform
from ..utils import tfutils


class CauchyRingMap(Mapping):
    """Two-dimensional Cauchy ring mapping."""

    def __init__(self, r0: float = 0.0, gamma: float = 1.0, **kwargs):
        """
        Args:
            r0: float, location of the peak - radius of the ring
            gamma: float, scale parameter (FWHM) - width of the ring
        """
        super().__init__(StandardUniform([2]), **kwargs)
        self._shape = tf.TensorShape([2])

        self.r0 = tf.constant(r0, dtype=self._dtype)
        self.gamma = tf.constant(gamma, dtype=self._dtype)
        self.tf_pi = tf.constant(np.pi, dtype=self._dtype)
        self.prefactor = tf.constant(1 / (np.pi * self.gamma), dtype=self._dtype)
        self.c0 = tf.math.atan(self.r0 / self.gamma) / self.tf_pi

    def _to_sphericals(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        r = tf.math.sqrt(x1 ** 2 + x2 ** 2)
        phi = tf.math.atan2(x2, x1) + self.tf_pi
        return r, phi

    def _to_cartesian(self, rvec):
        r, phi = rvec[:, :1], rvec[:, 1:] - self.tf_pi
        x1 = r * tf.math.cos(phi)
        x2 = r * tf.math.sin(phi)
        return x1, x2

    def _forward(self, x, condition):
        """The forward pass of the mapping"""
        # Note: the condition is ignored.
        del condition

        # Map onto sphericals
        r, phi = self._to_sphericals(x)

        # Map onto unit-hypercube
        c1 = 2 / (1 + 2 * self.c0)
        z1 = c1 / self.tf_pi * tf.math.atan((r - self.r0) / self.gamma) + self.c0 * c1
        z2 = phi / (2 * self.tf_pi)

        logdet = self.log_det(x)
        return tf.concat([z1, z2], axis=1), logdet

    def _inverse(self, z, condition):
        """The inverse pass of the mapping"""
        # Note: the condition is ignored.
        del condition

        z1 = z[:, :1]
        z2 = z[:, 1:]

        # Map onto sphericals
        r = self.r0 + self.gamma * tf.math.tan(self.tf_pi * (z1 / 2 + self.c0 * (z1 - 1)))
        phi = 2 * self.tf_pi * z2
        rvec = tf.concat([r, phi], axis=1)

        # Map to cartesians
        x1, x2 = self._to_cartesian(rvec)
        logdet = self.log_det(z, inverse=True)
        return tf.concat([x1, x2], axis=1), logdet

    def _det(self, x_or_z, condition, inverse=False):
        # Note: the condition is ignored.
        del condition

        if inverse:
            # the derivative of the inverse pass (dF^{-1}/dz)
            # first part of the pass
            z1 = x_or_z[:, :1]
            d11 = (
                (0.5 + self.c0)
                * self.tf_pi
                * self.gamma
                * 1
                / (tf.math.cos(self.tf_pi * (z1 / 2 + self.c0 * (z1 - 1))) ** 2)
            )
            d22 = 2 * self.tf_pi
            det_peak = d11 * d22

            # map onto cartesian
            r = self.r0 + self.gamma * tf.math.tan(
                self.tf_pi * (z1 / 2 + self.c0 * (z1 - 1))
            )
            return tf.squeeze(det_peak * r)
        else:
            # the derivative of the forward pass (dF/dx)
            # first part of the pass
            # x1 = x_or_z[:, :1]
            # x2 = x_or_z[:, 1:]
            # det_pol = 1 / tf.math.sqrt(x1 ** 2 + x2 ** 2)

            # map out the peaks
            r, _ = self._to_sphericals(x_or_z)
            d11 = (
                2
                * self.gamma
                / self.tf_pi
                / ((r - self.r0) ** 2 + self.gamma**2)
                / (1 + 2 * self.c0)
            )
            d22 = 1 / (2 * self.tf_pi)
            det_peak = d11 * d22
            return tf.squeeze(det_peak / r)

    def _sample(self, num_samples, condition):
        z_values = self.base_dist.sample(num_samples, condition)

        # # Sample from quantile
        # if condition is not None:
        #     # Merge the condition dimension with sample dimension in order to call log_prob.
        #     z_values = tfutils.merge_leading_dims(z_values, num_dims=2)
        #     condition = tfutils.repeat_rows(condition, num_reps=num_samples)
        #     assert z_values.shape[0] == condition.shape[0]

        sample, _ = self.inverse(z_values, condition)

        # if condition is not None:
        #     # Split the context dimension from sample dimension.
        #     sample = tfutils.split_leading_dim(sample, shape=[-1, num_samples])

        return sample


class CauchyLineMap(Mapping):
    """Two-dimensional Cauchy line mapping."""

    def __init__(self, means: List[float], gammas: List[float], alpha: float, **kwargs):
        """
        Args:
            means (List[float]): peak locations.
            gammas (List[float]): scale parameters (FWHM).
            alpha (float): rotation anlge.
        """
        super().__init__(StandardUniform([2]), **kwargs)
        self._shape = tf.TensorShape([2])

        # check that its floats
        if any(isinstance(mean, int) for mean in means):
            means = [float(i) for i in means]

        if any(isinstance(gamma, int) for gamma in gammas):
            gammas = [float(i) for i in gammas]

        self.means = tf.constant(means, dtype=self._dtype)
        self.gammas = tf.constant(gammas, dtype=self._dtype)
        self.alpha = tf.constant(alpha, dtype=self._dtype)
        self.tf_pi = tf.constant(np.pi, dtype=self._dtype)

    def _rotate(self, x, alpha, inverse=False):
        # Rotate the line to some diagonal
        if inverse:
            x1 = x[:, :1] * tf.math.cos(alpha) + x[:, 1:] * tf.math.sin(alpha)
            x2 = -x[:, :1] * tf.math.sin(alpha) + x[:, 1:] * tf.math.cos(alpha)
            return tf.concat([x1, x2], axis=-1)

        y1 = x[:, :1] * tf.math.cos(alpha) - x[:, 1:] * tf.math.sin(alpha)
        y2 = x[:, :1] * tf.math.sin(alpha) + x[:, 1:] * tf.math.cos(alpha)
        return tf.concat([y1, y2], axis=-1)

    def _forward(self, x, condition):
        """The forward pass of the mapping"""
        # Note: the condition is ignored.
        del condition

        # Rotate to be parallel with x-axis
        y = self._rotate(x, self.alpha)
        y1, y2 = tf.split(y, 2, axis=-1)

        # Map onto unit-hypercube
        z1 = 1 / self.tf_pi * tf.math.atan((y1 - self.means[0]) / self.gammas[0]) + 0.5
        z2 = 1 / self.tf_pi * tf.math.atan((y2 - self.means[1]) / self.gammas[1]) + 0.5
        logdet = self.log_det(x)

        return tf.concat([z1, z2], axis=1), logdet

    def _inverse(self, z, condition):
        """The inverse pass of the mapping"""
        # Note: the condition is ignored.
        del condition

        z1, z2 = tf.split(z, 2, axis=-1)

        # Map out peaks
        y1 = self.means[0] + self.gammas[0] * tf.math.tan(self.tf_pi * (z1 - 0.5))
        y2 = self.means[1] + self.gammas[1] * tf.math.tan(self.tf_pi * (z2 - 0.5))
        y = tf.concat([y1, y2], axis=1)

        # Rotate back
        x = self._rotate(y, self.alpha, inverse=True)
        logdet = self.log_det(z, inverse=True)

        return x, logdet

    def _det(self, x_or_z, condition, inverse=False):
        # Note: the condition is ignored.
        if inverse:
            # the derivative of the inverse pass (dF^{-1}/dz)
            z1, z2 = tf.split(x_or_z, 2, axis=-1)
            d11 = self.tf_pi * self.gammas[0] * 1 / (tf.math.sin(self.tf_pi * z1) ** 2)
            d22 = self.tf_pi * self.gammas[1] * 1 / (tf.math.sin(self.tf_pi * z2) ** 2)
            return tf.squeeze(d11 * d22)
        else:
            # the derivative of the forward pass (dF/dx)
            y = self._rotate(x_or_z, self.alpha)
            y1, y2 = tf.split(y, 2, axis=-1)
            d11 = self.gammas[0] / ((y1 - self.means[0]) ** 2 + self.gammas[0] ** 2)
            d22 = self.gammas[1] / ((y2 - self.means[1]) ** 2 + self.gammas[1] ** 2)
            return tf.squeeze(d11 * d22 / (self.tf_pi ** 2))

    def _sample(self, num_samples, condition):
        z_values = self.base_dist.sample(num_samples, condition)

        # # Sample from quantile
        # if condition is not None:
        #     # Merge the condition dimension with sample dimension in order to call log_prob.
        #     z_values = tfutils.merge_leading_dims(z_values, num_dims=2)
        #     condition = tfutils.repeat_rows(condition, num_reps=num_samples)
        #     assert z_values.shape[0] == condition.shape[0]

        sample, _ = self.inverse(z_values, condition)

        # if condition is not None:
        #     # Split the context dimension from sample dimension.
        #     sample = tfutils.split_leading_dim(sample, shape=[-1, num_samples])

        return sample
