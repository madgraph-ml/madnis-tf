import numpy as np
import tensorflow as tf

from ..distributions.uniform import StandardUniform
from .base import Mapping

class TwoParticlePhasespaceA(Mapping):

    def __init__(
        self,
        sqrt_s_min: float = 50.,
        e_beam: float = 6500.,
        s_mass: float = 0.,
        s_gamma: float = 0.,
        nu: float = 0.95,
        **kwargs
    ):
        super().__init__(StandardUniform([4]), **kwargs)
        self._shape = tf.TensorShape([4])

        self.e_beam = tf.constant(e_beam, dtype=self._dtype)
        self.s_min = tf.constant(sqrt_s_min**2, dtype=self._dtype)
        self.s_max = tf.constant(4*e_beam**2, dtype=self._dtype)
        self.tf_pi = tf.constant(np.pi, dtype=self._dtype)

        self.massless = s_mass == 0.
        if self.massless:
            self.nu = nu
        else:
            self.y1 = tf.math.atan((self.s_min - s_mass**2) / (s_mass*s_gamma))
            self.y2 = tf.math.atan((self.s_max - s_mass**2) / (s_mass*s_gamma))
            self.s_mass = tf.constant(s_mass, dtype=self._dtype)
            self.s_gamma = tf.constant(s_gamma, dtype=self._dtype)

    def _sublogdet(self, s, r2, r3, r4):
        del r4
        return tf.math.log(
            self.tf_pi * (s/self.s_max)**(-2 * r2) *
            (-r3*s + (-1 + r3) * (s/self.s_max)**(2 * r2) * self.s_max) *
            (s - r3 * s + r3 * (s/self.s_max)**(2 * r2) * self.s_max) *
            tf.math.log(s/self.s_max) / (4 * self.s_max)
        )

    def _forward(self, p, condition):
        # Note: the condition is ignored.
        del condition

        px1, py1, pz1, pz2 = tf.unstack(p, axis=-1)
        e1 = tf.math.sqrt(px1**2 + py1**2 + pz1**2)
        e2 = tf.math.sqrt(px1**2 + py1**2 + pz2**2)
        pz_tot = pz1 + pz2
        e_tot = e1 + e2
        x1 = (e_tot + pz_tot) / (2*self.e_beam)
        x2 = (e_tot - pz_tot) / (2*self.e_beam)
        s = self.s_max * x1 * x2
        r2 = tf.math.log(x1) / tf.math.log(s / self.s_max)
        r3 = (pz1/self.e_beam + x2) / (x1 + x2)
        r4 = tf.math.atan2(py1, px1) / (2*self.tf_pi) + 0.5

        if self.massless:
            r1 = (
                (s**(1-self.nu) - self.s_min**(1-self.nu))
                / (self.s_max**(1-self.nu) - self.s_min**(1-self.nu))
            )
            logdet = - tf.math.log(
                (1 - self.nu) /
                (s**self.nu * (self.s_max**(1-self.nu) - self.s_min**(1-self.nu)))
            )
        else:
            r1 = (
                (tf.math.atan((s - self.s_mass**2) / (self.s_mass * self.s_gamma)) - self.y1)
                / (self.y2 - self.y1)
            )
            logdet = - tf.math.log(
                self.s_mass * self.s_gamma / (
                    (self.y2 - self.y1) *
                    ((s - self.s_mass**2)**2 + self.s_mass**2 * self.s_gamma**2)
                )
            )

        r = tf.stack((r1, r2, r3, r4), axis=-1)
        return r, -logdet - self._sublogdet(s, r2, r3, r4)

    def _inverse(self, r, condition):
        # Note: the condition is ignored.
        del condition

        r1, r2, r3, r4 = tf.unstack(r, axis=-1)
        # Mapping of s as defined in https://arxiv.org/pdf/hep-ph/0206070.pdf (p. 17)
        if self.massless:
            s = (
                r1 * self.s_max**(1-self.nu) +
                (1 - r1) * self.s_min**(1-self.nu)
            ) ** (1 / (1-self.nu))
            logdet = - tf.math.log(
                (1 - self.nu) /
                (s**self.nu * (self.s_max**(1-self.nu) - self.s_min**(1-self.nu)))
            )
        else:
            s = (
                self.s_mass * self.s_gamma * tf.math.tan(self.y1 + (self.y2 - self.y1)*r1)
                + self.s_mass**2
            )
            logdet = - tf.math.log(
                self.s_mass * self.s_gamma / (
                    (self.y2 - self.y1) *
                    ((s - self.s_mass**2)**2 + self.s_mass**2 * self.s_gamma**2)
                )
            )
        x1 = (s / self.s_max)**r2
        x2 = (s / self.s_max)**(1-r2)
        pz1 = self.e_beam * (x1*r3 + x2*(r3-1))
        pz2 = self.e_beam * (x1*(1-r3) - x2*r3)
        pt = tf.math.sqrt(s*r3*(1-r3))
        phi = 2 * self.tf_pi * (r4 - 0.5)
        px1 = pt * tf.math.cos(phi)
        py1 = pt * tf.math.sin(phi)
        p = tf.stack((px1, py1, pz1, pz2), axis=-1)
        return p, logdet + self._sublogdet(s, r2, r3, r4)

    def _log_det(self, x_or_z, condition, inverse=False):
        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            _, logdet = self._inverse(x_or_z, condition)
            return logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            _, logdet = self._forward(x_or_z, condition)
            return logdet

    def _sample(self, num_samples, condition):
        r_values = self.base_dist.sample(num_samples, condition)
        sample, _ = self.inverse(r_values, condition)
        return sample


class TwoParticlePhasespaceB(Mapping):

    def __init__(
        self,
        sqrt_s_min: float = 50.,
        e_beam: float = 6500.,
        s_mass: float = 0.,
        s_gamma: float = 0.,
        nu: float = 0.95,
        **kwargs
    ):
        super().__init__(StandardUniform([4]), **kwargs)
        self._shape = tf.TensorShape([4])

        self.e_beam = tf.constant(e_beam, dtype=self._dtype)
        self.s_min = tf.constant(sqrt_s_min**2, dtype=self._dtype)
        self.s_max = tf.constant(4*e_beam**2, dtype=self._dtype)
        self.tf_pi = tf.constant(np.pi, dtype=self._dtype)

        self.massless = s_mass == 0.
        if self.massless:
            self.nu = nu
        else:
            self.y1 = tf.math.atan((self.s_min - s_mass**2) / (s_mass*s_gamma))
            self.y2 = tf.math.atan((self.s_max - s_mass**2) / (s_mass*s_gamma))
            self.s_mass = tf.constant(s_mass, dtype=self._dtype)
            self.s_gamma = tf.constant(s_gamma, dtype=self._dtype)

    def _sublogdet(self, s):
        det = 4 * self.tf_pi * tf.math.log(self.s_max/s) / self.s_max
        return tf.math.log(det)

    def _forward(self, p, condition):
        # Note: the condition is ignored.
        del condition

        x1, x2, costheta, phi = tf.unstack(p, axis=-1)
        s = self.s_max * x1 * x2
        r2 = tf.math.log(x1) / tf.math.log(s / self.s_max)
        r3 = (costheta + 1)/2
        r4 = phi / (2*self.tf_pi) + 0.5

        if self.massless:
            r1 = (
                (s**(1-self.nu) - self.s_min**(1-self.nu))
                / (self.s_max**(1-self.nu) - self.s_min**(1-self.nu))
            )
            logdet = tf.math.log(
                (1 - self.nu) /
                (s**self.nu * (self.s_max**(1-self.nu) - self.s_min**(1-self.nu)))
            )
        else:
            r1 = (
                (tf.math.atan((s - self.s_mass**2) / (self.s_mass * self.s_gamma)) - self.y1)
                / (self.y2 - self.y1)
            )
            logdet = tf.math.log(
                self.s_mass * self.s_gamma / (
                    (self.y2 - self.y1) *
                    ((s - self.s_mass**2)**2 + self.s_mass**2 * self.s_gamma**2)
                )
            )

        r = tf.stack((r1, r2, r3, r4), axis=-1)
        return r, logdet - self._sublogdet(s)

    def _inverse(self, r, condition):
        # Note: the condition is ignored.
        del condition

        r1, r2, r3, r4 = tf.unstack(r, axis=-1)
        # Mapping of s as defined in https://arxiv.org/pdf/hep-ph/0206070.pdf (p. 17)
        if self.massless:
            s = (
                r1 * self.s_max**(1-self.nu) +
                (1 - r1) * self.s_min**(1-self.nu)
            ) ** (1 / (1-self.nu))
            logdet = tf.math.log(
                (1 - self.nu) /
                (s**self.nu * (self.s_max**(1-self.nu) - self.s_min**(1-self.nu)))
            )
        else:
            s = (
                self.s_mass * self.s_gamma * tf.math.tan(self.y1 + (self.y2 - self.y1)*r1)
                + self.s_mass**2
            )
            logdet = tf.math.log(
                self.s_mass * self.s_gamma / (
                    (self.y2 - self.y1) *
                    ((s - self.s_mass**2)**2 + self.s_mass**2 * self.s_gamma**2)
                )
            )
        x1 = (s / self.s_max)**r2
        x2 = (s / self.s_max)**(1-r2)
        costheta = 2 * r3 - 1
        phi = 2 * self.tf_pi * (r4 - 0.5)

        p = tf.stack((x1, x2, costheta, phi), axis=-1)
        return p, self._sublogdet(s) - logdet

    def _log_det(self, x_or_z, condition, inverse=False):
        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            _, logdet = self._inverse(x_or_z, condition)
            return logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            _, logdet = self._forward(x_or_z, condition)
            return logdet

    def _sample(self, num_samples, condition):
        r_values = self.base_dist.sample(num_samples, condition)
        sample, _ = self.inverse(r_values, condition)
        return sample


class TwoParticlePhasespaceFlatB(Mapping):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(StandardUniform([4]), **kwargs)
        self._shape = tf.TensorShape([4])
        self.tf_pi = tf.constant(np.pi, dtype=self._dtype)

    def _forward(self, p, condition):
        # Note: the condition is ignored.
        del condition

        x1, x2, costheta, phi = tf.unstack(p, axis=-1)

        r1 = x1
        r2 = x2
        r3 = (costheta + 1)/2
        r4 = phi / (2*self.tf_pi) + 0.5

        logdet = tf.math.log(4 * self.tf_pi)
        r = tf.stack((r1, r2, r3, r4), axis=-1)
        return r, -logdet

    def _inverse(self, r, condition):
        # Note: the condition is ignored.
        del condition

        r1, r2, r3, r4 = tf.unstack(r, axis=-1)

        x1 = r1
        x2 = r2
        costheta = 2 * r3 - 1
        phi = 2 * self.tf_pi * (r4 - 0.5)

        logdet = tf.math.log(4 * self.tf_pi)
        p = tf.stack((x1, x2, costheta, phi), axis=-1)
        return p, logdet

    def _log_det(self, x_or_z, condition, inverse=False):
        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            _, logdet = self._inverse(x_or_z, condition)
            return logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            _, logdet = self._forward(x_or_z, condition)
            return logdet

    def _sample(self, num_samples, condition):
        r_values = self.base_dist.sample(num_samples, condition)
        sample, _ = self.inverse(r_values, condition)
        return sample
