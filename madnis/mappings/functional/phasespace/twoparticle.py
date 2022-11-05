""" Implement two-particle mappings.
    Bases on the mappings described in
    https://freidok.uni-freiburg.de/data/154629"""

import tensorflow as tf
import math as m
from .helper import kaellen

# TODO: Continue...
def two_particle_phasespace(
    r: tf.Tensor,
    s: tf.Tensor,
    m1: tf.Tensor = None,
    m2: tf.Tensor = None,
):
    """Two particle phase space
    parametrized in terms of the solid angle, i.e

        dPhi_{2} = V_{2}/(4 Pi) dcostheta dphi,
                 = V_{2} dr1 dr2,

    where V_{2} is the two-particle phase-space volume, given by

        V_{2} = 1/(8 Pi) * \lambda(s, m1^2, m2^2)^(1/2) / s
              = 1/(8 Pi), if [m1 = m2 = 0]

    with the Kaellen function `\lambda`.

    Args:
        r (tf.Tensor): random numbers input.
        s (tf.Tensor): squared CM energy.
        m1 (tf.Tensor, optional): mass of particle 1. Defaults to None.
        m2 (tf.Tensor, optional): mass of particle 2. Defaults to None.
    """

    if m1 is None:
        m1 = tf.constant(0.0, dtype=r.dtype)
    if m2 is None:
        m2 = tf.constant(0.0, dtype=r.dtype)

    tf_pi = tf.constant(m.pi, dtype=r.dtype)

    # Define phase-space volume
    log_volume = (
        0.5 * tf.math.log(kaellen(s, m1**2, m2**2))
        - tf.math.log(s)
        - tf.math.log(8 * tf_pi)
    )

    # do the mapping (linked to determinant)
    r1, r2 = tf.unstack(r, axis=-1)
    cos_theta = 2 * r1 - 1
    sin_theta = tf.math.sqrt(1 - cos_theta**2)
    phi = 2 * tf_pi * (r2 - 0.5)
    logdet = tf.math.log(4 * tf_pi)

    # parametrize the momenta in CM (not linked to determinant)
    p01 = (s + m1**2 - m2**2) / tf.math.sqrt(4 * s)
    p02 = (s - m1**2 + m2**2) / tf.math.sqrt(4 * s)
    pp = tf.math.sqrt(kaellen(s, m1**2, m2**2)) / tf.math.sqrt(4 * s)
    px1 = pp * sin_theta * tf.math.cos(phi)
    py1 = pp * sin_theta * tf.math.sin(phi)
    pz1 = pp * cos_theta

    p = tf.stack((p01, px1, py1, pz1, p02, -px1, -py1, -pz1), axis=-1)
    return p, log_volume - logdet
