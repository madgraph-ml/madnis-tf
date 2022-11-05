""" Helper functions needed for phase-space mappings """

import tensorflow as tf


def kaellen(a: tf.Tensor, b: tf.Tensor, c: tf.Tensor):
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * b * c - 2 * c * a


def boost(q, ph):
    metric = tf.linalg.diag([1.0, -1.0, -1.0, -1.0])
    rsq = tf.math.sqrt(tf.einsum("kd,dd,kd->k", q, metric, q))

    p0 = tf.einsum("ki,ki->k", q, ph) / rsq
    c1 = (ph[:, 0] + p0) / (rsq + q[:, 0])
    px = ph[:, 1] + c1 * q[:, 1]
    py = ph[:, 2] + c1 * q[:, 2]
    pz = ph[:, 3] + c1 * q[:, 3]
    p = tf.stack((p0, px, py, pz), axis=-1)

    return p


def boost_z(q, rapidity, inverse=False):
    sign = -1.0 if inverse else 1.0

    pi0 = q[:, :, 0] * tf.math.cosh(rapidity) + sign * q[:, :, 3] * tf.math.sinh(
        rapidity
    )
    pix = q[:, :, 1]
    piy = q[:, :, 2]
    piz = q[:, :, 3] * tf.math.cosh(rapidity) + sign * q[:, :, 0] * tf.math.sinh(
        rapidity
    )
    p = tf.stack((pi0, pix, piy, piz), axis=-1)

    return p
