"""
Implementation of functions that are important for training.
"""

import tensorflow as tf


def integrate(integrand: tf.Tensor):
    """
    Multi-channel integration
    Args:
        integrand: Tensor, with shape (samples, n_channels)
    """
    means = tf.math.reduce_mean(integrand, axis=0)
    result = tf.reduce_sum(means)
    return result


def error(integrand: tf.Tensor):
    """
    Error of Multi-channel integration
    Args:
        integrand: Tensor, with shape (samples, n_channels)
    """
    n = integrand.shape[0]
    means = tf.math.reduce_mean(integrand, axis=0)
    means2 = tf.math.reduce_mean(integrand ** 2, axis=0)
    var = tf.math.reduce_sum(means2 - means ** 2)
    return tf.sqrt(var / (n - 1.0))
