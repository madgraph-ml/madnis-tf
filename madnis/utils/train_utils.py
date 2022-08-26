"""
Implementation of functions that are important for training.
"""

import tensorflow as tf


def integrate(integrand: tf.Tensor):
    """Integrate the function with given integrand.

    This method estimates the value of the integral based on
    Monte Carlo integration. It returns a tuple of two
    tf.tensors. The first one is the mean, i.e. the estimate of
    the integral. The second one gives the variance of the integrand.
    To get the variance of the estimated mean, the returned variance
    needs to be divided by (nsamples - 1).

    Args:
        integrand (tf.tensor): integrand with shape (samples, n_channels)

    Returns:
        tuple of 2 tf.tensors: mean and mc error

    """
    if len(integrand.shape) == 1:
        n_channels = 1
        nsamples = integrand.shape[0]
    else:
        n_channels = integrand.shape[0]
        nsamples = integrand.shape[0]
        
    means, vars = tf.nn.moments(integrand, axes=[0])
    mean, var = tf.reduce_sum(means), tf.reduce_sum(vars)

    return mean, tf.sqrt(n_channels * var / (nsamples - 1.0))


def parse_schedule(sched_str):
    tokens = []
    tmp = ""
    for c in sched_str:
        if "0" <= c <= "9":
            tmp += c
        else:
            if tmp != "":
                tokens.append(int(tmp))
                tmp = ""
            if c in ["g", "r", "d", "(", ")"]:
                tokens.append(c)
    if tmp != "":
        tokens.append(int(tmp))

    return _parse_schedule_rec(iter(tokens))


def _parse_schedule_rec(tokens):
    multiplier = 1
    schedule = []
    for t in tokens:
        if isinstance(t, int):
            multiplier *= t
        elif t in ["g", "r", "d"]:
            schedule.extend([t] * multiplier)
            multiplier = 1
        elif t == "(":
            schedule.extend(_parse_schedule_rec(tokens) * multiplier)
            multiplier = 1
        elif t == ")":
            return schedule
    return schedule
