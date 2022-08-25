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
