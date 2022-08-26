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


def parse_schedule(sched_str: str) -> list[str]:
    """Parses strings that describe two-stage training schedules.

    This function parses string with a compact syntax to describe training
    schedules. Different actions are expressed with single letters:
      g: generate samples and optimize
      r: reuse saved samples for optimization
      d: delete all saved samples
    Numbers in front of actions can be used to perform them multiple times.
    Parentheses can be used to group actions together.
    Examples:
      "5g":         5 epochs of sampling+training
      "3gd5(g2r)":  3 epochs of sampling+training training, then delete
                    all samples, then 5 times 1 epoch of sampling+training
                    followed by 2 epochs of training on saved samples

    Args:
        sched_str: string with schedule expression

    Returns:
        list of strings "g", "r" and "d"

    """
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
