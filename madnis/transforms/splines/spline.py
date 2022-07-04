""" This module implements utility functions for splines.

These utility functions are used in many different spline types. Having them
all in one location allows for transparency in the code. Some of the common functions
include the ability to ensure that the inputs are in the correct range, to shift inputs
from an arbitrary range to be between zero and 1, etc.

"""

# pylint: disable=invalid-name, too-many-arguments

import tensorflow as tf


def _check_bounds(inputs, left, right, top, bottom, inverse):
    left = tf.cast(left, dtype=tf.float64)
    right = tf.cast(right, dtype=tf.float64)
    bottom = tf.cast(bottom, dtype=tf.float64)
    top = tf.cast(top, dtype=tf.float64)

    if not inverse:
        out_of_bounds = (inputs < left) | (inputs > right)
        inputs = tf.where(out_of_bounds, left, inputs)
    else:
        out_of_bounds = (inputs < bottom) | (inputs > top)
        inputs = tf.where(out_of_bounds, bottom, inputs)

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    return inputs


def _shift_output(outputs, logabsdet, left, right, top, bottom, inverse):
    left = tf.cast(left, dtype=tf.float64)
    right = tf.cast(right, dtype=tf.float64)
    bottom = tf.cast(bottom, dtype=tf.float64)
    top = tf.cast(top, dtype=tf.float64)

    outputs = tf.clip_by_value(outputs, 0, 1)

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - \
            tf.math.log(top - bottom) + tf.math.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + \
            tf.math.log(top - bottom) - tf.math.log(right - left)

    return outputs, logabsdet


def _padded(t, lhs, rhs=None):
    """Left pads and optionally right pads the innermost axis of `t`."""
    lhs = tf.convert_to_tensor(lhs, dtype=t.dtype)
    zeros = tf.zeros([tf.rank(t) - 1, 2], dtype=tf.int32)
    lhs_paddings = tf.concat([zeros, [[1, 0]]], axis=0)
    result = tf.pad(t, paddings=lhs_paddings, constant_values=lhs)
    if rhs is not None:
        rhs = tf.convert_to_tensor(lhs, dtype=t.dtype)
        rhs_paddings = tf.concat([zeros, [[0, 1]]], axis=0)
        result = tf.pad(result, paddings=rhs_paddings, constant_values=rhs)
    return result


def _knot_positions(bin_sizes, range_min):
    return _padded(tf.cumsum(bin_sizes, axis=-1) + range_min, lhs=range_min)


def _gather_squeeze(params, indices):
    rank = len(indices.shape)
    if rank is None:
        raise ValueError('`indices` must have a statically known rank.')
    return tf.gather(params, indices, axis=-1, batch_dims=rank - 1)[..., 0]


def _search_sorted(cdf, inputs):
    return tf.maximum(tf.zeros([], dtype=tf.int32),
                      tf.searchsorted(
                          cdf[..., :-1],
                          inputs[..., tf.newaxis],
                          side='right',
                          out_type=tf.int32) - 1)


def _cube_root(x):
    return tf.sign(x) * tf.exp(tf.math.log(tf.abs(x))/3.0)
