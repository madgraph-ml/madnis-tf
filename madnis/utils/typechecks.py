"""Functions that check types."""

import tensorflow as tf
import numpy as np


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False


def as_numpy_dtype(dtype):
  """Returns a `np.dtype` based on this `dtype`."""
  dtype = tf.as_dtype(dtype)
  if hasattr(dtype, 'as_numpy_dtype'):
    return dtype.as_numpy_dtype
  return dtype


def base_dtype(dtype):
  """Returns a non-reference `dtype` based on this `dtype`."""
  dtype = None if dtype is None else tf.as_dtype(dtype)
  if hasattr(dtype, 'base_dtype'):
    return dtype.base_dtype
  return dtype


def base_equal(a, b):
  """Returns `True` if base dtypes are identical."""
  return base_dtype(a) == base_dtype(b)
