"""Various TensorFlow utility functions."""

import typechecks as check
import tensorflow as tf


def sum_except_batch(x, num_batch_dims=1):
    """
    Sums all dimensions except for the first `num_batch_dims` dimensions.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_batch_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    """
    if not check.is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = tuple(range(num_batch_dims, len(x.shape)))
    return tf.math.reduce_sum(x, axis=reduce_dims)

def partion_into_list(nsamples: int, parts: int):
    """Splits a given number n into a m parts of equal size
    that sum up to n again.

    Args:
        nsamples (int): number of samples
        parts (int): numper of parts

    Returns:
        list: the partioned number of samples
    """
    nsamples_list = [nsamples // parts for _ in range(parts - 1)] + [ nsamples // parts + nsamples % parts]
    return nsamples_list
    

def prod_except_batch(x, num_batch_dims=1):
    """
    Multiplies all dimensions except for the first `num_batch_dims` dimensions.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_batch_dims: int, number of batch dims (default=1)

    Returns:
        x_prod: Tensor, shape (batch_size,)
    """
    if not check.is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = tuple(range(num_batch_dims, len(x.shape)))
    return tf.math.reduce_prod(x, axis=reduce_dims)


def mean_except_batch(x, num_batch_dims=1):
    """
    Averages all dimensions except for the first `num_batch_dims` dimensions.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_batch_dims: int, number of batch dims (default=1)

    Returns:
        x_mean: Tensor, shape (batch_size,)
    """
    if not check.is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = tuple(range(num_batch_dims, len(x.shape)))
    return tf.math.reduce_mean(x, axis=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = tf.TensorShape(shape) + x.shape[1:]
    return tf.reshape(x, new_shape)


def merge_leading_dims(x, num_dims=2):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not check.is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > len(x.shape):
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = [-1] + list(x.shape[num_dims:])
    return tf.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not check.is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    nldims = len(x.shape) - 1
    multiples = [num_reps] + [1] * nldims
    return tf.tile(x, multiples)


def logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: tf.linalg.logdet only works for positive determinant.
    _, res = tf.linalg.slogdet(x)
    return res


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = tf.random.normal((size, size))
    q, _ = tf.linalg.qr(x)
    return q
