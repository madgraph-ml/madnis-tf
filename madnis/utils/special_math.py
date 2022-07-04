"""Special Math Ops."""

# Adopted and copied from https://github.com/tensorflow/probability/

import numpy as np
import tensorflow as tf

import madml.utils.typechecks as checks

# log_ndtr uses different functions over the ranges
# (-infty, lower](lower, upper](upper, infty)
# Lower bound values were chosen by examining where the support of ndtr
# appears to be zero, relative to scipy's (which is always 64bit). They were
# then made more conservative just to be safe. (Conservative means use the
# expansion more than we probably need to.)
LOGNDTR_FLOAT64_LOWER = -20.0
LOGNDTR_FLOAT32_LOWER = -10.0

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
LOGNDTR_FLOAT64_UPPER = 8.0
LOGNDTR_FLOAT32_UPPER = 5.0


def ndtr(x, name="ndtr"):
    """Normal distribution function.
    Returns the area under the Gaussian probability density function, integrated
    from minus infinity to x:
    ```
                      1       / x
      ndtr(x)  = ----------  |    exp(-0.5 t**2) dt
                  sqrt(2 pi)  /-inf
                = 0.5 (1 + erf(x / sqrt(2)))
                = 0.5 erfc(x / sqrt(2))
    ```
    Args:
      x: `Tensor` of type `float32`, `float64`.
      name: Python string. A name for the operation (default="ndtr").

    Returns:
      ndtr: `Tensor` with `dtype=x.dtype`.

    Raises:
      TypeError: if `x` is not floating-type.
    """

    with tf.name_scope(name):
        x = tf.convert_to_tensor(x, name="x")
        if checks.as_numpy_dtype(x.dtype) not in [np.float32, np.float64]:
            raise TypeError(
                "x.dtype=%s is not handled, see docstring for supported types."
                % x.dtype
            )
        return _ndtr(x)


def _ndtr(x):
    """Implements ndtr core logic."""
    half_sqrt_2 = tf.constant(0.5 * np.sqrt(2.0), dtype=x.dtype, name="half_sqrt_2")
    w = x * half_sqrt_2
    z = tf.abs(w)
    y = tf.where(
        z < half_sqrt_2,
        1.0 + tf.math.erf(w),
        tf.where(w > 0.0, 2.0 - tf.math.erfc(z), tf.math.erfc(z)),
    )
    return 0.5 * y


def log_ndtr(x, series_order=3, name="log_ndtr"):
    """Log Normal distribution function.
    For details of the Normal distribution function see `ndtr`.
    This function calculates `(log o ndtr)(x)` by either calling `log(ndtr(x))` or
    using an asymptotic series. Specifically:
    - For `x > upper_segment`, use the approximation `-ndtr(-x)` based on
      `log(1-x) ~= -x, x << 1`.
    - For `lower_segment < x <= upper_segment`, use the existing `ndtr` technique
      and take a log.
    - For `x <= lower_segment`, we use the series approximation of erf to compute
      the log CDF directly.
    The `lower_segment` is set based on the precision of the input:

    ```
    lower_segment = { -20,  x.dtype=float64
                    { -10,  x.dtype=float32
    upper_segment = {   8,  x.dtype=float64
                    {   5,  x.dtype=float32
    ```

    Check https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/internal/special_math.py
    for details.

    Args:
      x: `Tensor` of type `float32`, `float64`.
      series_order: Positive Python `integer`. Maximum depth to
        evaluate the asymptotic expansion. This is the `N` above.
      name: Python string. A name for the operation (default="log_ndtr").

    Returns:
      log_ndtr: `Tensor` with `dtype=x.dtype`.

    Raises:
      TypeError: if `x.dtype` is not handled.
      TypeError: if `series_order` is a not Python `integer.`
      ValueError:  if `series_order` is not in `[0, 30]`.
    """
    if not isinstance(series_order, int):
        raise TypeError("series_order must be a Python integer.")
    if series_order < 0:
        raise ValueError("series_order must be non-negative.")
    if series_order > 30:
        raise ValueError("series_order must be <= 30.")

    with tf.name_scope(name):
        x = tf.convert_to_tensor(x, name="x")

        if checks.base_equal(x.dtype, tf.float64):
            lower_segment = np.array(LOGNDTR_FLOAT64_LOWER, dtype=np.float64)
            upper_segment = np.array(LOGNDTR_FLOAT64_UPPER, dtype=np.float64)
        elif checks.base_equal(x.dtype, tf.float32):
            lower_segment = np.array(LOGNDTR_FLOAT32_LOWER, dtype=np.float32)
            upper_segment = np.array(LOGNDTR_FLOAT32_UPPER, dtype=np.float32)
        else:
            raise TypeError("x.dtype=%s is not supported." % x.dtype)

        return tf.where(
            x > upper_segment,
            -_ndtr(-x),  # log(1-x) ~= -x, x << 1
            tf.where(
                x > lower_segment,
                tf.math.log(_ndtr(tf.maximum(x, lower_segment))),
                _log_ndtr_lower(tf.minimum(x, lower_segment), series_order),
            ),
        )


def _log_ndtr_lower(x, series_order):
    """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
    x_2 = tf.square(x)
    # Log of the term multiplying (1 + sum)
    log_scale = (
        -0.5 * x_2
        - tf.math.log(-x)
        - tf.constant(0.5 * np.log(2.0 * np.pi), dtype=x.dtype)
    )
    return log_scale + tf.math.log(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
    """Calculates the asymptotic series used in log_ndtr."""
    npdt = checks.as_numpy_dtype(x.dtype)
    if series_order <= 0:
        return npdt(1)
    x_2 = tf.square(x)
    even_sum = tf.zeros_like(x)
    odd_sum = tf.zeros_like(x)
    x_2n = x_2  # Start with x^{2*1} = x^{2*n} with n = 1.
    for n in range(1, series_order + 1):
        y = npdt(_double_factorial(2 * n - 1)) / x_2n
        if n % 2:
            odd_sum += y
        else:
            even_sum += y
        x_2n *= x_2
    return 1.0 + even_sum - odd_sum


def _double_factorial(n):
    """The double factorial function for small Python integer `n`."""
    return np.prod(np.arange(n, 1, -2))
