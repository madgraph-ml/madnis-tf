""" Implement Rational Quadratic splines.
    Based on the pytorch implementation of
    https://github.com/bayesiains/nsf """

# pylint: disable=too-many-arguments, too-many-locals, invalid-name

import tensorflow as tf
from .spline import _knot_positions, _gather_squeeze, _search_sorted

DEFAULT_MIN_BIN_WIDTH = 1e-15
DEFAULT_MIN_BIN_HEIGHT = 1e-15
DEFAULT_MIN_DERIVATIVE = 1e-15


def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE):
    r""" Implementation of rational quadratic spline.

        Calculates a set of input points given an unnormalized widths distribution,
        an unnormalized heights distribution, and an unnormalized derivatives distribution.
        The forward pass through the rational quadratic spline is defined as:

        .. math::

            \theta &= \frac{x - x_i}{W_i}, \\
            y &= x_i + \frac{(y_{i+1}-y_i)(s_i\theta^2+d_i\theta(1-\theta))}
                            {s_i + (d_{i+1}+d_{i}-2s_i)\theta(1-\theta)}, \\
            \log\left(\frac{dy}{dx}\right) &= \log(s_i^2(d_{i+1}\theta^2+2s_i\theta(1-\theta)
                                                   +d_i(1-\theta)^2)) \\
                                           &- 2\log(s_i + (d_{i+1}+d_{i}-2s_i)\theta(1-\theta)),

        where :math:`x` is the input value, :math:`(x_i, y_i)` is the position of the ith knot,
        :math:`s_i` is the slope at the ith knot, and :math:`d_i` is the derivative of the
        of the spline at the ith knot. While the inverse pass is defined as:

        .. math::

            a &= (y - y_i)(d_i+d_{i+1}-2s_i) + (y_i-y_{i-1})(s_i-d_i), \\
            b &= (y_i - y_{i-1})d_i-(y-y_i)(d_i+d_{i+1}-2s_i), \\
            c &= -s_i(y-y_i), \\
            \theta &= \frac{2c}{-b-\sqrt{b^2-4ac}}, \\
            x &= \theta W_i + x_i, \\
            \log\left(\frac{dx}{dy}\right) &= -\log(s_i^2(d_{i+1}\theta^2+2s_i\theta(1-\theta)
                                                   +d_i(1-\theta)^2)) \\
                                           &+ 2\log(s_i + (d_{i+1}+d_{i}-2s_i)\theta(1-\theta))

        where :math:`y` is the input value, and the rest are the same as the forward pass.

        Args:
            inputs (tf.Tensor): An array of inputs to be transformed by the spline.
            unnormalized_widths (tf.Tensor): A set of unnormalized widths for the knots.
            unnormalized_heights (tf.Tensor): A set of unnormalized heights for the knots.
            unnormalized_derivatives (tf.Tensor): A set of unnormalized derivatives for the knots.
            inverse (bool): Whether to calculate the forward or inverse pass
            left (float64): Left edge of the valid spline region
            right (float64): Right edge of the valid spline region
            bottom (float64): Bottom edge of the valid spline region
            top (float64): Top edge of the valid spline region
            min_bin_width (float64): The minimum allowed width of a given bin
            min_bin_height (float64): The minimum allowed height of a given knot
            min_derivative (float64): The minimum allowed derivative of a given knot

        Returns:
            tuple: The transformation and the associated log jacobian
    """

    out_of_bounds = (inputs < left) | (inputs > right)
    tf.where(out_of_bounds, tf.cast(left, dtype=inputs.dtype), inputs)

    num_bins = unnormalized_widths.shape[-1]
    # check that number of widths, heights, and derivatives match
    assert num_bins == unnormalized_heights.shape[-1] \
            == unnormalized_derivatives.shape[-1]-1

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = tf.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = _knot_positions(widths, 0)
    cumwidths = (right - left) * cumwidths + left
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = ((min_derivative + tf.nn.softplus(unnormalized_derivatives))
                   / (tf.cast(min_derivative + tf.math.log(2.), tf.float64)))

    heights = tf.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = _knot_positions(heights, 0)
    cumheights = (top - bottom) * cumheights + bottom
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = _search_sorted(cumheights, inputs)
    else:
        bin_idx = _search_sorted(cumwidths, inputs)

    input_cumwidths = _gather_squeeze(cumwidths, bin_idx)
    input_bin_widths = _gather_squeeze(widths, bin_idx)

    input_cumheights = _gather_squeeze(cumheights, bin_idx)
    delta = heights / widths
    input_delta = _gather_squeeze(delta, bin_idx)

    input_derivatives = _gather_squeeze(derivatives, bin_idx)
    input_derivatives_p1 = _gather_squeeze(derivatives[..., 1:], bin_idx)

    input_heights = _gather_squeeze(heights, bin_idx)

    if inverse:
        a = ((inputs - input_cumheights) * (input_derivatives
                                            + input_derivatives_p1
                                            - 2 * input_delta)
             + input_heights * (input_delta - input_derivatives))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_p1
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b**2 - 4 * a * c

        theta = (2 * c) / (-b - tf.sqrt(discriminant))
        outputs = theta * input_bin_widths + input_cumwidths

        theta_one_minus_theta = theta * (1 - theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_p1
                                      - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta**2 * (input_derivatives_p1
                                                 * theta**2
                                                 + 2 * input_delta
                                                 * theta_one_minus_theta
                                                 + input_derivatives
                                                 * (1 - theta)**2)
        logabsdet = tf.math.log(derivative_numerator) - \
            2 * tf.math.log(denominator)

        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (input_delta * theta**2
                                 + input_derivatives
                                 * theta_one_minus_theta)
    denominator = input_delta + ((input_derivatives + input_derivatives_p1
                                  - 2 * input_delta)
                                 * theta_one_minus_theta)
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta**2 * (input_derivatives_p1
                                             * theta**2
                                             + 2 * input_delta
                                             * theta_one_minus_theta
                                             + input_derivatives
                                             * (1 - theta)**2)
    logabsdet = tf.math.log(derivative_numerator) - \
        2 * tf.math.log(denominator)

    return outputs, logabsdet
