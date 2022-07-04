""" Implement quadratic splines.
    Based on the pytorch implementation of
    https://github.com/bayesiains/nsf """

# pylint: disable=too-many-arguments, too-many-locals, too-many-branches
# pylint: disable=too-many-statements, invalid-name

import tensorflow as tf
from .spline import _padded, _knot_positions, _gather_squeeze, _search_sorted

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3


def quadratic_spline(inputs,
                     unnormalized_widths,
                     unnormalized_heights,
                     inverse=False,
                     left=0., right=1., bottom=0., top=1.,
                     min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                     min_bin_height=DEFAULT_MIN_BIN_HEIGHT):
    r""" Implementation of quadratic spline.

        Calculates a set of input points given an unnormalized widths distribution,
        and an unnormalized heights distribution. The forward pass through the
        quadratic spline is defined as:

        .. math::

            a &=\frac{1}{2}\left(V_{i+1} - V_i\right) W_i, \\
            b &=V_i W_i, \\
            c &=\sum_j=1^{i-1} V_i, \\
            \alpha &= \frac{x - x_i}{W_i}, \\
            y &=a\alpha^2 + b\alpha + c, \\
            \log\left(\frac{dy}{dx}\right) &= \log\left(\alpha \left(V_{i+1}
                                               - V_i\right) + V_i\right),

        where :math:`x` is the input value, :math:`x_i, V_i, W_i` are the values of the
        left edge, the height of the left edge, and the width of the bin respectively.
        While the inverse pass is defined as:

        .. math::

            \alpha &= \frac{2\left(c-y\right)}{-b-\sqrt{b^2-4a\left(c-y\right)}}, \\
            x &= \alpha W_i + x_i, \\
            \log\left(\frac{dx}{dy}\right) &= -\log\left(\alpha \left(V_{i+1}
                                                - V_i\right) + V_i\right),

        where :math:`y` is the input value, and the rest are the same as the forward pass.

        Args:
            inputs (tf.Tensor): An array of inputs to be transformed by the spline.
            unnormalized_widths (tf.Tensor): A set of unnormalized widths for the bins.
            unnormalized_heights (tf.Tensor): A set of unnormalized heights for the bins.
            inverse (bool): Whether to calculate the forward or inverse pass
            left (float64): Left edge of the valid spline region
            right (float64): Right edge of the valid spline region
            bottom (float64): Bottom edge of the valid spline region
            top (float64): Top edge of the valid spline region
            min_bin_width (float64): The minimum allowed width of a given bin
            min_bin_height (float64): The minimum allowed height of a given knot

        Returns:
            tuple: The transformation and the associated log jacobian
    """

    left = tf.cast(left, dtype=tf.float64)
    right = tf.cast(right, dtype=tf.float64)
    bottom = tf.cast(bottom, dtype=tf.float64)
    top = tf.cast(top, dtype=tf.float64)

    if not inverse:
        out_of_bounds = (inputs < left) | (inputs > right)
        tf.where(out_of_bounds, left, inputs)
    else:
        out_of_bounds = (inputs < bottom) | (inputs > top)
        tf.where(out_of_bounds, bottom, inputs)

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = tf.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    unnormalized_heights_exp = tf.math.exp(unnormalized_heights)

    if unnormalized_heights_exp.shape[-1] == num_bins - 1:
        # Set boundary heights s.t. after normalization they are exactly 1.
        first_widths = 0.5 * widths[..., 0]
        last_widths = 0.5 * widths[..., -1]
        numerator = (0.5 * first_widths * unnormalized_heights_exp[..., 0]
                     + 0.5 * last_widths * unnormalized_heights_exp[..., -1]
                     + tf.reduce_sum(
                         ((unnormalized_heights_exp[..., :-1]
                           + unnormalized_heights_exp[..., 1:]) / 2)
                         * widths[..., 1:-1], axis=-1))

        constant = numerator / (1. - 0.5 * first_widths - 0.5 * last_widths)
        constant = constant[..., tf.newaxis]
        unnormalized_heights_exp = tf.concat(
            [constant, unnormalized_heights_exp, constant], axis=-1)

    unnormalized_area = tf.reduce_sum(
        ((unnormalized_heights_exp[..., :-1]
          + unnormalized_heights_exp[..., 1:]) / 2.)
        * widths, axis=-1)[..., tf.newaxis]

    heights = unnormalized_heights_exp / unnormalized_area
    heights = min_bin_height + (1. - min_bin_height) * heights

    bin_left_cdf = tf.cumsum(
        ((heights[..., :-1] + heights[..., 1:]) / 2.) * widths, axis=-1)
    bin_left_cdf = _padded(bin_left_cdf, lhs=0.)

    bin_locations = _knot_positions(widths, 0.)

    if inverse:
        bin_idx = _search_sorted(bin_left_cdf, inputs)
    else:
        bin_idx = _search_sorted(bin_locations, inputs)

    input_bin_locations = _gather_squeeze(bin_locations, bin_idx)
    input_bin_widths = _gather_squeeze(widths, bin_idx)

    input_left_cdf = _gather_squeeze(bin_left_cdf, bin_idx)
    input_left_heights = _gather_squeeze(heights, bin_idx)
    input_right_heights = _gather_squeeze(heights, bin_idx+1)

    a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
    b = input_left_heights * input_bin_widths
    c = input_left_cdf

    if inverse:
        c_ = c - inputs
        alpha = 2*c_/(-b-tf.sqrt(b**2 - 4*a*c_))
        outputs = alpha * input_bin_widths + input_bin_locations
    else:
        alpha = (inputs - input_bin_locations) / input_bin_widths
        outputs = a * alpha**2 + b * alpha + c

    outputs = tf.clip_by_value(outputs, 0, 1)
    logabsdet = tf.math.log((alpha * (input_right_heights - input_left_heights)
                             + input_left_heights))

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = -logabsdet - \
            tf.math.log(top - bottom) + tf.math.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + \
            tf.math.log(top - bottom) - tf.math.log(right - left)

    return outputs, logabsdet
