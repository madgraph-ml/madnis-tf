""" Implement linear splines.
    Based on the pytorch implementation of
    https://github.com/bayesiains/nsf """

# pylint: disable=too-many-arguments, too-many-locals

import tensorflow as tf
from .spline_utils import _knot_positions, _gather_squeeze, _search_sorted
from .spline_utils import _check_bounds, _shift_output


def unconstrained_linear_spline(
    inputs,
    unnormalized_pdf,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
):

    if not inverse:
        inside_interval_mask = tf.math.reduce_all(
            (inputs >= left) & (inputs <= right), axis=-1
        )
    else:
        inside_interval_mask = tf.math.reduce_all(
            (inputs >= bottom) & (inputs <= top), axis=-1
        )

    outputs = []
    logabsdets = []
    splittings = tf.cast(inside_interval_mask, tf.int32)

    ins = tf.dynamic_partition(inputs, splittings, 2)
    unnorm_pdfs = tf.dynamic_partition(unnormalized_pdf, splittings, 2)
    idx = tf.dynamic_partition(tf.range(tf.shape(inputs)[0]), splittings, 2)

    # Logs and outputs outside of domain
    logabsdets.append(tf.zeros_like(ins[0]))
    outputs.append(ins[0])

    # Logs and outputs inside of domain
    outputs_inside, logabsdet_inside = linear_spline(
        inputs=ins[1],
        unnormalized_pdf=unnorm_pdfs[1],
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )

    outputs.append(outputs_inside)
    logabsdets.append(logabsdet_inside)

    # Combine all of them
    output = tf.dynamic_stitch(idx, outputs)
    logabsdet = tf.dynamic_stitch(idx, logabsdets)

    return output, logabsdet


def linear_spline(
    inputs,
    unnormalized_pdf,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
):
    r""" Implementation of linear spline.

        Calculates a set of input points given an unnormalized pdf distribution.
        The forward pass through the linear spline is defined as:

        .. math::

            y &= \frac{x - x_i}{m_i}, \\
            \log\left(\frac{dy}{dx}\right) &= -\log(x) + \log(w),

        where :math:`x` is the input value, :math:`x_i\ (m_i)` is the left bin edge
        (slope) for the bin the given input falls within, and :math:`w` is the width
        of the bins of the spline. While the inverse pass is defined as:

        .. math::

            x &= N\cdot y - b_i + \sum_{j=1}^{i-1} y_j, \\
            \log\left(\frac{dx}{dy}\right) &= \log(x) - \log(w),

        where :math:`y` is the input value, :math:`b_i` is the bin in which :math:`y`
        falls, :math:`N` is the number of bins, and :math:`y_j` is the height of the
        :math:`j^{\text{th}}` bin.

        Args:
            inputs (tf.Tensor): An array of inputs to be transformed by the spline.
            unnormalized_pdf (tf.Tensor): An unnormalized pdf describing the
                                          transformation function.
            inverse (bool): Whether to calculate the forward or inverse pass
            left (float64): Left edge of the valid spline region
            right (float64): Right edge of the valid spline region
            bottom (float64): Bottom edge of the valid spline region
            top (float64): Top edge of the valid spline region

        Returns:
            tuple: The transformation and the associated log jacobian
    """

    inputs = _check_bounds(inputs, left, right, top, bottom, inverse)

    num_bins = unnormalized_pdf.shape[-1]
    pdf = tf.nn.softmax(unnormalized_pdf, axis=-1)
    cdf = _knot_positions(pdf, 0)

    if inverse:
        inv_bin_idx = _search_sorted(cdf, inputs)
        bin_boundaries = tf.cast(tf.linspace(0.0, 1.0, num_bins + 1), dtype=tf.float64)
        slopes = (cdf[..., 1:] - cdf[..., :-1]) / (
            bin_boundaries[..., 1:] - bin_boundaries[..., :-1]
        )
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        input_slopes = _gather_squeeze(slopes, inv_bin_idx)
        input_offsets = _gather_squeeze(offsets, inv_bin_idx)

        outputs = (inputs - input_offsets) / input_slopes

        input_pdfs = _gather_squeeze(pdf, inv_bin_idx)
        bin_width = tf.cast(1.0 / num_bins, dtype=tf.float64)
        logabsdet = -tf.math.log(input_pdfs) + tf.math.log(bin_width)
    else:
        bin_pos = inputs * num_bins
        bin_idx_float = tf.floor(bin_pos)
        bin_idx = tf.cast(bin_idx_float, dtype=tf.int32)[..., tf.newaxis]

        alpha = bin_pos - bin_idx_float
        input_pdfs = _gather_squeeze(pdf, bin_idx)

        outputs = _gather_squeeze(cdf[..., :-1], bin_idx)
        outputs += alpha * input_pdfs

        bin_width = tf.cast(1.0 / num_bins, dtype=tf.float64)
        logabsdet = tf.math.log(input_pdfs) - tf.math.log(bin_width)

    return _shift_output(outputs, logabsdet, left, right, top, bottom, inverse)
