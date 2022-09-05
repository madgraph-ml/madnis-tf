""" Implement cubic spline. 
    Based on the pytorch implementation of
    https://github.com/bayesiains/nsf """

# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
# pylint: disable=invalid-name

import tensorflow as tf
from .spline_utils import _knot_positions, _gather_squeeze, _search_sorted
from .spline_utils import _cube_root, _check_bounds, _shift_output

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_EPS = 1e-5
DEFAULT_QUADRATIC_THRESHOLD = 1e-3

def unconstrained_cubic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnorm_derivatives_left,
    unnorm_derivatives_right,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    eps=DEFAULT_EPS,
    quadratic_threshold=DEFAULT_QUADRATIC_THRESHOLD,
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
    unnorm_wds = tf.dynamic_partition(unnormalized_widths, splittings, 2)
    unnorm_hts = tf.dynamic_partition(unnormalized_heights, splittings, 2)
    unnorm_divs_left = tf.dynamic_partition(unnorm_derivatives_left, splittings, 2)
    unnorm_divs_right = tf.dynamic_partition(unnorm_derivatives_right, splittings, 2)
    idx = tf.dynamic_partition(tf.range(tf.shape(inputs)[0]), splittings, 2)

    # Logs and outputs outside of domain
    logabsdets.append(tf.zeros_like(ins[0]))
    outputs.append(ins[0])

    # Logs and outputs inside of domain
    outputs_inside, logabsdet_inside = cubic_spline(
        inputs=ins[1],
        unnormalized_widths=unnorm_wds[1],
        unnormalized_heights=unnorm_hts[1],
        unnorm_derivatives=unnorm_divs_left[1],
        unnorm_derivatives=unnorm_divs_right[1],
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        eps=eps,
        quadratic_threshold=quadratic_threshold
    )

    outputs.append(outputs_inside)
    logabsdets.append(logabsdet_inside)

    # Combine all of them
    output = tf.dynamic_stitch(idx, outputs)
    logabsdet = tf.dynamic_stitch(idx, logabsdets)

    return output, logabsdet


def cubic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnorm_derivatives_left,
    unnorm_derivatives_right,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    eps=DEFAULT_EPS,
    quadratic_threshold=DEFAULT_QUADRATIC_THRESHOLD,
):
    
    inputs = _check_bounds(inputs, left, right, top, bottom, inverse)

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = tf.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = _knot_positions(widths, 0)

    heights = tf.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = _knot_positions(heights, 0)

    slopes = heights / widths
    min_slope_1 = tf.minimum(tf.abs(slopes[..., :-1]), tf.abs(slopes[..., 1:]))
    min_slope_2 = (
        0.5
        * (widths[..., 1:] * slopes[..., :-1] + widths[..., :-1] * slopes[..., 1:])
        / (widths[..., :-1] + widths[..., 1:])
    )
    min_slope = tf.minimum(min_slope_1, min_slope_2)

    derivatives_left = (
        tf.nn.sigmoid(unnorm_derivatives_left) * 3 * slopes[..., 0][..., tf.newaxis]
    )
    derivatives_right = (
        tf.nn.sigmoid(unnorm_derivatives_right) * 3 * slopes[..., -1][..., tf.newaxis]
    )

    derivatives = min_slope * (tf.sign(slopes[..., :-1]) + tf.sign(slopes[..., 1:]))
    derivatives = tf.concat([derivatives_left, derivatives, derivatives_right], axis=-1)

    a = (derivatives[..., :-1] + derivatives[..., 1:] - 2 * slopes) / widths ** 2
    b = (3 * slopes - 2 * derivatives[..., :-1] - derivatives[..., 1:]) / widths
    c = derivatives[..., :-1]
    d = cumheights[..., :-1]

    if inverse:
        bin_idx = _search_sorted(cumheights, inputs)
    else:
        bin_idx = _search_sorted(cumwidths, inputs)

    inputs_a = _gather_squeeze(a, bin_idx)
    inputs_b = _gather_squeeze(b, bin_idx)
    inputs_c = _gather_squeeze(c, bin_idx)
    inputs_d = _gather_squeeze(d, bin_idx)

    input_left_cumwidths = _gather_squeeze(cumwidths, bin_idx)
    input_right_cumwidths = _gather_squeeze(cumwidths, bin_idx + 1)

    if inverse:
        # Modified coefficients for solving the cubic.
        inputs_b_ = (inputs_b / inputs_a) / 3.0
        inputs_c_ = (inputs_c / inputs_a) / 3.0
        inputs_d_ = (inputs_d - inputs) / inputs_a

        delta_1 = -(inputs_b_ ** 2) + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_ ** 2

        discriminant = 4.0 * delta_1 * delta_3 - delta_2 ** 2

        depressed_1 = -2 * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1

        def one_root():
            # Deal with one root cases
            p = _cube_root((-depressed_1 + tf.sqrt(-discriminant)) / 2.0)
            q = _cube_root((-depressed_1 - tf.sqrt(-discriminant)) / 2.0)

            return (p + q) - inputs_b_ + input_left_cumwidths

        def three_roots():
            # Deal with three root cases
            theta = tf.atan2(tf.sqrt(discriminant), -depressed_1)
            theta /= 3.0

            cubic_root_1 = tf.cos(theta)
            cubic_root_2 = tf.sin(theta)

            root_1 = cubic_root_1
            root_2 = (
                -0.5 * cubic_root_1
                - tf.cast(0.5 * tf.sqrt(3.0), dtype=tf.float64) * cubic_root_2
            )
            root_3 = (
                -0.5 * cubic_root_1
                + tf.cast(0.5 * tf.sqrt(3.0), dtype=tf.float64) * cubic_root_2
            )

            root_scale = 2 * tf.sqrt(-depressed_2)
            root_shift = -inputs_b_ + input_left_cumwidths

            root_1 = root_1 * root_scale + root_shift
            root_2 = root_2 * root_scale + root_shift
            root_3 = root_3 * root_scale + root_shift

            root1_mask = tf.cast(
                (input_left_cumwidths - eps) < root_1, dtype=tf.float64
            )
            root1_mask *= tf.cast(
                root_1 < (input_right_cumwidths + eps), dtype=tf.float64
            )

            root2_mask = tf.cast(
                (input_left_cumwidths - eps) < root_2, dtype=tf.float64
            )
            root2_mask *= tf.cast(
                root_2 < (input_right_cumwidths + eps), dtype=tf.float64
            )

            root3_mask = tf.cast(
                (input_left_cumwidths - eps) < root_3, dtype=tf.float64
            )
            root3_mask *= tf.cast(
                root_3 < (input_right_cumwidths + eps), dtype=tf.float64
            )

            roots = tf.stack([root_1, root_2, root_3], axis=-1)
            masks = tf.stack([root1_mask, root2_mask, root3_mask], axis=-1)
            mask_index = tf.argsort(masks, axis=-1, direction="DESCENDING")[..., 0][
                ..., tf.newaxis
            ]

            return _gather_squeeze(roots, mask_index)

        def quadratic():
            # Deal with a -> 0 (almost quadratic) cases

            a = inputs_b
            b = inputs_c
            c = inputs_d - inputs
            alpha = tf.where(
                tf.abs(a) > 1e-16, (-b + tf.sqrt(b ** 2 - 4 * a * c)) / (2 * a), -c / b
            )
            return alpha + input_left_cumwidths

        outputs = tf.where(
            tf.abs(inputs_a) < quadratic_threshold,
            quadratic(),
            tf.where(discriminant < 0, one_root(), three_roots()),
        )

        shifted_outputs = outputs - input_left_cumwidths
        logabsdet = -tf.math.log(
            3.0 * inputs_a * shifted_outputs ** 2
            + 2.0 * inputs_b * shifted_outputs
            + inputs_c
        )
        print(outputs, logabsdet)

    else:
        shifted_inputs = inputs - input_left_cumwidths
        outputs = (
            inputs_a * shifted_inputs ** 3
            + inputs_b * shifted_inputs ** 2
            + inputs_c * shifted_inputs
            + inputs_d
        )

        logabsdet = tf.math.log(
            3.0 * inputs_a * shifted_inputs ** 2
            + 2.0 * inputs_b * shifted_inputs
            + inputs_c
        )

    return _shift_output(outputs, logabsdet, left, right, top, bottom, inverse)
