""" RQS Coupling Blocks """

from typing import Dict, Union
import tensorflow as tf

from .base import Transform
from .splines.rational_quadratic import rational_quadratic_spline
from ..utils.tfutils import sum_except_batch

import warnings
import numpy as np


class RationalQuadraticSplineBlock(Transform):

    # Maybe choose bigger defaults?
    DEFAULT_MIN_BIN_WIDTH = 1e-15
    DEFAULT_MIN_BIN_HEIGHT = 1e-15
    DEFAULT_MIN_DERIVATIVE = 1e-15

    def __init__(
        self,
        dims_in,
        dims_c=[],
        subnet_meta: Dict = None,
        subnet_constructor: callable = None,
        num_bins: int = 10,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        with_permute: bool = True,
        seed: Union[int, None] = None,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
    ):

        super().__init__(dims_in, dims_c)

        self.channels = self.dims_in[-1]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(self.dims_in) - 1

        # tuple containing all dims except for
        # batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        assert all(
            [
                tuple(self.dims_c[i][:-1]) == tuple(self.dims_in[:-1])
                for i in range(len(self.dims_c))
            ]
        ), "Dimensions of input and one or more conditions don't agree."
        self.conditional = len(self.dims_c) > 0
        self.condition_length = sum(
            [self.dims_c[i][-1] for i in range(len(self.dims_c))]
        )

        split_len1 = self.channels // 2
        split_len2 = self.channels - self.channels // 2
        self.splits = [split_len1, split_len2]

        # Number of bins
        self.num_bins = num_bins
        if self.DEFAULT_MIN_BIN_WIDTH * self.num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if self.DEFAULT_MIN_BIN_HEIGHT * self.num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")

        # Definitions for spline
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Define the permutation matrix
        self.with_permute = with_permute
        np.random.seed(seed)
        w = np.zeros((self.channels, self.channels))
        for i, j in enumerate(np.random.permutation(self.channels)):
            w[i, j] = 1.0

        self.w_perm = self.add_weight(
            "w_perm",
            shape=(*([1] * self.input_rank), self.channels, self.channels),
            initializer=tf.keras.initializers.Constant(w),
            trainable=False,
        )

        self.w_perm_inv = self.add_weight(
            "w_perm_inv",
            shape=(*([1] * self.input_rank), self.channels, self.channels),
            initializer=tf.keras.initializers.Constant(w.T),
            trainable=False,
        )
        
        self.permute_function = lambda x, w: tf.linalg.matvec(w, x, transpose_a=True)

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )

        self.subnet = subnet_constructor(
            subnet_meta,
            self.splits[0] + self.condition_length,
            (3 * self.num_bins + 1) * self.splits[1],
        )

    def _permute(self, x, rev=False):
        """Performs the random permutation coupling operation.
        As the logdet = 0, we do not return it."""
        if rev:
            x_permute = self.permute_function(x, self.w_perm_inv)
            return x_permute

        x_permute = self.permute_function(x, self.w_perm)
        return x_permute

    def _rational_quadratic_spline(self, x, a, rev=False):
        """Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the RQS coupling operation.
        Returns both the transformed inputs and the LogJacDet."""

        # split into different contributions
        unnormalized_widths = a[..., : self.num_bins]
        unnormalized_heights = a[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = a[..., 2 * self.num_bins :]

        y, ldj_elementwise = rational_quadratic_spline(
            x,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=rev,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        
        ldj = sum_except_batch(ldj_elementwise)

        return y, ldj

    def call(self, x, c=None, jac=True):

        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # RQS Block
        a1 = tf.reshape(
            self.subnet(x1c), (x1c.shape[0], self.splits[1], 3 * self.num_bins + 1)
        )
        x2, j2 = self._rational_quadratic_spline(x2, a1, rev=False)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        # Permutation
        if self.with_permute:
            x_out = self._permute(x_out, rev=False)

        if not jac:
            return x_out

        return x_out, log_jac_det

    def inverse(self, x, c=None, jac=True):
        # Permutation
        if self.with_permute:
            x = self._permute(x, rev=True)

        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # RQS Block
        a1 = tf.reshape(
            self.subnet(x1c), (x1c.shape[0], self.splits[1], 3 * self.num_bins + 1)
        )
        x2, j2 = self._rational_quadratic_spline(x2, a1, rev=True)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        if not jac:
            return x_out

        return x_out, log_jac_det
