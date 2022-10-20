""" RQS Coupling Blocks """

from typing import Dict, Union
import tensorflow as tf

from .base import CouplingTransform
from ..functional import splines
from ...utils.tfutils import sum_except_batch

import warnings
import numpy as np


class LinearSplineBlock(CouplingTransform):
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
    ):

        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=(lambda u: u))

        self.splits = [self.split_len1, self.split_len2]

        # Definitions for spline
        self.num_bins = num_bins
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )

        self.subnet = subnet_constructor(
            subnet_meta,
            self.splits[0] + self.condition_length,
            self._output_dim_multiplier() * self.splits[1],
        )

    def _output_dim_multiplier(self):
        return self.num_bins

    def _elementwise_function(self, x, a, rev=False):
        tf.debugging.assert_equal(tf.shape(a)[-1], self._output_dim_multiplier())
        y, ldj_elementwise = splines.unconstrained_linear_spline(
            x,
            a,
            inverse=rev,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
        )

        ldj = sum_except_batch(ldj_elementwise)

        return y, ldj

    def call(self, x, c=None, jac=True):
        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # Linear Spline Block
        a1 = tf.reshape(
            self.subnet(x1c),
            (tf.shape(x1c)[0], self.splits[1], self._output_dim_multiplier()),
        )
        x2, j2 = self._elementwise_function(x2, a1, rev=False)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        if not jac:
            return x_out

        return x_out, log_jac_det

    def inverse(self, x, c=None, jac=True):
        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # Linear Spline Block
        a1 = tf.reshape(
            self.subnet(x1c),
            (tf.shape(x1c)[0], self.splits[1], self._output_dim_multiplier()),
        )
        x2, j2 = self._elementwise_function(x2, a1, rev=True)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        if not jac:
            return x_out

        return x_out, log_jac_det
    

class QuadraticSplineBlock(CouplingTransform):
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
    ):

        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=(lambda u: u))

        self.splits = [self.split_len1, self.split_len2]

        # Definitions for spline
        self.num_bins = num_bins
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )

        self.subnet = subnet_constructor(
            subnet_meta,
            self.splits[0] + self.condition_length,
            self._output_dim_multiplier() * self.splits[1],
        )

    def _coupling1(self, x1, u2, rev=False):
        pass

    def _coupling2(self, x2, u1, rev=False):
        pass

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 1

    def _elementwise_function(self, x, a, rev=False):
        tf.debugging.assert_equal(tf.shape(a)[-1], self._output_dim_multiplier())
        
        # split into different contributions
        unnormalized_widths = a[..., :self.num_bins]
        unnormalized_heights = a[..., self.num_bins:]
        
        y, ldj_elementwise = splines.unconstrained_quadratic_spline(
            x,
            unnormalized_widths,
            unnormalized_heights,
            inverse=rev,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
        )

        ldj = sum_except_batch(ldj_elementwise)

        return y, ldj

    def call(self, x, c=None, jac=True):
        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # Quadrartic Spline Block
        a1 = tf.reshape(
            self.subnet(x1c),
            (tf.shape(x1c)[0], self.splits[1], self._output_dim_multiplier()),
        )
        x2, j2 = self._elementwise_function(x2, a1, rev=False)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        if not jac:
            return x_out

        return x_out, log_jac_det

    def inverse(self, x, c=None, jac=True):
        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # Quadrartic Spline Block
        a1 = tf.reshape(
            self.subnet(x1c),
            (tf.shape(x1c)[0], self.splits[1], self._output_dim_multiplier()),
        )
        x2, j2 = self._elementwise_function(x2, a1, rev=True)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        if not jac:
            return x_out

        return x_out, log_jac_det


class RationalQuadraticSplineBlock(CouplingTransform):
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
    ):

        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=(lambda u: u))

        self.splits = [self.split_len1, self.split_len2]

        # Definitions for spline
        self.num_bins = num_bins
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )

        self.subnet = subnet_constructor(
            subnet_meta,
            self.splits[0] + self.condition_length,
            self._output_dim_multiplier() * self.splits[1],
        )

    def _coupling1(self, x1, u2, rev=False):
        pass

    def _coupling2(self, x2, u1, rev=False):
        pass

    def _output_dim_multiplier(self):
        return 3 * self.num_bins + 1

    def _elementwise_function(self, x, a, rev=False):
        tf.debugging.assert_equal(tf.shape(a)[-1], self._output_dim_multiplier())

        # split into different contributions
        unnormalized_widths = a[..., : self.num_bins]
        unnormalized_heights = a[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = a[..., 2 * self.num_bins :]

        y, ldj_elementwise = splines.unconstrained_rational_quadratic_spline(
            x,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=rev,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
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
            self.subnet(x1c),
            (tf.shape(x1c)[0], self.splits[1], self._output_dim_multiplier()),
        )
        x2, j2 = self._elementwise_function(x2, a1, rev=False)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        if not jac:
            return x_out

        return x_out, log_jac_det

    def inverse(self, x, c=None, jac=True):
        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # RQS Block
        a1 = tf.reshape(
            self.subnet(x1c),
            (tf.shape(x1c)[0], self.splits[1], self._output_dim_multiplier()),
        )
        x2, j2 = self._elementwise_function(x2, a1, rev=True)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        if not jac:
            return x_out

        return x_out, log_jac_det
    
    
class CubicSplineBlock(CouplingTransform):
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
    ):

        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=(lambda u: u))

        self.splits = [self.split_len1, self.split_len2]

        # Definitions for spline
        self.num_bins = num_bins
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )

        self.subnet = subnet_constructor(
            subnet_meta,
            self.splits[0] + self.condition_length,
            self._output_dim_multiplier() * self.splits[1],
        )

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 2

    def _elementwise_function(self, x, a, rev=False):
        tf.debugging.assert_equal(tf.shape(a)[-1], self._output_dim_multiplier())

        # split into different contributions
        unnormalized_widths = a[..., :self.num_bins]
        unnormalized_heights = a[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = a[..., 2*self.num_bins:2*self.num_bins+1]
        unnorm_derivatives_right = a[..., 2*self.num_bins+1:]

        y, ldj_elementwise = splines.unconstrained_cubic_spline(
            x,
            unnormalized_widths,
            unnormalized_heights,
            unnorm_derivatives_left,
            unnorm_derivatives_right,
            inverse=rev,
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
        )

        ldj = sum_except_batch(ldj_elementwise)

        return y, ldj

    def call(self, x, c=None, jac=True):
        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # Cubic Spline Block
        a1 = tf.reshape(
            self.subnet(x1c),
            (tf.shape(x1c)[0], self.splits[1], self._output_dim_multiplier()),
        )
        x2, j2 = self._elementwise_function(x2, a1, rev=False)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        if not jac:
            return x_out

        return x_out, log_jac_det

    def inverse(self, x, c=None, jac=True):
        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # Cubic Spline Block
        a1 = tf.reshape(
            self.subnet(x1c),
            (tf.shape(x1c)[0], self.splits[1], self._output_dim_multiplier()),
        )
        x2, j2 = self._elementwise_function(x2, a1, rev=True)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        if not jac:
            return x_out

        return x_out, log_jac_det
