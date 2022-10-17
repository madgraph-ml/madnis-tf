"""All in one coupling Block"""

import warnings
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from scipy.stats import special_ortho_group

from .base import CouplingTransform


# pylint: disable=C0103, R1729, E1124, E1120, W0221
class AllInOneBlock(CouplingTransform):
    """Module combining the most common operations in a normalizing flow or
    similar model. It combines affine coupling, permutation, and
    global affine transformation ('ActNorm'). It can also be used as
    GIN coupling block and use an inverted pre-permutation.
    The affine transformation includes a soft clamping mechanism,
    first used in Real-NVP.
    """

    def __init__(
            self,
            dims_in,
            dims_c=None,
            subnet_meta: Dict = None,
            subnet_constructor: callable = None,
            clamp: float = 2.0,
            gin_block: bool = False,
            global_affine_init: float = 1.0,
            global_affine_type: str = "SOFTPLUS",
            permute_soft: bool = False,
            reverse_permutation: bool = False,
            seed: Union[int, None] = None,
    ):
        """
        Args:
          subnet_meta:
            meta defining the structure of the subnet like number of layers,
            units, activation functions etc.
          subnet_constructor:
            class or callable ``f``, called as
            ``f(meta, channels_in, channels_out)`` and
            should return a tf.keras.layer.Layer.
            Predicts coupling coefficients :math:`s, t`.
          clamp:
            Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          gin_block:
            Turn the block into a GIN block from Sorrenson et al, 2019.
            Makes it so that the coupling operations as a whole is
            volume preserving.
          global_affine_init:
            Initial value for the global affine scaling.
          global_affine_type:
            ``'SIGMOID'``, ``'SOFTPLUS'``, or ``'EXP'``. Defines the activation
            to be used on the beta for the global affine scaling.
          permute_soft:
            bool, whether to sample the permutation matrix `R` from `SO(N)`,
            or to use hard permutations instead. Note, ``permute_soft=True``
            is very slow when working with >512 dimensions.
          reverse_permutation:
            Reverse the permutation before the block, as introduced by Putzky
            et al, 2019. Turns on the :math:`R^{-1}` pre-multiplication above.
          seed:
            Int seed for the permutation (numpy is used for RNG). If seed is
            None, do not reseed RNG.
        """
        super().__init__(dims_in, dims_c, clamp, clamp_activation=(lambda u: u))

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

        self.splits = [self.split_len1, self.split_len2]

        self.permute_function = lambda x, w: tf.linalg.matvec(w, x, transpose_a=True)

        self.GIN = gin_block
        self.reverse_pre_permute = reverse_permutation

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # global_scale_activation(global_scale) = global_affine_init
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        if global_affine_type == "SIGMOID":
            global_scale = 2.0 - np.log(10.0 / global_affine_init - 1.0)
            self.global_scale_activation = lambda a: 10 * tf.sigmoid(a - 2.0)
        elif global_affine_type == "SOFTPLUS":
            global_scale = 2.0 * np.log(np.exp(0.5 * 10.0 * global_affine_init) - 1)
            self.global_scale_activation = (
                lambda a: 0.1 * 2.0 * tf.math.softplus(0.5 * a)
            )
        elif global_affine_type == "EXP":
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = tf.exp
        else:
            raise ValueError(
                'Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"'
            )

        self.global_scale = self.add_weight(
            "global_scale",
            shape=(1, *([1] * self.input_rank), self.channels),
            initializer=tf.keras.initializers.Constant(global_scale),
            trainable=True,
        )

        self.global_offset = self.add_weight(
            "global_offset",
            shape=(1, *([1] * self.input_rank), self.channels),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

        # Define the permutation matrix
        # either some random rotation matrix
        # or a hard permutation instead.
        if permute_soft and self.channels > 512:
            warnings.warn(
                (
                    "Soft permutation will take a very long time to initialize "
                    f"with {self.channels} feature channels. Consider using hard permutation instead."
                )
            )

        if permute_soft:
            w = special_ortho_group.rvs(self.channels, random_state=seed)
        else:
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

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )

        self.subnet = subnet_constructor(
            subnet_meta, self.splits[0] + self.condition_length, 2 * self.splits[1]
        )
        self.last_jac = None

    def _permute(self, x, rev=False):
        """Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation."""
        if self.GIN:
            scale = 1.0
            perm_log_jac = 0.0
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = tf.reduce_sum((tf.math.log(scale)), axis=self.sum_dims)

        if rev:
            x_permute = (
                self.permute_function(x, self.w_perm_inv) - self.global_offset
            ) / scale
            return x_permute, perm_log_jac

        x_permute = self.permute_function(x * scale + self.global_offset, self.w_perm)
        return x_permute, perm_log_jac

    def _pre_permute(self, x, rev=False):
        """Permutes before the coupling block, only used if
        reverse_permutation is set"""
        if rev:
            return self.permute_function(x, self.w_perm)

        return self.permute_function(x, self.w_perm_inv)

    def _affine(self, x, a, rev=False):
        """Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the affine coupling operation.
        Returns both the transformed inputs and the LogJacDet."""

        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        a *= 0.1
        s, t = tf.split(a, 2, -1)
        sub_jac = self.clamp * tf.math.tanh(s)
        if self.GIN:
            sub_jac -= tf.reduce_mean(sub_jac, axis=self.sum_dims, keepdims=True)

        if not rev:
            return x * tf.exp(sub_jac) + t, tf.reduce_sum(sub_jac, axis=self.sum_dims)

        return (x - t) * tf.exp(-sub_jac), -tf.reduce_sum(sub_jac, axis=self.sum_dims)

    def call(self, x, c=None, jac=True):
        # Pre-permute
        if self.reverse_pre_permute:
            x = self._pre_permute(x, rev=False)

        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # Affine
        a1 = self.subnet(x1c)
        x2, j2 = self._affine(x2, a1)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        # Act Norm
        x_out, global_scaling_jac = self._permute(x_out, rev=False)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        # Fix n_pixels
        # n_pixels = tf.size(x) / self.channels
        n_pixels = tf.size(x_out[0,...,:1], out_type=x_out.dtype)
        log_jac_det += 1 * global_scaling_jac * n_pixels

        if not jac:
            return x_out

        return x_out, log_jac_det
    
    def inverse(self, x, c=None, jac=True):
        # Act Norm
        x, global_scaling_jac = self._permute(x, rev=True)

        x1, x2 = tf.split(x, self.splits, axis=-1)

        if self.conditional:
            x1c = tf.concat([x1, *c], -1)
        else:
            x1c = x1

        # affine block
        a1 = self.subnet(x1c)
        x2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2
        x_out = tf.concat([x1, x2], -1)

        # Undo pre-permutation
        if self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        # Fix n_pixels
        n_pixels = tf.size(x_out[0,...,:1], out_type=x_out.dtype)
        log_jac_det += (-1) * global_scaling_jac * n_pixels

        if not jac:
            return x_out

        return x_out, log_jac_det
