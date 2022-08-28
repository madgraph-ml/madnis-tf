""" Random Permutation """

from typing import Union
import numpy as np
import tensorflow as tf
from scipy.stats import special_ortho_group

from .base import Transform


class ActNorm(Transform):
    '''
    Transform for activation normalization [1].

    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    '''

    def __init__(self, 
        dims_in,
        dims_c=None,
        global_affine_init: float = 1.0,
        global_affine_type: str = "SOFTPLUS",
        eps: float = 1e-6,
    ):
        """
        Args:
            global_affine_init (float): Initial value for the global affine scaling. Defaults to 1.
            global_affine_type (str): ``'SIGMOID'``, ``'SOFTPLUS'``, or ``'EXP'``. 
                Defines the activation to be used on the beta for the global 
                affine scaling. Defaults to ``'SOFTPLUS'``.
            eps (float, optional): Small epsilon value. Defaults to 1e-6.
        """
        super().__init__(dims_in, dims_c)
        
        self.channels = self.dims_in[-1]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(self.dims_in) - 1
        self.sum_dims = tuple(range(1, 2 + self.input_rank))
        self.eps = eps
        
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

    def call(self, x, c=None, jac=True):
        scale = self.global_scale_activation(self.global_scale)
        z = x * scale + self.global_offset
        ldj = tf.reduce_sum((tf.math.log(scale)), axis=self.sum_dims) * self.ldj_multiplier(x)
        return z, ldj

    def inverse(self, z, c=None, jac=True):
        scale = self.global_scale_activation(self.global_scale)
        x = (z - self.global_offset) / scale
        ldj = (-1) * tf.reduce_sum((tf.math.log(scale)), axis=self.sum_dims) * self.ldj_multiplier(z)
        return x, ldj

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        raise tf.size(x[0,...,:1], out_type=x.dtype)