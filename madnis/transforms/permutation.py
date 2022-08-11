""" Random Permutation """

from typing import Union
import numpy as np
import tensorflow as tf

from .base import Transform


# pylint: disable=C0103
class PermuteRandom(Transform):
    """Constructs a random permutation, that stays fixed during training.
    Permutes along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in, dims_c=None, seed: Union[int, None] = None):
        """
        Additional args in docstring of base class base.InvertibleModule.
        Args:
          seed: Int seed for the permutation (numpy is used for RNG). If seed is
            None, do not reseed RNG.
        """
        super().__init__(dims_in, dims_c)

        self.channels = self.dims_in[0]
        self.input_rank = len(self.dims_in) - 1

        self.permute_function = lambda x, w: tf.linalg.matvec(w, x, transpose_a=True)

        if seed is not None:
            np.random.seed(seed)
        permutation = np.random.permutation(self.channels)

        # Get the permutation matrix
        w = np.zeros((self.channels, self.channels))
        for i, j in enumerate(permutation):
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

    def call(self, x, jac=True):  # pylint: disable=W0221
        y = self.permute_function(x, self.w_perm)
        if jac:
            return y, 0.0

        return y

    def inverse(self, x, jac=True):  # pylint: disable=W0221
        y = self.permute_function(x, self.w_perm_inv)
        if jac:
            return y, 0.0

        return y
