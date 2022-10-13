""" Random Permutation """

from typing import Union
import numpy as np
import tensorflow as tf
from scipy.stats import special_ortho_group

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

    def call(self, x, c=None, jac=True):  # pylint: disable=W0221
        y = self.permute_function(x, self.w_perm)
        if jac:
            return y, 0.0

        return y

    def inverse(self, x, c=None, jac=True):  # pylint: disable=W0221
        y = self.permute_function(x, self.w_perm_inv)
        if jac:
            return y, 0.0

        return y

# pylint: disable=C0103
class SoftPermute(Transform):
    """Constructs a soft permutation, that stays fixed during training.
    Perfoms a rotation along the first (channel-) dimension for multi-dimenional tensors."""

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

        w = special_ortho_group.rvs(self.channels, random_state=seed)

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

    def call(self, x, c=None, jac=True):  # pylint: disable=W0221
        y = self.permute_function(x, self.w_perm)
        if jac:
            return y, 0.0

        return y

    def inverse(self, x, c=None, jac=True):  # pylint: disable=W0221
        y = self.permute_function(x, self.w_perm_inv)
        if jac:
            return y, 0.0

        return y

# pylint: disable=C0103
class SoftPermuteLearn(Transform):
    """Constructs a soft permutation, that is learnable in training.
    Perfoms a rotation along the first (channel-) dimension for multi-dimenional tensors.
    based on "Generalization of Euler Angles to N‐Dimensional Orthogonal Matrices"
    Journal of Mathematical Physics 13, 528 (1972); https://doi.org/10.1063/1.1666011
    David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
    Algorithm inspired by
    https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/utils/mathematics.py
    """

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

        # initialize k*(k-1)/2 angles. The majority is in [-pi/2, pi/2], but some are in [-pi, pi]
        # trainable parameters are unbounded, so we have to ensure the boundaries ourselves

        # which indices are in the larger domain:
        which_full = list(np.cumsum(-np.arange(self.channels-1))-1)
        num_all = int(self.channels * (self.channels-1) / 2)
        num_reduced = num_all - len(which_full)
        # initialize trainable parameters such that they cover final angle space more or less
        # uniformly. Found empirically based on subsequent transformations.
        init_reduced = np.random.randn(num_reduced)*1.5
        init_full = np.random.rand(len(which_full))*2. -1.

        self.perm_ang_train_red = self.add_weight(
            "perm_ang_train_red",
            shape=(num_reduced),
            initializer=tf.keras.initializers.Constant(init_reduced),
            trainable=True,
            dtype=tf.float64
        )
        self.perm_ang_train_full = self.add_weight(
            "perm_ang_train_full",
            shape=(len(which_full)),
            initializer=tf.keras.initializers.Constant(init_full),
            trainable=True,
            dtype=tf.float64
        )
        # ensure that it stays in reduced domain:
        self.perm_ang_train_red = (1./(1.+tf.math.exp(-self.perm_ang_train_red)))-0.5
        # build up full tensor with all angles:
        self.perm_ang = tf.zeros((num_all), dtype=tf.float64)
        indices_full = np.array(which_full) + num_all
        mask_full = np.zeros((num_all), dtype=bool)
        mask_full[indices_full] = True
        indices_red = np.arange(num_all)[~mask_full]
        self.perm_ang = tf.tensor_scatter_nd_update(self.perm_ang, indices_full.reshape(-1, 1),
                                                    self.perm_ang_train_full)
        self.perm_ang = tf.tensor_scatter_nd_update(self.perm_ang, indices_red.reshape(-1, 1),
                                                    self.perm_ang_train_red)
        self.perm_ang = self.perm_ang*np.pi

        self.w_perm = self._gea_orthogonal_from_angles_tf(self.perm_ang)
        self.w_perm_inv = tf.transpose(self.w_perm)

    def call(self, x, c=None, jac=True):  # pylint: disable=W0221
        y = self.permute_function(x, self.w_perm)
        if jac:
            return y, 0.0

        return y

    def inverse(self, x, c=None, jac=True):  # pylint: disable=W0221
        y = self.permute_function(x, self.w_perm_inv)
        if jac:
            return y, 0.0

        return y

    def _gea_orthogonal_from_angles_tf(self, angles_list):
        """
        Generalized Euler Angles
        Return the orthogonal matrix from its generalized angles

        Generalization of Euler Angles to N-Dimensional Orthogonal Matrices
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011

        Algorithm inspired by
        https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/utils/mathematics.py

        :param angles_list: List of angles, for a k-dimensional space the total number
                        of angles is k*(k-1)/2
        """

        b = tf.eye(2, dtype=angles_list.dtype)
        n = int(tf.sqrt(tf.cast(angles_list.shape, tf.float32)*8+1)/2+0.5)
        tmp = tf.identity(angles_list)

        # For SO(k) there are k*(k-1)/2 angles that are grouped in k-1 sets
        # { (k-1 angles), (k-2 angles), ... , (1 angle)}
        for i in range(1, n):
            angles = tf.concat((tmp[-i:], [np.pi/2]), 0)
            tmp = tmp[:-i]
            ma = self._gea_matrix_a_tf(angles)  # matrix i+1 x i+1
            b = tf.transpose(tf.linalg.matmul(b, ma, transpose_b=True))
            # We skip doing making a larger matrix for the last iteration, numpy is fine here
            if i < n-1:
                c = np.eye(i+2, i+2)
                c[:-1, :-1] = b
                b = c
        return b

    def _gea_matrix_a_tf(self, angles):
        """
        Generalized Euler Angles
        Return the parametric angles described on Eqs. 15-19 from the paper:

        Generalization of Euler Angles to N-Dimensional Orthogonal Matrices
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011

        Algorithm inspired by
        https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/utils/mathematics.py
        """
        n = len(angles)
        matrix_a = tf.eye(n, dtype=angles.dtype)
        # region I, eq. 16:
        matrix_a = tf.multiply(matrix_a, tf.math.cos(angles))
        # region II, eq. 17 tan:
        tan_vec = tf.math.tan(angles)
        # region II, eq. 17 cos:
        cos_vec = tf.math.cumprod(tf.math.cos(angles))
        # region II, eq. 17 all:
        matrix_a += tf.concat([tf.zeros((n, n-1), dtype=angles.dtype),
                               tf.reshape(tf.multiply(tan_vec, cos_vec), (n, 1))], 1)
        # region III, eq. 18 tan:
        region_iii_tan = -tf.multiply(tf.reshape(tf.math.tan(angles), (n, 1)),
                                      tf.reshape(tf.math.tan(angles), (1, n)))
        # region III, eq. 18, cos:
        #shifted_cos = tf.concat([cos_vec[:-1], tf.ones((1, ), dtype=angles.dtype)], 0)
        shifted_cos = tf.math.cumprod(tf.math.cos(angles), exclusive=True)
        region_iii_cos = tf.multiply(tf.reshape(cos_vec, (n, 1)),
                                     tf.reshape(1./shifted_cos, (1, n)))
        matrix_a += tf.linalg.band_part(tf.multiply(region_iii_tan, region_iii_cos), -1, 0) -\
            tf.linalg.band_part(tf.multiply(region_iii_tan, region_iii_cos), 0, 0)

        return matrix_a
