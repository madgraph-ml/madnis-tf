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

        #w = special_ortho_group.rvs(self.channels, random_state=seed)
        # initialize k*(k-1)/2 angles in [-0.5, 0.5]
        ang = np.random.rand(int(self.channels * (self.channels-1) / 2)) - 0.5
        # not all of them are in [-0.5, 0.5], some are in [-1, 1]
        which_mult = list(np.cumsum(-np.arange(self.channels-1))-1)
        ang[which_mult] = ang[which_mult]*2.
        #self.perm_ang = self.add_weight(
        #    "perm_ang",
        #    shape=(*([1] * self.input_rank), self.channels, self.channels),
        #    initializer=tf.keras.initializers.Constant(ang),
        #    trainable=True,
        #)
        self.perm_ang = self.add_weight(
            "perm_ang",
            shape=(int(self.channels * (self.channels-1) / 2)),
            initializer=tf.keras.initializers.Constant(ang),
            trainable=True,
            dtype=tf.float64
        )

        # find inverse in function?
        #self.w_perm_inv = self.add_weight(
        #    "w_perm_inv",
        #    shape=(*([1] * self.input_rank), self.channels, self.channels),
        #    initializer=tf.keras.initializers.Constant(w.T),
        #    trainable=False,
        #)
        print(self.perm_ang)
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

    def _gea_orthogonal_from_angles(self, angles_list):
        """
        Generalized Euler Angles
        Return the orthogonal matrix from its generalized angles

        Generalization of Euler Angles to N-Dimensional Orthogonal Matrices
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011

        :param angles_list: List of angles, for a k-dimensional space the total number
                        of angles is k*(k-1)/2
        """

        b = np.eye(2)
        n = int(np.sqrt(len(angles_list)*8+1)/2+0.5)
        tmp = np.array(angles_list)

        # For SO(k) there are k*(k-1)/2 angles that are grouped in k-1 sets
        # { (k-1 angles), (k-2 angles), ... , (1 angle)}
        for i in range(1, n):
            angles = np.concatenate((tmp[-i:], [np.pi/2]))
            tmp = tmp[:-i]
            ma = self._gea_matrix_a(angles)  # matrix i+1 x i+1
            b = np.dot(b, ma.T).T
            # We skip doing making a larger matrix for the last iteration
            if i < n-1:
                c = np.eye(i+2, i+2)
                c[:-1, :-1] = b
                b = c
        return b

    def _gea_orthogonal_from_angles_tf(self, angles_list):
        """
        Generalized Euler Angles
        Return the orthogonal matrix from its generalized angles

        Generalization of Euler Angles to N-Dimensional Orthogonal Matrices
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011

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
            # We skip doing making a larger matrix for the last iteration
            if i < n-1:
                c = np.eye(i+2, i+2)
                c[:-1, :-1] = b
                b = c
        return b

    def _gea_matrix_a(self, angles):
        """
        Generalized Euler Angles
        Return the parametric angles described on Eqs. 15-19 from the paper:

        Generalization of Euler Angles to N-Dimensional Orthogonal Matrices
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011

        """
        n = len(angles)
        matrix_a = np.eye(n)

        for i in range(n-1):
            matrix_a[i][i] = np.cos(angles[i])
            matrix_a[i][n-1] = np.tan(angles[i])
            for j in range(i+1):
                matrix_a[i][n-1] *= np.cos(angles[j])

        for i in range(n):
            for k in range(n-1):
                if i > k:
                    matrix_a[i][k] = -np.tan(angles[i])*np.tan(angles[k])
                    for l in range(k, i+1):
                        matrix_a[i][k] *= np.cos(angles[l])

        matrix_a[n - 1][n - 1] = np.tan(angles[n-1])
        for j in range(n):
            matrix_a[n - 1][n - 1] *= np.cos(angles[j])

        return matrix_a

    def _gea_matrix_a_tf(self, angles):
        """
        Generalized Euler Angles
        Return the parametric angles described on Eqs. 15-19 from the paper:

        Generalization of Euler Angles to N-Dimensional Orthogonal Matrices
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011

        """
        n = len(angles)
        matrix_a = tf.eye(n, dtype=angles.dtype)
        # region I, eq. 16:
        matrix_a = tf.multiply(matrix_a, tf.math.cos(angles))
        # region II, eq. 17 tan, excl. last row:
        #tan_vec = tf.concat([tf.math.tan(angles)[:n-1], tf.zeros((1, ), dtype=angles.dtype)], 0)
        tan_vec = tf.math.tan(angles)
        # region II, eq. 17 cos, excl. last row:
        #cos_vec = tf.concat([tf.math.cumprod(tf.math.cos(angles[:n-1])),
        #                     tf.zeros((1, ), dtype=angles.dtype)], 0)
        cos_vec = tf.math.cumprod(tf.math.cos(angles))
        matrix_a += tf.concat([tf.zeros((n, n-1), dtype=angles.dtype),
                               tf.reshape(tf.multiply(tan_vec, cos_vec), (n, 1))], 1)
        print("matrix a tf", matrix_a)

        matrix_np = np.eye(n)
        for i in range(n-1):
            matrix_np[i][i] = np.cos(angles[i])
            matrix_np[i][n-1] = np.tan(angles[i])
            for j in range(i+1):
                matrix_np[i][n-1] *= np.cos(angles[j])
        #for i in range(n):
        #    for k in range(n-1):
        #        if i > k:
        #            matrix_a[i][k] = -np.tan(angles[i])*np.tan(angles[k])
        #            for l in range(k, i+1):
        #                matrix_a[i][k] *= np.cos(angles[l])

        matrix_np[n - 1][n - 1] = np.tan(angles[n-1])
        for j in range(n):
            matrix_np[n - 1][n - 1] *= np.cos(angles[j])

        print("matrix np", matrix_np)
        return matrix_a
