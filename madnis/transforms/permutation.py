""" Random Permutation """

from typing import Union
import numpy as np
import tensorflow as tf
from scipy.stats import special_ortho_group

from .base import Transform


# pylint: disable=C0103
class Permutation(Transform):
    """Base class for simple permutations
    Permutes along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in, dims_c=None, permutation=None, permutation_matrix=None):
        """
        Additional args in docstring of base class base.InvertibleModule.
        Args:
            only one of the two needed:
            - permutation: a permutation of range(dims_in)
            - permutation_matrix: a (dims_in x dims_in) matrix describing the permutation
        """
        super().__init__(dims_in, dims_c)

        self.channels = self.dims_in[0]
        self.input_rank = len(self.dims_in) - 1

        self.permute_function = lambda x, w: tf.linalg.matvec(w, x, transpose_a=True)

        # check inputs
        if permutation is None and permutation_matrix is None:
            raise ValueError("Permutation must be given, as permuted array XOR matrix!")
        if permutation is not None and permutation_matrix is not None:
            raise ValueError("Permutation must be given, as permuted array XOR matrix!")

        if permutation is not None:
            # Get the permutation matrix
            permutation_matrix = np.zeros((self.channels, self.channels))
            for i, j in enumerate(permutation):
                permutation_matrix[i, j] = 1.0

        self.w_perm = self.add_weight(
            "w_perm",
            shape=(*([1] * self.input_rank), self.channels, self.channels),
            initializer=tf.keras.initializers.Constant(permutation_matrix),
            trainable=False,
        )

        self.w_perm_inv = self.add_weight(
            "w_perm_inv",
            shape=(*([1] * self.input_rank), self.channels, self.channels),
            initializer=tf.keras.initializers.Constant(permutation_matrix.T),
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
class PermuteExchange(Permutation):
    """Constructs a permutation that just exchanges the sets A and B of the Coupling Layer.
    Permutes along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in, dims_c=None):
        """
        Additional args in docstring of base class base.InvertibleModule.
        Args:
        """
        self.channels = dims_in[0]

        # taken from CouplingTransform:
        self.split_len1 = self.channels // 2
        self.split_len2 = self.channels - self.channels // 2

        w = np.eye(self.channels, k=-self.split_len1) + np.eye(self.channels, k=self.split_len2)
        super().__init__(dims_in, dims_c, permutation_matrix=w)


# pylint: disable=C0103
class PermuteRandom(Permutation):
    """Constructs a random permutation, that stays fixed during training.
    Permutes along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in, dims_c=None, seed: Union[int, None] = None):
        """
        Additional args in docstring of base class base.InvertibleModule.
        Args:
          seed: Int seed for the permutation (numpy is used for RNG). If seed is
            None, do not reseed RNG.
        """
        self.channels = dims_in[0]
        if seed is not None:
            np.random.seed(seed)
        permutation = np.random.permutation(self.channels)

        # Get the permutation matrix
        w = np.zeros((self.channels, self.channels))
        for i, j in enumerate(permutation):
            w[i, j] = 1.0

        super().__init__(dims_in, dims_c, permutation_matrix=w)

# pylint: disable=C0103
class PermuteSoft(Permutation):
    """Constructs a soft permutation, that stays fixed during training.
    Perfoms a rotation along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in, dims_c=None, seed: Union[int, None] = None):
        """
        Additional args in docstring of base class base.InvertibleModule.
        Args:
          seed: Int seed for the permutation (numpy is used for RNG). If seed is
            None, do not reseed RNG.
        """
        self.channels = dims_in[0]
        w = special_ortho_group.rvs(self.channels, random_state=seed)
        super().__init__(dims_in, dims_c, permutation_matrix=w)

# pylint: disable=C0103
class PermuteSoftLearn(Transform):
    """Constructs a soft permutation, that is learnable in training.
    Perfoms a rotation along the first (channel-) dimension for multi-dimenional tensors.
    Rotations are parametrized by their Euler angles.
    Formulas are based on "Generalization of Euler Angles to N‐Dimensional Orthogonal Matrices"
    Journal of Mathematical Physics 13, 528 (1972); https://doi.org/10.1063/1.1666011
    David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
    Algorithm inspired by
    https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/utils/mathematics.py
    """

    def __init__(self, dims_in, dims_c=None):
        """
        Additional args in docstring of base class base.InvertibleModule.
        Args:
        """
        super().__init__(dims_in, dims_c)

        self.channels = self.dims_in[0]
        self.input_rank = len(self.dims_in) - 1

        self.permute_function = lambda x, w: tf.linalg.matvec(w, x, transpose_a=True)

        # initialize k*(k-1)/2 angles. The majority is in [-pi/2, pi/2], but some are in [-pi, pi]
        # trainable parameters are unbounded, so we have to ensure the boundaries ourselves

        # which indices are in the larger domain:
        self.which_full = list(np.cumsum(-np.arange(self.channels-1))-1)
        # number of all angles
        self.num_all = int(self.channels * (self.channels-1) / 2)
        # number of angles in reduced domain
        num_reduced = self.num_all - len(self.which_full)
        # initialize trainable parameters such that they cover final angle space more or less
        # uniformly. Found empirically based on subsequent transformations.
        init_reduced = np.random.randn(num_reduced)*1.5
        init_full = np.random.rand(len(self.which_full))*2. -1.

        self.perm_ang_train_red = self.add_weight(
            "perm_ang_train_red",
            shape=(num_reduced),
            initializer=tf.keras.initializers.Constant(init_reduced),
            trainable=True,
            dtype=tf.float64
        )
        self.perm_ang_train_full = self.add_weight(
            "perm_ang_train_full",
            shape=(len(self.which_full)),
            initializer=tf.keras.initializers.Constant(init_full),
            trainable=True,
            dtype=tf.float64
        )
        # initialize to later show angles that describe current permutation
        self.perm_ang = None
        self.w_perm = self._translate_to_matrix()

    def call(self, x, c=None, jac=True):  # pylint: disable=W0221
        self.w_perm = self._translate_to_matrix()
        y = self.permute_function(x, self.w_perm)
        if jac:
            return y, 0.0

        return y

    def inverse(self, x, c=None, jac=True):  # pylint: disable=W0221
        self.w_perm = self._translate_to_matrix()
        w_perm_inv = tf.transpose(self.w_perm)
        y = self.permute_function(x, w_perm_inv)
        if jac:
            return y, 0.0

        return y

    def _translate_to_matrix(self):
        """ translates the trainable parameters to angles in the right domain and then to
            the rotation matrix
        """
        # ensure that it stays in reduced domain:
        perm_ang_train_red_t = tf.math.sigmoid(self.perm_ang_train_red)-0.5
        # build up full tensor with all angles:
        perm_ang = tf.zeros((self.num_all), dtype=tf.float64)
        indices_full = np.array(self.which_full) + self.num_all
        mask_full = np.zeros((self.num_all), dtype=bool)
        mask_full[indices_full] = True
        indices_red = np.arange(self.num_all)[~mask_full]
        perm_ang = tf.tensor_scatter_nd_add(perm_ang, indices_full.reshape(-1, 1),
                                            self.perm_ang_train_full)
        perm_ang = tf.tensor_scatter_nd_add(perm_ang, indices_red.reshape(-1, 1),
                                            perm_ang_train_red_t)
        self.perm_ang = perm_ang*np.pi
        w_perm = self._gea_orthogonal_from_angles_tf(self.perm_ang)
        return w_perm

    def _gea_orthogonal_from_angles_tf(self, angles_list):
        """
        Generalized Euler Angles
        Return the orthogonal matrix from its generalized angles

        Formulas are based on "Generalization of Euler Angles to N-Dimensional Orthogonal Matrices"
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011

        Algorithm adapted from numpy version at
        https://github.com/MaterialsDiscovery/PyChemia/blob/master/pychemia/utils/mathematics.py

        :param angles_list: List of angles, for a k-dimensional space the total number
                        of angles is k*(k-1)/2
        """

        b = tf.eye(2, dtype=angles_list.dtype)
        n = self.channels
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
                c = tf.pad(b, tf.constant([[0, 1,], [0, 1]]), "CONSTANT")
                corr = tf.eye(i+2, dtype=angles_list.dtype) -\
                    tf.pad(tf.eye(i+1, dtype=angles_list.dtype), tf.constant([[0, 1,], [0, 1]]),
                           "CONSTANT")
                b = c + corr
        return b

    def _gea_matrix_a_tf(self, angles):
        """
        Generalized Euler Angles
        Return the parametric angles described on Eqs. 15-19 from the paper:

        Formulas are based on "Generalization of Euler Angles to N-Dimensional Orthogonal Matrices"
        David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
        Journal of Mathematical Physics 13, 528 (1972)
        doi: 10.1063/1.1666011

        Algorithm adapted from numpy version at
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
        shifted_cos = tf.math.cumprod(tf.math.cos(angles), exclusive=True)
        region_iii_cos = tf.multiply(tf.reshape(cos_vec, (n, 1)),
                                     tf.reshape(1./shifted_cos, (1, n)))
        matrix_a += tf.linalg.band_part(tf.multiply(region_iii_tan, region_iii_cos), -1, 0) -\
            tf.linalg.band_part(tf.multiply(region_iii_tan, region_iii_cos), 0, 0)

        return matrix_a
