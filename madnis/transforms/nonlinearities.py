""" Sigmoid functions with inverse and gradient """

import tensorflow as tf
from .base import Transform, InverseTransform
from ..utils.tfutils import sum_except_batch


class Sigmoid(Transform):
    """
    Sigmoid function which can also return the logarithm
    of the determinant which is suitable for INN
    architectures.
    """

    def __init__(self, dims_in, dims_c=None, temperature=1.0, epsilon=1e-8):
        """
        Args:
            epsilon (float, optional): Regularization of the logarithm in the inverse. Defaults to 1e-8.
        """

        super().__init__(dims_in, dims_c)

        self.epsilon = epsilon
        self._dtype = tf.keras.backend.floatx()
        self.temperature = tf.constant(temperature, dtype=self._dtype)

    def call(self, x, c=None, jac=True):

        # --- Forward pass --- #
        x = self.temperature * x
        z = tf.sigmoid(x)
        if jac:
            log_det = sum_except_batch(
                tf.math.log(self.temperature)
                - tf.math.softplus(-x)
                - tf.math.softplus(x)
            )
            return z, log_det
        return z

    def inverse(self, z, c=None, jac=True):

        # --- Inverse pass --- #
        z = tf.clip_by_value(
            z, clip_value_min=self.epsilon, clip_value_max=1 - self.epsilon
        )
        x = (1 / self.temperature) * (tf.math.log(z) - tf.math.log1p(-z))
        if jac:
            log_det = -sum_except_batch(
                tf.math.log(self.temperature)
                - tf.math.softplus(-self.temperature * x)
                - tf.math.softplus(self.temperature * x)
            )
            return x, log_det
        return x

    def get_config(self):
        new_config = {"epsilon": self.epsilon, "temperature": self.temperature}
        config = super().get_config()
        config.update(new_config)
        return config


class Logit(InverseTransform):
    """
    Logit function which can also return the logarithm
    of the determinant which is suitable for INN
    architectures.
    """

    def __init__(self, dims_in, dims_c=None, temperature=1.0, epsilon=1e-8):
        """
        Args:
            epsilon (float, optional): Regularization of the logarithm in the inverse. Defaults to 1e-8.
        """
        super().__init__(
            Sigmoid(dims_in, dims_c, temperature=temperature, epsilon=epsilon)
        )

        self.epsilon = epsilon
        self._dtype = tf.keras.backend.floatx()
        self.temperature = tf.constant(temperature, dtype=self._dtype)

    def get_config(self):
        new_config = {"epsilon": self.epsilon, "temperature": self.temperature}
        config = super().get_config()
        config.update(new_config)
        return config


class Tanh(Transform):
    """
    Tanh function which can also return the logarithm
    of the determinant which is suitable for flow
    architectures.
    """

    def __init__(self, dims_in, dims_c=None):
        super().__init__(dims_in, dims_c)
        self._dtype = tf.keras.backend.floatx()

    def call(self, x, c=None, jac=True):

        # --- Forward pass --- #
        z = tf.math.tanh(x)
        if jac:
            log_det = sum_except_batch(tf.math.log(1 - z ** 2), num_batch_dims=1)
            return z, log_det
        return z

    def inverse(self, z, c=None, jac=True):
        if tf.reduce_min(z) <= -1 or tf.reduce_max(z) >= 1:
            raise ValueError("Input is not in the correct domain")

        # --- Inverse pass --- #
        x = tf.math.atanh(z)
        if jac:
            log_det = sum_except_batch(-tf.math.log(1 - z ** 2), num_batch_dims=1)
            return x, log_det
        return x


class ArcTanh(InverseTransform):
    """
    ArcTanh function which can also return the logarithm
    of the determinant which is suitable for flow
    architectures.
    """

    def __init__(self, dims_in, dims_c=None):
        super().__init__(Tanh(dims_in, dims_c))
