""" Sigmoid functions with inverse and gradient """

import tensorflow as tf
from .base import Transform, InverseTransform


class Sigmoid(Transform):
    """"
    Sigmoid function which can also return the logarithm
    of the determinant which is suitable for INN
    architectures.
    """
    def __init__(self, dims_in, dims_c=None, epsilon=1e-8):
        """
        Args:
            epsilon (float, optional): Regularization of the logarithm in the inverse. Defaults to 1e-8.
        """

        super().__init__(dims_in, dims_c)

        self.epsilon = epsilon

    def call(self, x, c=None, jac=True):

        # --- Forward pass --- #
        y = tf.sigmoid(x)
        if jac:
            grad = y * (1-y)
            det = tf.math.reduce_prod(grad, axis=1)
            log_det = tf.math.log(det)
            return y, log_det
        return y
        
    def inverse(self, x, c=None, jac=True):

        # --- Inverse pass --- #
        x = tf.clip_by_value(x, clip_value_min=self.epsilon, clip_value_max=1-self.epsilon)
        y = tf.math.log(x/(1-x))
        if jac:
            grad = 1/(x * (1-x))
            det = tf.math.reduce_prod(grad,  axis=1)
            log_det = tf.math.log(det)
            return y, log_det
        return y

    def get_config(self):
        new_config = {
            'epsilon': self.epsilon
        }
        config = super().get_config()
        config.update(new_config)
        return config
    

class Logit(InverseTransform):
    """"
    Logit function which can also return the logarithm
    of the determinant which is suitable for INN
    architectures.
    """
    def __init__(self, dims_in, dims_c=None, epsilon=1e-8):
        """
        Args:
            epsilon (float, optional): Regularization of the logarithm in the inverse. Defaults to 1e-8.
        """
        super().__init__(Sigmoid(dims_in, dims_c, epsilon=epsilon))

        self.epsilon = epsilon

    def get_config(self):
        new_config = {
            'epsilon': self.epsilon
        }
        config = super().get_config()
        config.update(new_config)
        return config
