"""Base Transform Class"""

from typing import Tuple, Iterable
import tensorflow as tf


class Transform(tf.keras.layers.Layer):
    """
    Generic transform invertible
    neural network structures.

    Used to implement `Nice`, `Glow`, `RNVP` etc.
    """

    def __init__(self, dims_in: Tuple[int], dims_c: Iterable[Tuple[int]] = None):
        """
        Args:
            dims_in: a tuple specifying the shape of the input,
                     excluding the batch dimension, to this
                     operator: ``dims_in = (dim_0,..., channels)``
            dims_c:  a list of tuples specifying the shape
                     of the conditions to this operator,
                     excluding the batch dimension

        ** Note  to implementors:**

        - The shapes are in the standard TensorFlow 'channels_last' format.
        """
        super().__init__()
        if dims_c is None:
            dims_c = []
        self.dims_in = tuple(dims_in)
        self.dims_c = list(dims_c)

    def call(  # pylint: disable=W0221
            self,
            x: tf.Tensor,
            c: Iterable[tf.Tensor] = None,
            jac: bool = True,
    ):
        """
        Perform a forward pass through this layer.
        Args:
            x:      input data (array-like of one or more tensors)
            c:      conditioning data (array-like of none or more tensors)
            jac:    return Jacobian associated to the direction
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide call(...) method"
        )

    def inverse(  # pylint: disable=W0221
            self,
            z: tf.Tensor,
            c: Iterable[tf.Tensor] = None,
            jac: bool = True,
    ):
        """
        Perform a backward pass through this layer.
        Args:
            z:      input data (array-like of one or more tensors)
            c:      conditioning data (array-like of none or more tensors)
            jac:    return Jacobian associated to the direction
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide inverse(...) method"
        )

    def get_config(self):
        "Needed within TensorFlow to serialize this layer"
        config = {"dims_in": self.dims_in, "dims_c": self.dims_c}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InverseTransform(Transform):
    """Creates a transform that is the inverse of a given transform."""

    def __init__(self, transform: Transform):
        """Constructor.
        Args:
            transform: An object of type `Transform`.
        """
        super().__init__(transform.dims_in, transform.dims_c)
        self._transform = transform

    def call(self, x, c=None, jac=True):
        return self._transform.inverse(x, c=c, jac=jac)

    def inverse(self, x, c=None, jac=True):
        return self._transform(x, c=c, jac=jac)
