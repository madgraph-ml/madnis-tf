""" Base class for networks """

from typing import Dict
import tensorflow as tf


# pylint: disable=C0103
class SubNet(tf.keras.layers.Layer):
    """Base class to implement various subnetworks.  It takes care of
    checking the dimensions. Each child class only has
    to implement the _network() method.
    """

    def __init__(self, meta: Dict, channels_in: int, channels_out: int):
        """
        Args:
          meta:
            Dictionary with defining parameters
            to construct the network.
          channels_in_in:
            Number of input channels.
          channels_in_out:
            Number of output channels.
        """
        super().__init__()
        self.meta = meta
        self.channels_in = channels_in
        self.channels_out = channels_out

    def call(self, x):  # pylint: disable=W0221
        """
        Perform a forward pass through this layer.
        Args:
          x: input data (array-like of one or more tensors)
            of the form: ``x = input_tensor_1``.
        """
        out = self._network(x)
        return out

    def _network(self, x):
        """The network operation used in the call() function.
        Args:
          x (Tensor): the input tensor.
        Returns:
          y (Tensor): the output tensor.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _network(...) method"
        )

    def build(self, input_shape):
        """
        Helps to prevent wrong usage of modules and helps for debugging.
        """
        assert (
            input_shape[-1] == self.channels_in
        ), f"Channel dimension of input ({input_shape[-1]}) and given input channels ({self.channels_in}) don't agree."

        super().build(input_shape)

    def get_config(self):
        config = {
            "meta": self.meta,
            "channels_in": self.channels_in,
            "channels_out": self.channels_out,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
