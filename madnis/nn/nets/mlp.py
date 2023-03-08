""" MLP subnetwork """

import tensorflow as tf
from .base import SubNet


class MLP(SubNet):
    """
    Creates a dense subnetwork
    which can be used within the invertible modules.
    """

    def __init__(self, meta, channels_in, channels_out, initialize_zero=False):
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
        super().__init__(meta, channels_in, channels_out)

        # which activation
        if isinstance(meta["activation"], str):
            if meta["activation"] == "relu":
                activation = tf.keras.activations.relu
            elif meta["activation"] == "elu":
                activation = tf.keras.activations.elu
            elif meta["activation"] == "leakyrelu":
                activation = tf.keras.layers.LeakyReLU()
            elif meta["activation"] == "tanh":
                activation = tf.keras.activations.tanh
            else:
                raise ValueError(f'Unknown activation "{meta["activation"]}"')
        else:
            activation = meta["activation"]

        layer_constructor = meta.get("layer_constructor", tf.keras.layers.Dense)

        # Define the layers
        self.hidden_layers = [
            layer_constructor(
                self.meta["units"],
                activation=activation,
                kernel_initializer=self.meta["initializer"],
            )
            for i in range(self.meta["layers"])
        ]

        self.dense_out = layer_constructor(
            self.channels_out, kernel_initializer=self.meta["initializer"]
        )

    def _network(self, x):
        """The used layers in this Subnetwork.
        Returns:
            layers (tf.keras.layers): Some stacked keras layers.
        """
        for layer in self.hidden_layers:
            x = layer(x)

        y = self.dense_out(x)
        return y
    
class MLPZero(SubNet):
    """
    Creates a dense subnetwork
    which can be used within the invertible modules
    that is initialized with zero weights and bias
    in the last layer.
    """

    def __init__(self, meta, channels_in, channels_out, initialize_zero=False):
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
        super().__init__(meta, channels_in, channels_out)

        # which activation
        if isinstance(meta["activation"], str):
            if meta["activation"] == "relu":
                activation = tf.keras.activations.relu
            elif meta["activation"] == "elu":
                activation = tf.keras.activations.elu
            elif meta["activation"] == "leakyrelu":
                activation = tf.keras.layers.LeakyReLU()
            elif meta["activation"] == "tanh":
                activation = tf.keras.activations.tanh
            else:
                raise ValueError(f'Unknown activation "{meta["activation"]}"')
        else:
            activation = meta["activation"]

        layer_constructor = meta.get("layer_constructor", tf.keras.layers.Dense)

        # Define the layers
        self.hidden_layers = [
            layer_constructor(
                self.meta["units"],
                activation=activation,
                kernel_initializer=self.meta["initializer"],
            )
            for i in range(self.meta["layers"])
        ]

        self.dense_out = layer_constructor(
            self.channels_out, kernel_initializer="zeros", bias_initializer="zeros"
        )

    def _network(self, x):
        """The used layers in this Subnetwork.
        Returns:
            layers (tf.keras.layers): Some stacked keras layers.
        """
        for layer in self.hidden_layers:
            x = layer(x)

        y = self.dense_out(x)
        return y
