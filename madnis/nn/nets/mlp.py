""" MLP subnetwork """

import tensorflow as tf
from .base import SubNet


class MLP(SubNet):
    """
    Creates a dense subnetwork
    which can be used within the invertible modules.
    """

    def __init__(self, meta, channels_in, channels_out):
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

        # Define the layers
        self.hidden_layers = [
            tf.keras.layers.Dense(
                self.meta["units"],
                activation=activation,
                kernel_initializer=self.meta["initializer"],
            )
            for i in range(self.meta["layers"])
        ]

        self.dense_out = tf.keras.layers.Dense(
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
