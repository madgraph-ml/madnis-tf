""" Residual Layer """

import tensorflow as tf


class ResidualWeight(tf.keras.layers.Layer):
    """Defines a layer that builds a residual weight"""

    def __init__(
        self,
        channels_out: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # # coupling to trainable weight initialized with ones
        # self.w_res = self.add_weight(
        #     "w_res",
        #     shape=(channels_out,),
        #     initializer=tf.keras.initializers.Ones(),
        #     trainable=True,
        # )

        self.w_x = self.add_weight(
            "w_x",
            shape=(channels_out,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, x, residual):
        r"""To simplify the task for the NN,
        the network only learns a correction to a
        physics-based channel-weight:

        1. :math:`alpha_i = \frac{|M_i|^2}{\Sum_j |M_j|^2}`
        2. :math:`alpha_i = \Prod_{k in Prop} \frac{1}{|p_k^2 - M_k^2 -i M_k \Gamma_k^2|^2}

        Then the network ``f`` simply yields a correction
        and gives the output:

        :math:`alpha_{i, new} = theta_{i} * f_{i} + w_{i} * alpha_{i}

        with a trainable weight w_i.
        """
        # out = self.w_x * x + self.w_res * residual
        out = self.w_x * x + tf.math.log(residual)  # log guarantees the residual is restored after normalization
        return out
