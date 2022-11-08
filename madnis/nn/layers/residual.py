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
        2. :math:`alpha_i = \Prod_{k in Prop} \frac{1}{|p_k^2 - M_k^2 -i M_k \Gamma_k^2|^2}`

        Then the network ``f`` simply yields a correction
        and gives the output:

        :math:`alpha_{i, new} = \theta_{i} * f_{i} + \log alpha_{i}`

        with a trainable weight :math:`\theta_i`.
        """
        out = self.w_x * x + tf.math.log(residual)  # log guarantees the residual is restored after normalization
        return out

class AdditiveResidualWeight(tf.keras.layers.Layer):
    """ Defines a layer that adds a residual """

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.scale = self.add_weight(
            "scale",
            shape=(1,),
            initializer=tf.keras.initializers.Ones(),
            #initializer=tf.keras.initializers.Constant(10.),
            trainable=True,
            )

    def call(self, x, residual):
        """ combines x and residual with a trainable relative weight beta """

        beta = tf.sigmoid(10.*self.scale)

        return beta*residual + (1. - beta) * x
