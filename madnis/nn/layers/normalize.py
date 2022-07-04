""" Normalization Layer """

import tensorflow as tf


class NormalizeWeights(tf.keras.layers.Layer):
    """Defines a layer that normalizes the output"""

    def __init__(self, use_probs: bool = True, **kwargs):
        r"""
        Args:
            use_probs: bool, if True then :math:`0 < alpha \le 1`, otherwise
                the individual channel weights are unbounded.
        """
        super().__init__(**kwargs)

        # coupling to fixed weight
        self.use_probs = use_probs

    def call(self, x):
        r"""
        Normalizes the input such that:

        1. :math:`alpha_i \in (0,1)`
        2. :math:`\Sum_i alpha_i = 1`

        Using softmax this guarantees this
        to be true and stabilizes the training while
        having easy and tractable gradients.
        """

        if self.use_probs:
            return tf.keras.activations.softmax(x, axis=-1)
        else:
            norm = tf.expand_dims(tf.reduce_sum(x, axis=-1), -1)
            return x / norm
