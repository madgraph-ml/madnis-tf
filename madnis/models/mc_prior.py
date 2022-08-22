""" Implement multi-channel weight prior. """

import tensorflow as tf
from typing import List, Union, Callable
from ..distributions.base import Distribution
import sys


class WeightPrior:
    """Class that returns the multi-channel weight pripr
    """

    def __init__(
        self,
        func: List[Distribution],
        n_channels: int,
        **kwargs,
    ):
        """
        Args:
            func (Union[Callable, Distribution]):
                Function to be integrated
            n_channels (int): number of channel of integrations
        """
        self._dtype = tf.keras.backend.floatx()
        self.func = func
        self.n_channels = n_channels
        assert len(self.func) == self.n_channels
    
    def get_prior_weights(self, inputs: tf.Tensor):
        gs = []
        g_tot = 0
        for i in range(self.n_channels):
            gi = self.func[i].prob(inputs)
            g_tot += gi[..., None]
            gs.append(gi[..., None])

        prior_list = [g / g_tot for g in gs]
        prior_weights = tf.concat(prior_list, axis=-1)
        return prior_weights