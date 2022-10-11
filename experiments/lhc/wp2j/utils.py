from typing import Tuple
import tensorflow as tf
from typing import Dict

def translate_channels(ch_in: tf.Tensor):
    """Makes translation:
    {0,1,2,3} -> {0,2,3,6}
    """
    ch_in = tf.cast(ch_in, dtype=tf.float32)
    ch_out = 1/2 * ch_in**3 - 2 * ch_in**2 + 7/2 * ch_in
    return tf.cast(ch_out, dtype=tf.int32)