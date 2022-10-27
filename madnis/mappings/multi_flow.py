from typing import List
import tensorflow as tf

from ..mappings.flow import Flow
from ..mappings.base import Mapping

class MultiFlow(Mapping):
    def __init__(
        self,
        flows: List[Flow],
        **kwargs
    ):
        super().__init__(flows[0].base_dist, **kwargs)
        self.flows = flows
        self.n_channels = len(flows)

    def _forward(self, x: tf.Tensor, condition: tf.Tensor):
        channels = tf.argmax(condition, axis=1, output_type=tf.dtypes.int32)
        xs = tf.dynamic_partition(x, channels, self.n_channels)
        idx = tf.dynamic_partition(tf.range(tf.shape(x)[0]), channels, self.n_channels)
        zs = []
        log_dets = []
        for xi, flow in zip(xs, self.flows):
            zi, log_det_i = flow._forward(xi)
            zs.append(zi)
            log_dets.append(log_det_i)
        z = tf.dynamic_stitch(idx, zs)
        log_det = tf.dynamic_stitch(idx, log_dets)
        return z, log_det

    def _inverse(self, z: tf.Tensor, condition: tf.Tensor):
        channels = tf.argmax(condition, axis=1, output_type=tf.dtypes.int32)
        zs = tf.dynamic_partition(z, channels, self.n_channels)
        idx = tf.dynamic_partition(tf.range(tf.shape(z)[0]), channels, self.n_channels)
        xs = []
        log_dets = []
        for zi, flow in zip(zs, self.flows):
            xi, log_det_i = flow._forward(zi)
            xs.append(xi)
            log_dets.append(log_det_i)
        x = tf.dynamic_stitch(idx, xs)
        log_det = tf.dynamic_stitch(idx, log_dets)
        return z, log_det

    def _log_det(self, x_or_z: tf.Tensor = None, inverse: bool = False):
        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            _, logdet = self._inverse(x_or_z, embedded_condition)
            return logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            _, logdet = self._forward(x_or_z, embedded_condition)
            return logdet

    def _sample(self, num_samples: int, condition: tf.Tensor):
        channels = tf.argmax(condition, axis=1, output_type=tf.dtypes.int32)
        xs = []
        idx = tf.dynamic_partition(tf.range(num_samples), channels, self.n_channels)
        xs = [flow._sample(tf.shape(idx_i)[0]) for idx_i, flow in zip(idx, self.flows)]
        return tf.dynamic_stitch(idx, xs)
