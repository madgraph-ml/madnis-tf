""" Coupling Blocks """

from typing import Dict, Callable, Union
import tensorflow as tf

from ..base import Transform


# pylint: disable=C0103, R1729, E1120, E1124, W0221
class CouplingTransform(Transform):
    """Base class to implement various coupling schemes.  It takes care of
    checking the dimensions, conditions, clamping mechanism, etc.
    Each child class only has to implement the _coupling1 and _coupling2 methods
    for the left and right coupling operations.
    (In some cases, call() is also overridden)
    """

    def __init__(
        self,
        dims_in,
        dims_c=None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "ATAN",
    ):
        """
        Additional args in docstring of base class.
        Args:
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        """
        super().__init__(dims_in, dims_c)

        # Note: The shapes are in the standard
        # tensorflow 'channels_last' format which
        # means that shape = (dim_1,..., dim_channels)
        self.channels = self.dims_in[-1]

        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(self.dims_in) - 1

        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        self.split_len1 = self.channels // 2
        self.split_len2 = self.channels - self.channels // 2

        self.clamp = clamp

        assert all(
            [
                tuple(self.dims_c[i][:-1]) == tuple(self.dims_in[:-1])
                for i in range(len(self.dims_c))
            ]
        ), "Dimensions of input and one or more conditions don't agree."
        self.conditional = len(self.dims_c) > 0
        self.condition_length = sum(
            [self.dims_c[i][-1] for i in range(len(self.dims_c))]
        )

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = lambda u: 0.636 * tf.math.atan(u)
            elif clamp_activation == "TANH":
                self.f_clamp = tf.math.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = lambda u: 2.0 * (tf.sigmoid(u) - 0.5)
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def call(self, x, c=None, jac=True):
        """
        Perform a forward pass
        through this layer operator.
        Args:
            x:      input data (array-like of one or more tensors)
                    of the form: ``x = [input_tensor_1]``
            c:      conditioning data (array-like of none or more tensors)
                    of the form: ``x = [cond_tensor_1, cond_tensor_2, ...] ``
            rev:    perform backward pass
            jac:    return Jacobian associated to the direction
        """

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations

        x1, x2 = tf.split(x, [self.split_len1, self.split_len2], axis=-1)

        x2_c = tf.concat([x2, *c], -1) if self.conditional else x2
        y1, j1 = self._coupling1(x1, x2_c)

        y1_c = tf.concat([y1, *c], -1) if self.conditional else y1
        y2, j2 = self._coupling2(x2, y1_c)

        if jac:
            return tf.concat([y1, y2], -1), j1 + j2

        return tf.concat([y1, y2], -1)

    def inverse(self, z, c=None, jac=True):
        """
        Perform a backward pass
        through this layer operator.
        Args:
            z:      input data (array-like of one or more tensors)
                    of the form: ``x = [input_tensor_1]``
            c:      conditioning data (array-like of none or more tensors)
                    of the form: ``x = [cond_tensor_1, cond_tensor_2, ...] ``
            jac:    return Jacobian associated to the direction
        """

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations

        z1, z2 = tf.split(z, [self.split_len1, self.split_len2], axis=-1)

        z1_c = tf.concat([z1, *c], -1) if self.conditional else z1
        y2, j2 = self._coupling2(z2, z1_c, rev=True)

        y2_c = tf.concat([y2, *c], -1) if self.conditional else y2
        y1, j1 = self._coupling1(z1, y2_c, rev=True)

        if jac:
            return tf.concat([y1, y2], -1), j1 + j2

        return tf.concat([y1, y2], -1)

    def _coupling1(self, x1, u2, rev=False):
        """The first/left coupling operation in a two-sided coupling block.
        Args:
          x1 (Tensor): the 'active' half being transformed.
          u2 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y1 (Tensor): same shape as x1, the transformed 'active' half.
          j1 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        """
        raise NotImplementedError()

    def _coupling2(self, x2, u1, rev=False):
        """The second/right coupling operation in a two-sided coupling block.
        Args:
          x2 (Tensor): the 'active' half being transformed.
          u1 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y2 (Tensor): same shape as x1, the transformed 'active' half.
          j2 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        """
        raise NotImplementedError()

    def get_config(self):
        config = {"clamp": self.clamp, "clamp_activation": self.f_clamp}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
