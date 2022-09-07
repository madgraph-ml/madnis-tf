""" Coupling Blocks """

from typing import Dict, Callable, Union
import tensorflow as tf

from .base import CouplingTransform


class NICECouplingBlock(CouplingTransform):
    """Coupling Block following the NICE (Dinh et al, 2015) design.
    The inputs are split in two halves. For 2D, 3D, 4D inputs, the split is
    performed along the channel (first) dimension. Then, residual coefficients are
    predicted by two subnetworks that are added to each half in turn.
    """

    def __init__(
        self,
        dims_in,
        dims_c=None,
        subnet_meta: Dict = None,
        subnet_constructor: Callable = None,
    ):
        """
        Additional args in docstring of base class.
        Args:
          subnet_meta:
            meta defining the structure of the subnet like number of layers, units,
            activation functions etc.
          subnet_constructor:
            Class or Callable ``f``, called as ``f(meta, channels_in, channels_out)`` and
            should return a tf.keras.layer.Layer.
            Two of these subnetworks will be initialized inside the block.
        """
        super().__init__(dims_in, dims_c, clamp=0.0, clamp_activation=(lambda u: u))

        self.F = subnet_constructor(
            subnet_meta, self.split_len2 + self.condition_length, self.split_len1
        )
        self.G = subnet_constructor(
            subnet_meta, self.split_len1 + self.condition_length, self.split_len2
        )

    def _coupling1(self, x1, u2, rev=False):
        if rev:
            return x1 - self.F(u2), 0.0
        return x1 + self.F(u2), 0.0

    def _coupling2(self, x2, u1, rev=False):
        if rev:
            return x2 - self.G(u1), 0.0
        return x2 + self.G(u1), 0.0


class RNVPCouplingBlock(CouplingTransform):
    """Coupling Block following the RealNVP design (Dinh et al, 2017) with some
    minor differences. The inputs are split in two halves. For 2D, 3D, 4D
    inputs, the split is performed along the channel dimension. Two affine
    coupling operations are performed in turn on both halves of the input.
    """

    def __init__(
        self,
        dims_in,
        dims_c=None,
        subnet_meta: Dict = None,
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):
        """
        Additional args in docstring of base class.
        Args:
          subnet_meta:
            meta defining the structure of the subnet like number of layers, units,
            activation functions etc.
          subnet_constructor:
            Class or Callable ``f``, called as ``f(meta, channels_in, channels_out)`` and
            should return a tf.keras.layer.Layer. Four of these subnetworks will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        """

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        self.subnet_s1 = subnet_constructor(
            subnet_meta, self.split_len1 + self.condition_length, self.split_len2
        )
        self.subnet_t1 = subnet_constructor(
            subnet_meta, self.split_len1 + self.condition_length, self.split_len2
        )
        self.subnet_s2 = subnet_constructor(
            subnet_meta, self.split_len2 + self.condition_length, self.split_len1
        )
        self.subnet_t2 = subnet_constructor(
            subnet_meta, self.split_len2 + self.condition_length, self.split_len1
        )

    def _coupling1(self, x1, u2, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        s2, t2 = self.subnet_s2(u2), self.subnet_t2(u2)
        s2 = self.clamp * self.f_clamp(s2)
        j1 = tf.reduce_sum(s2, self.sum_dims)

        if rev:
            y1 = (x1 - t2) * tf.math.exp(-s2)
            return y1, -j1

        y1 = tf.math.exp(s2) * x1 + t2
        return y1, j1

    def _coupling2(self, x2, u1, rev=False):
        s1, t1 = self.subnet_s1(u1), self.subnet_t1(u1)
        s1 = self.clamp * self.f_clamp(s1)
        j2 = tf.reduce_sum(s1, axis=self.sum_dims)

        if rev:
            y2 = (x2 - t1) * tf.math.exp(-s1)
            return y2, -j2

        y2 = tf.math.exp(s1) * x2 + t1
        return y2, j2


class GLOWCouplingBlock(CouplingTransform):
    """Coupling Block following the GLOW design. Note, this is only the coupling
    part itself, and does not include ActNorm, invertible 1x1 convolutions, etc.
    See AllInOneBlock for a block combining these functions at once.
    The only difference to the RNVPCouplingBlock coupling blocks
    is that it uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate
    subnetworks. This reduces computational cost and speeds up learning.
    """

    def __init__(
        self,
        dims_in,
        dims_c=None,
        subnet_meta: Dict = None,
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):
        """
        Additional args in docstring of base class.
        Args:
          subnet_meta:
            meta defining the structure of the subnet like number of layers, units,
            activation functions etc.
          subnet_constructor:
            Class or Callable ``f``, called as ``f(meta, channels_in, channels_out)`` and
            should return a tf.keras.layer.Layer. Four of these subnetworks will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        """

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        self.subnet1 = subnet_constructor(
            subnet_meta, self.split_len1 + self.condition_length, self.split_len2 * 2
        )
        self.subnet2 = subnet_constructor(
            subnet_meta, self.split_len2 + self.condition_length, self.split_len1 * 2
        )

    def _coupling1(self, x1, u2, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a2 = self.subnet2(u2)
        s2, t2 = tf.split(a2, [self.split_len1, self.split_len1], axis=-1)
        s2 = self.clamp * self.f_clamp(s2)
        j1 = tf.reduce_sum(s2, axis=self.sum_dims)

        if rev:
            y1 = (x1 - t2) * tf.math.exp(-s2)
            return y1, -j1

        y1 = tf.math.exp(s2) * x1 + t2
        return y1, j1

    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = tf.split(a1, [self.split_len2, self.split_len2], axis=-1)
        s1 = self.clamp * self.f_clamp(s1)
        j2 = tf.reduce_sum(s1, axis=self.sum_dims)

        if rev:
            y2 = (x2 - t1) * tf.math.exp(-s1)
            return y2, -j2

        y2 = tf.math.exp(s1) * x2 + t1
        return y2, j2


class GINCouplingBlock(CouplingTransform):
    """Coupling Block following the GIN design. The difference from
    GLOWCouplingBlock (and other affine coupling blocks) is that the Jacobian
    determinant is constrained to be 1.  This constrains the block to be
    volume-preserving. Volume preservation is achieved by subtracting the mean
    of the output of the s subnetwork from itself.  While volume preserving, GIN
    is still more powerful than NICE, as GIN is not volume preserving within
    each dimension.
    Note: this implementation differs slightly from the originally published
    implementation, which scales the final component of the s subnetwork so the
    sum of the outputs of s is zero. There was no difference found between the
    implementations in practice, but subtracting the mean guarantees that all
    outputs of s are at most ±exp(clamp), which might be more stable in certain
    cases.
    """

    def __init__(
        self,
        dims_in,
        dims_c=None,
        subnet_meta: Dict = None,
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):
        """
        Additional args in docstring of base class.
        Args:
          subnet_meta:
            meta defining the structure of the subnet like number of layers, units,
            activation functions etc.
          subnet_constructor:
            Class or Callable ``f``, called as ``f(meta, channels_in, channels_out)`` and
            should return a tf.keras.layer.Layer. Two of these subnetworks will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        """

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        self.subnet1 = subnet_constructor(
            subnet_meta, self.split_len1 + self.condition_length, self.split_len2 * 2
        )
        self.subnet2 = subnet_constructor(
            subnet_meta, self.split_len2 + self.condition_length, self.split_len1 * 2
        )

    def _coupling1(self, x1, u2, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a2 = self.subnet2(u2)
        s2, t2 = tf.split(a2, [self.split_len1, self.split_len1], axis=-1)
        s2 = self.clamp * self.f_clamp(s2)
        s2 -= tf.reduce_mean(s2, axis=-1, keepdims=True)

        if rev:
            y1 = (x1 - t2) * tf.math.exp(-s2)
            return y1, 0.0

        y1 = tf.math.exp(s2) * x1 + t2
        return y1, 0.0

    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = tf.split(a1, [self.split_len2, self.split_len2], axis=-1)
        s1 = self.clamp * self.f_clamp(s1)
        s1 -= tf.reduce_mean(s1, axis=-1, keepdims=True)

        if rev:
            y2 = (x2 - t1) * tf.math.exp(-s1)
            return y2, 0.0

        y2 = tf.math.exp(s1) * x2 + t1
        return y2, 0.0


class AffineCoupling(CouplingTransform):
    """Half of a coupling block following the GLOWCouplingBlock design.  This
    means only one affine transformation on half the inputs. In the case where
    random permutations or orthogonal transforms are used after every block,
    this is not a restriction and simplifies the design.
    """

    def __init__(
        self,
        dims_in,
        dims_c=None,
        subnet_meta: Dict = None,
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):
        """
        Additional args in docstring of base class.
        Args:
          subnet_meta:
            meta defining the structure of the subnet like number of layers, units,
            activation functions etc.
          subnet_constructor:
            Class or Callable ``f``, called as ``f(meta, channels_in, channels_out)`` and
            should return a tf.keras.layer.Layer. One subnetwork will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        """

        super().__init__(dims_in, dims_c, clamp, clamp_activation)
        self.subnet = subnet_constructor(
            subnet_meta, self.split_len1 + self.condition_length, 2 * self.split_len2
        )

    def _coupling1(self, x1, u2, rev=False):
        pass

    def _coupling2(self, x2, u1, rev=False):
        pass

    def call(self, x, c=None, jac=True):
        x1, x2 = tf.split(x, [self.split_len1, self.split_len2], axis=-1)
        x1_c = tf.concat([x1, *c], -1) if self.conditional else x1

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a = self.subnet(x1_c)
        s, t = tf.split(a, [self.split_len2, self.split_len2], axis=-1)
        s = self.clamp * self.f_clamp(s)
        j = tf.reduce_sum(s, axis=self.sum_dims)
        y2 = x2 * tf.math.exp(s) + t

        return tf.concat([x1, y2], -1), j

    def inverse(self, x, c=None, jac=True):
        x1, x2 = tf.split(x, [self.split_len1, self.split_len2], axis=-1)
        x1_c = tf.concat([x1, *c], -1) if self.conditional else x1

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a = self.subnet(x1_c)
        s, t = tf.split(a, [self.split_len2, self.split_len2], axis=-1)
        s = self.clamp * self.f_clamp(s)
        j = tf.reduce_sum(s, axis=self.sum_dims)

        # if rev
        y2 = (x2 - t) * tf.math.exp(-s)

        return tf.concat([x1, y2], -1), -j


class ConditionalAffineTransform(CouplingTransform):
    """Similar to the conditioning layers from SPADE (Park et al, 2019): Perform
    an affine transformation on the whole input, where the affine coefficients
    are predicted from only the condition.
    """

    def __init__(
        self,
        dims_in,
        dims_c=None,
        subnet_meta: Dict = None,
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "TANH",
    ):
        """
        Additional args in docstring of base class.
        Args:
          subnet_meta:
            meta defining the structure of the subnet like number of layers, units,
            activation functions etc.
          subnet_constructor:
            Class or Callable ``f``, called as ``f(meta, channels_in, channels_out)`` and
            should return a tf.keras.layer.Layer. One subnetwork will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        """

        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        if not self.conditional:
            raise ValueError("ConditionalAffineTransform must have a condition")

        self.subnet = subnet_constructor(
            subnet_meta, self.condition_length, 2 * self.channels
        )

    def _coupling1(self, x1, u2, rev=False):
        pass

    def _coupling2(self, x2, u1, rev=False):
        pass

    def call(self, x, c=None, jac=True):
        if len(c) > 1:
            cond = tf.concat(c, -1)
        else:
            cond = c[0]

        # notation:
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a = self.subnet(cond)
        s, t = tf.split(a, [self.channels, self.channels], axis=-1)
        s = self.clamp * self.f_clamp(s)
        j = tf.reduce_sum(s, axis=self.sum_dims)
        y = tf.math.exp(s) * x + t

        if jac:
            return y, j

        return y

    def inverse(self, x, c=None, jac=True):
        if len(c) > 1:
            cond = tf.concat(c, -1)
        else:
            cond = c[0]

        # notation:
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a = self.subnet(cond)
        s, t = tf.split(a, [self.channels, self.channels], axis=-1)
        s = self.clamp * self.f_clamp(s)
        j = tf.reduce_sum(s, axis=self.sum_dims)
        # if rev
        y = (x - t) * tf.math.exp(-s)

        if jac:
            return y, -j

        return y
