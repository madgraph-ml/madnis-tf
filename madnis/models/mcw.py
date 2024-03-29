import tensorflow as tf

from ..nn.nets.mlp import MLP, MLPZero
from ..nn.layers.residual import ResidualWeight, AdditiveResidualWeight
from ..nn.layers.normalize import NormalizeWeights


def mcw_model(
    dims_in: int,
    n_channels: int,
    meta: dict,
    name: str = "MCW",
    **kwargs,
):
    """Defines a network to fit multi channel weights"""

    x_input = tf.keras.Input((dims_in,))
    x = MLPZero(meta, dims_in, n_channels)(x_input)
    x_out = NormalizeWeights(use_probs=True)(x)

    return tf.keras.Model(inputs=[x_input], outputs=x_out, name=name)


def residual_mcw_model(
    dims_in: int,
    n_channels: int,
    meta: dict,
    name: str = "ResidualMCW",
    **kwargs,
):
    """Defines a network to fit multi channel weights
    with residual inputs of some prior assumptions to make
    the training easier.
    """

    x_input = tf.keras.Input((dims_in,))
    residual = tf.keras.Input((n_channels,))
    x = MLP(meta, dims_in, n_channels)(x_input)
    x = ResidualWeight(n_channels)(x, residual)
    x_out = NormalizeWeights(use_probs=True)(x)

    return tf.keras.Model(inputs=[x_input, residual], outputs=x_out, name=name)

def additive_residual_mcw_model(
        dims_in: int,
        n_channels: int,
        meta: dict,
        name: str = "AdditiveResidualMCW",
        **kwargs,
        ):
    """Defines a network to fit multi channel weights
    with residual inputs of some prior assumptions to make
    the training easier.
    """

    x_input = tf.keras.Input((dims_in,))
    residual = tf.keras.Input((n_channels,))
    x = MLP(meta, dims_in, n_channels)(x_input)
    x = NormalizeWeights(use_probs=True)(x)
    x_out = AdditiveResidualWeight()(x, residual)

    return tf.keras.Model(inputs=[x_input, residual], outputs=x_out, name=name)
