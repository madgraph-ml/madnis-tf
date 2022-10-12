import tensorflow as tf

class VBLinear(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        prior_width=1.0,
        logsig2_init=-9.,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        self.prior_width = tf.constant(prior_width, dtype=self.dtype)
        self.logsig2_init = tf.constant(logsig2_init, dtype=self.dtype)
        self.map = False
        self.training = True

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        self.bias = self.add_weight(
            "bias",
            shape=(self.units, ),
            initializer=self.bias_initializer,
            dtype=dtype,
            trainable=True
        )
        self.mu_w = self.add_weight(
            "mu_w",
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            dtype=dtype,
            trainable=True
        )
        self.logsig2_w = self.add_weight(
            "logsig2_w",
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.RandomNormal(
                mean=self.logsig2_init, stddev=0.001),
            dtype=dtype,
            trainable=True
        )
        self.random = self.add_weight(
            "random",
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0., stddev=1.),
            dtype=dtype,
            trainable=False
        )
        self.built = True

    def sample_weights(self):
        self.random.assign(tf.random.normal(tf.shape(self.random), dtype=self.dtype))

    def kl(self):
        logsig2_w = tf.clip_by_value(self.logsig2_w, -11, 11)
        return 0.5 * tf.math.reduce_sum(
            self.prior_width * (self.mu_w**2 + tf.math.exp(logsig2_w))
            - logsig2_w - 1 - tf.math.log(self.prior_width)
        )

    def call(self, inputs):
        if self.training:
            mu_out = tf.matmul(inputs, self.mu_w) + self.bias
            logsig2_w = tf.clip_by_value(self.logsig2_w, -11, 11)
            s2_w = tf.math.exp(logsig2_w)
            var_out = tf.matmul(inputs**2, s2_w) + 1e-8
            return mu_out + tf.math.sqrt(var_out) * tf.random.normal(
                tf.shape(mu_out), dtype=self.dtype)

        else:
            if self.map:
                return tf.matmul(inputs, self.mu_w) + self.bias

            logsig2_w = tf.clip_by_value(self.logsig2_w, -11, 11)
            s2_w = tf.math.exp(logsig2_w)
            weight = self.mu_w + tf.math.sqrt(s2_w) * self.random
            return tf.matmul(inputs, weight) + self.bias + 1e-8


class BayesianHelper:
    def __init__(
        self,
        dataset_size: int,
        layer_kwargs: dict = {}
    ):
        self.dataset_size = dataset_size
        self.layer_kwargs = layer_kwargs
        self.layers = []

    def construct_dense(self, units, **kwargs):
        layer = VBLinear(units, **self.layer_kwargs, **kwargs)
        self.layers.append(layer)
        return layer

    def kl_loss(self):
        s = sum(layer.kl() for layer in self.layers) / self.dataset_size
        return s

    def sample_weights(self):
        for layer in self.layers:
            layer.sample_weights()

    def set_map(self, map_enabled):
        for layer in self.layers:
            layer.map = map_enabled

    def set_training(self, training):
        for layer in self.layers:
            layer.training = training
