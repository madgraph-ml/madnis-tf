""" Implementation of the divergences. """

import tensorflow as tf
from functools import wraps


def wrapped_multi_channel(func):
    """Implement multi-channel decorator.

    This decorator wraps the function to implement multi-channel
    functionality by splitting into the different contributions and
    summing up the different channel contributions

    Arguments:
        func: function to be wrapped

    Returns:
        Callable: decoratated divergence
    """
    @wraps(func)
    def wrapped_divergence(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        channels: tf.Tensor,
        q_sample: tf.Tensor = None,
        sigma: tf.Tensor = None,
    ):
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        if self.train_mcw:
            q_test = tf.stop_gradient(q_test)
            logq = tf.stop_gradient(logq)
        else:
            p_true = tf.stop_gradient(p_true)
            logp = tf.stop_gradient(logp)
        q_sample = tf.stop_gradient(q_sample)

        logps = tf.dynamic_partition(logp, channels, self.n_channels)
        logqs = tf.dynamic_partition(logq, channels, self.n_channels)
        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qs = tf.dynamic_partition(q_test, channels, self.n_channels)
        q_samps = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.unstack(sigma)
        n_samples = tf.cast(tf.shape(q_sample)[0], self._dtype)

        loss = 0
        for logpi, logqi, pi, qti, qsi, sigi in zip(
            logps, logqs, ps, qs, q_samps, sigmas
        ):
            ni = tf.cast(tf.shape(qsi)[0], self._dtype)
            loss += sigi * func(self, pi, qti, logpi, logqi, qsi) * ni / n_samples

        return loss

    return wrapped_divergence


class Divergence:
    """Divergence class container.

    This class contains a list of f-divergences that
    can serve as loss functions. f-divergences
    are a mathematical measure for how close two statistical
    distributions are.

    All of the implemented divergences must be called with
    the same four arguments, even though some of them only
    use two of them.

    **Remarks:**:
    - All losses are set-up in such a way, that they can be used
      to either optimize the ``q_test`` distribution for importance sampling
      or the multi-channel weight (``train_mcw = True``)

    - It uses importance sampling explicitly, i.e. the estimator is divided
      by an additional factor of ``q_sample``.

    """

    def __init__(
        self,
        alpha: float = None,
        beta: float = None,
        train_mcw: bool = False,
        n_channels: int = 1,
    ):
        """
        Args:
            alpha (float, optional): needed for (alpha, beta)-product divergence
                and Chernoff divergence. Defaults to None.
            beta (float, optional): needed for (alpha, beta)-product divergence.
                Defaults to None.
            train_mcw (bool, optional): returns losses such that the multi-channel
                weight can be trained. Requires different handling of tf.stop_gradient.
                Defaults to False.
            n_channels (int, optional): the number of channels used for the integration.
                Defaults to 1.

        """
        self.alpha = alpha
        self.beta = beta
        self.train_mcw = train_mcw
        self.n_channels = n_channels
        self._dtype = tf.keras.backend.floatx()
        self.divergences = [
            x
            for x in dir(self)
            if (
                "__" not in x
                and "alpha" not in x
                and "beta" not in x
                and "train_mcw" not in x
                and "n_channels" not in x
                and "_dtype" not in x
            )
        ]

    @wrapped_multi_channel
    def variance(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement variance loss.

        This function returns the variance loss for two given sets
        of functions, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        **Remark:**
        In the variance loss the ``p_true`` function does not have to be normalized to 1.

        tf.stop_gradient is used such that the correct gradient is returned
        when the variance is used as loss function.

        Arguments:
            p_true (tf.tensor): true function/probability. Does not have to be normalized.
            q_test (tf.tensor): estimated function/probability
            logp (tf.tensor): not used in variance
            logq (tf.tensor): not used in variance
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).
            q_sample (tf.tensor): sampling probability
            sigma (tf.tensor): loss weights with shape (n_channels,). Defaults to None.

        Returns:
            tf.tensor: computed variance loss

        """
        mean2 = tf.reduce_mean(p_true**2 / (q_test * q_sample), axis=0)
        mean = tf.reduce_mean(p_true / q_sample, axis=0)
        return mean2 - mean**2

    @wrapped_multi_channel
    def neyman_chi2(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement Neyman chi2 divergence.

        This function returns the Neyman chi2 divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the chi2 is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probabiliy
            logp (tf.tensor): not used in chi2
            logq (tf.tensor): not used in chi2
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Neyman chi2 divergence

        """
        return tf.reduce_mean((p_true - q_test) ** 2 / (q_test * q_sample), axis=0)

    @wrapped_multi_channel
    def pearson_chi2(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement Pearson chi2 divergence.

        This function returns the Pearson chi2 divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the chi2 is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probabiliy
            logp (tf.tensor): not used in chi2
            logq (tf.tensor): not used in chi2
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Pearson chi2 divergence

        """
        return tf.reduce_mean((q_test - p_true) ** 2 / (p_true * q_sample), axis=0)

    @wrapped_multi_channel
    def kl_divergence(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement Kullback-Leibler (KL) divergence.

        This function returns the Kullback-Leibler divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the KL is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed KL divergence

        """
        return tf.reduce_mean(p_true / q_sample * (logp - logq), axis=0)

    @wrapped_multi_channel
    def reverse_kl(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement reverse Kullback-Leibler (RKL) divergence.

        This function returns the reverse Kullback-Leibler divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the RKL is used as loss function.

        Arguments:
            p_true (tf.tensor): not used in RKL
            q_test (tf.tensor): not used in RKL
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed RKL divergence

        """
        return tf.reduce_mean(q_test / q_sample * (logq - logp), axis=0)

    @wrapped_multi_channel
    def hellinger(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement Hellinger distance.

        This function returns the Hellinger distance for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Hellinger is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): not used in hellinger
            logq (tf.tensor): not used in hellinger
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Hellinger distance

        """
        return tf.reduce_mean(
            2.0 * (tf.math.sqrt(p_true) - tf.math.sqrt(q_test)) ** 2 / q_sample,
            axis=0,
        )

    @wrapped_multi_channel
    def jeffreys(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement Jeffreys divergence.

        This function returns the Jeffreys divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Jeffreys is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Jeffreys divergence

        """
        return tf.reduce_mean((p_true - q_test) * (logp - logq) / q_sample, axis=0)

    @wrapped_multi_channel
    def chernoff(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement Chernoff divergence.

        This function returns the Chernoff divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Chernoff is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): not used in Chernoff
            logq (tf.tensor): not used in Chernoff
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Chernoff divergence

        Raises:
           ValueError: If there is no alpha defined or alpha is not between 0 and 1

        """
        if self.alpha is None:
            raise ValueError("Must give an alpha value to use Chernoff " "Divergence.")
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1.")
        prefactor = 4.0 / (1 - self.alpha**2)

        return prefactor * (
            1
            - tf.reduce_mean(
                tf.pow(p_true, (1.0 - self.alpha) / 2.0)
                * tf.pow(q_test, (1.0 + self.alpha) / 2.0)
                / q_sample,
                axis=0,
            )
        )

    @wrapped_multi_channel
    def exponential(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement Exponential divergence.

        This function returns the Exponential divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Exponential is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Exponential divergence

        """
        return tf.reduce_mean(p_true / q_sample * (logp - logq) ** 2, axis=0)

    @wrapped_multi_channel
    def exponential2(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement Exponential divergence with ``p_true`` and ``q_test`` interchanged.

        This function returns the Exponential2 divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Exponential2 is used as loss function.

        Arguments:
            p_true (tf.tensor): not used in Exponential2
            q_test (tf.tensor): not used in Exponential2
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Exponential2 divergence

        """
        return tf.reduce_mean(q_test / q_sample * (logq - logp) ** 2, axis=0)

    @wrapped_multi_channel
    def ab_product(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement (alpha, beta)-product divergence.

        This function returns the (alpha, beta)-product divergence for two given
        sets of probabilities, ``p_true`` and ``q_test``. It uses importance sampling,
        i.e. the estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the ab_product is used as loss function.

        Arguments:
            p_true (tf.tensor):  true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): not used in ab_product
            logq (tf.tensor): not used in ab_product
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed (alpha, beta)-product divergence

        Raises:
           ValueError: If there is no alpha defined or alpha is not between 0 and 1
           ValueError: If there is no beta defined or beta is not between 0 and 1

        """
        if self.alpha is None:
            raise ValueError(
                "Must give an alpha value to use (alpha, beta)-product Divergence."
            )
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if self.beta is None:
            raise ValueError(
                "Must give an beta value to use (alpha, beta)-product Divergence."
            )
        if not 0 < self.beta < 1:
            raise ValueError("Beta must be between 0 and 1.")

        prefactor = 2.0 / ((1 - self.alpha) * (1 - self.beta))

        return prefactor * tf.reduce_mean(
            (1 - tf.pow(q_test / p_true, (1 - self.alpha) / 2.0))
            * (1 - tf.pow(q_test / p_true, (1 - self.beta) / 2.0))
            * p_true
            / q_sample,
            axis=0,
        )

    @wrapped_multi_channel
    def js_divergence(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        q_sample: tf.Tensor,
    ):
        """Implement Jensen-Shannon (JS) divergence.

        This function returns the Jensen-Shannon divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_sample``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Jensen-Shannon is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nchannels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Jensen-Shannon divergence

        """
        logm = tf.math.log(0.5 * (q_test + p_true))
        return tf.reduce_mean(
            0.5 / q_sample * (p_true * (logp - logm) + q_test * (logq - logm)), axis=0
        )

    def __call__(self, name):
        func = getattr(self, name, None)
        if func is not None:
            return func
        raise NotImplementedError(
            f"The requested loss function {name} is not implemented. "
            + f"Allowed options are {self.divergences}."
        )
