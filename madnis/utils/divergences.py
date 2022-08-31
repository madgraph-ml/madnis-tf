""" Implementation of the divergences. """

import tensorflow as tf


class Divergence:
    """Divergence class conatiner.

    This class contains a list of f-divergences that
    can serve as loss functions. f-divergences
    are a mathematical measure for how close two statistical
    distributions are.

    All of the implemented divergences must be called with
    the same four arguments, even though some of them only
    use two of them.

    **Remarks:**:
    - All losses are set-up in such a way, that they can be used
      to optimize the ``q_test`` distribution for importance sampling.

    - It uses importance sampling explicitly, i.e. the estimator is divided
      by an additional factor of ``q_test``.

    """

    def __init__(
        self,
        alpha: float = None,
        beta: float = None,
        train_mcw: bool = False,
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

        """
        self.alpha = alpha
        self.beta = beta
        self.train_mcw = train_mcw
        self._dtype = tf.keras.backend.floatx()
        self.divergences = [
            x
            for x in dir(self)
            if (
                "__" not in x
                and "alpha" not in x
                and "beta" not in x
                and "train_mcw" not in x
                and "_dtype" not in x
            )
        ]

    def variance2(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement variance loss.

        This function returns the variance loss for two given sets
        of functions, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        **Remark:**
        In the variance loss the ``p_true`` function does not have to be normalized to 1.

        tf.stop_gradient is used such that the correct gradient is returned
        when the variance is used as loss function.

        Arguments:
            p_true (tf.tensor): true function/probability. Does not have to be normalized.
            q_test (tf.tensor): estimated function/probability
            logp (tf.tensor): not used in variance
            logq (tf.tensor): not used in variance
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed variance loss

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        ps = tf.dynamic_partition(p_true, channels, 2)
        qt = tf.dynamic_partition(q_test, channels, 2)
        qs = tf.dynamic_partition(q_sample, channels, 2)
        sigmas = tf.dynamic_partition(sigma, channels, 2)
        var = 0
        if self.train_mcw:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                mean2 = tf.reduce_mean(
                    sigi
                    * pi ** 2
                    / (tf.stop_gradient(qti) * tf.stop_gradient(qsi)),
                    axis=0,
                )
                mean = tf.reduce_mean(sigi * pi / tf.stop_gradient(qsi), axis=0)
                var += (mean2 - mean ** 2)
        else:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                mean2 += tf.reduce_mean(
                    sigi
                    * tf.stop_gradient(pi) ** 2
                    / (qti * tf.stop_gradient(qsi)),
                    axis=0,
                )
                mean += tf.reduce_mean(sigi * tf.stop_gradient(pi / qsi), axis=0)
                var += mean2 - mean ** 2

        return var

    # TODO: Check if this all actually correct with q_sample and q_test
    # Fine for now as long as q_sample = q_test.
    def variance(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement variance loss.

        This function returns the variance loss for two given sets
        of functions, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        **Remark:**
        In the variance loss the ``p_true`` function does not have to be normalized to 1.

        tf.stop_gradient is used such that the correct gradient is returned
        when the variance is used as loss function.

        Arguments:
            p_true (tf.tensor): true function/probability. Does not have to be normalized.
            q_test (tf.tensor): estimated function/probability
            logp (tf.tensor): not used in variance
            logq (tf.tensor): not used in variance
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed variance loss

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.train_mcw:
            mean2 = tf.reduce_mean(
                sigma
                * p_true ** 2
                / (tf.stop_gradient(q_test) * tf.stop_gradient(q_sample)),
                axis=0,
            )
            mean = tf.reduce_mean(sigma * p_true / tf.stop_gradient(q_sample), axis=0)
        else:
            mean2 = tf.reduce_mean(
                sigma
                * tf.stop_gradient(p_true) ** 2
                / (q_test * tf.stop_gradient(q_sample)),
                axis=0,
            )
            mean = tf.reduce_mean(sigma * tf.stop_gradient(p_true / q_sample), axis=0)

        return mean2 - mean ** 2

    def neyman_chi2(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement Neyman chi2 divergence.

        This function returns the Neyman chi2 divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the chi2 is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probabiliy
            logp (tf.tensor): not used in chi2
            logq (tf.tensor): not used in chi2
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed Neyman chi2 divergence

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.train_mcw:
            return tf.reduce_mean(
                sigma
                * (p_true - tf.stop_gradient(q_test)) ** 2
                / (tf.stop_gradient(q_test) * tf.stop_gradient(q_sample)),
                axis=0,
            )
        else:
            return tf.reduce_mean(
                sigma
                * (tf.stop_gradient(p_true) - q_test) ** 2
                / (q_test * tf.stop_gradient(q_sample)),
                axis=0,
            )

    def pearson_chi2(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement Pearson chi2 divergence.

        This function returns the Pearson chi2 divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the chi2 is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probabiliy
            logp (tf.tensor): not used in chi2
            logq (tf.tensor): not used in chi2
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed Pearson chi2 divergence

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.train_mcw:
            return tf.reduce_mean(
                sigma
                * (tf.stop_gradient(q_test) - p_true) ** 2
                / (p_true * tf.stop_gradient(q_sample)),
                axis=0,
            )
        else:
            return tf.reduce_mean(
                sigma
                * (q_test - tf.stop_gradient(p_true)) ** 2
                / (tf.stop_gradient(p_true) * tf.stop_gradient(q_sample)),
                axis=0,
            )

    def kl_divergence(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement Kullback-Leibler (KL) divergence.

        This function returns the Kullback-Leibler divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the KL is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed KL divergence

        """
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.train_mcw:
            return tf.reduce_mean(
                sigma
                * p_true
                / tf.stop_gradient(q_sample)
                * (logp - tf.stop_gradient(logq)),
                axis=0,
            )
        else:
            return tf.reduce_mean(
                sigma
                * tf.stop_gradient(p_true)
                / tf.stop_gradient(q_sample)
                * (tf.stop_gradient(logp) - logq),
                axis=0,
            )

    def reverse_kl(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement reverse Kullback-Leibler (RKL) divergence.

        This function returns the reverse Kullback-Leibler divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the RKL is used as loss function.

        Arguments:
            p_true (tf.tensor): not used in RKL
            q_test (tf.tensor): not used in RKL
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed RKL divergence

        """
        del p_true, q_test
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        sample_factor = tf.stop_gradient(q_test) / tf.stop_gradient(q_sample)
        if self.train_mcw:
            return tf.reduce_mean(
                sigma * sample_factor * (tf.stop_gradient(logq) - logp), axis=0
            )
        else:
            return tf.reduce_mean(
                sigma * sample_factor * (1 + tf.stop_gradient(logq - logp)) * logq,
                axis=0,
            )

    def hellinger(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement Hellinger distance.

        This function returns the Hellinger distance for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Hellinger is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): not used in hellinger
            logq (tf.tensor): not used in hellinger
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed Hellinger distance

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.train_mcw:
            return tf.reduce_mean(
                sigma
                * 2.0
                * (tf.math.sqrt(p_true) - tf.stop_gradient(tf.math.sqrt(q_test))) ** 2
                / tf.stop_gradient(q_sample),
                axis=0,
            )
        else:
            return tf.reduce_mean(
                sigma
                * 2.0
                * (tf.stop_gradient(tf.math.sqrt(p_true)) - tf.math.sqrt(q_test)) ** 2
                / tf.stop_gradient(q_sample),
                axis=0,
            )

    def jeffreys(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement Jeffreys divergence.

        This function returns the Jeffreys divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Jeffreys is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed Jeffreys divergence

        """
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.train_mcw:
            return tf.reduce_mean(
                sigma
                * (p_true - tf.stop_gradient(q_test))
                * (logp - tf.stop_gradient(logq))
                / tf.stop_gradient(q_sample),
                axis=0,
            )
        else:
            return tf.reduce_mean(
                sigma
                * (tf.stop_gradient(p_true) - q_test)
                * (tf.stop_gradient(logp) - logq)
                / tf.stop_gradient(q_sample),
                axis=0,
            )

    def chernoff(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement Chernoff divergence.

        This function returns the Chernoff divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Chernoff is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): not used in Chernoff
            logq (tf.tensor): not used in Chernoff
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed Chernoff divergence

        Raises:
           ValueError: If there is no alpha defined or alpha is not between 0 and 1

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.alpha is None:
            raise ValueError("Must give an alpha value to use Chernoff " "Divergence.")
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1.")

        prefactor = 4.0 / (1 - self.alpha ** 2)
        if self.train_mcw:
            return tf.reduce_mean(
                sigma
                * tf.pow(p_true, (1.0 - self.alpha) / 2.0)
                * tf.stop_gradient(tf.pow(q_test, (1.0 + self.alpha) / 2.0))
                / tf.stop_gradient(q_sample),
                axis=0,
            )
        else:
            return tf.reduce_mean(
                sigma
                * tf.stop_gradient(tf.pow(p_true, (1.0 - self.alpha) / 2.0))
                * tf.pow(q_test, (1.0 + self.alpha) / 2.0)
                / tf.stop_gradient(q_sample),
                axis=0,
            )

    def exponential(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement Exponential divergence.

        This function returns the Exponential divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Exponential is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed Exponential divergence

        """
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.train_mcw:
            return tf.reduce_mean(
                sigma
                * p_true
                / tf.stop_gradient(q_sample)
                * (logp - tf.stop_gradient(logq)) ** 2,
                axis=0,
            )
        else:
            return tf.reduce_mean(
                sigma
                * tf.stop_gradient(p_true)
                / tf.stop_gradient(q_sample)
                * (tf.stop_gradient(logp) - logq) ** 2,
                axis=0,
            )

    def exponential2(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement Exponential divergence with ``p_true`` and ``q_test`` interchanged.

        This function returns the Exponential2 divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Exponential2 is used as loss function.

        Arguments:
            p_true (tf.tensor): not used in Exponential2
            q_test (tf.tensor): not used in Exponential2
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed Exponential2 divergence

        """
        del p_true, q_test
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        sample_factor = tf.stop_gradient(q_test) / tf.stop_gradient(q_sample)
        if self.train_mcw:
            return tf.reduce_mean(
                sigma * sample_factor * (tf.stop_gradient(logq) - logp) ** 2, axis=0
            )
        else:
            return tf.reduce_mean(
                sigma
                * sample_factor
                * (
                    2 * tf.stop_gradient(logq - logp) * logq
                    + tf.stop_gradient(logq - logp) ** 2 * logq
                ),
                axis=0,
            )

    def ab_product(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
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
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed (alpha, beta)-product divergence

        Raises:
           ValueError: If there is no alpha defined or alpha is not between 0 and 1
           ValueError: If there is no beta defined or beta is not between 0 and 1

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.alpha is None:
            raise ValueError(
                "Must give an alpha value to use " "(alpha, beta)-product Divergence."
            )
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1.")

        if self.beta is None:
            raise ValueError(
                "Must give an beta value to use " "(alpha, beta)-product Divergence."
            )
        if not 0 < self.beta < 1:
            raise ValueError("Beta must be between 0 and 1.")

        prefactor = 2.0 / ((1 - self.alpha) * (1 - self.beta))
        if self.train_mcw:
            return prefactor * tf.reduce_mean(
                sigma
                * (
                    1
                    - tf.pow(tf.stop_gradient(q_test) / p_true, (1 - self.alpha) / 2.0)
                )
                * (1 - tf.pow(tf.stop_gradient(q_test) / p_true, (1 - self.beta) / 2.0))
                * p_true
                / tf.stop_gradient(q_sample),
                axis=0,
            )
        else:
            return prefactor * tf.reduce_mean(
                sigma
                * (
                    1
                    - tf.pow(q_test / tf.stop_gradient(p_true), (1 - self.alpha) / 2.0)
                )
                * (1 - tf.pow(q_test / tf.stop_gradient(p_true), (1 - self.beta) / 2.0))
                * tf.stop_gradient(p_true)
                / tf.stop_gradient(q_sample),
                axis=0,
            )

    def js_divergence(
        self,
        p_true: tf.Tensor,
        q_test: tf.Tensor,
        logp: tf.Tensor,
        logq: tf.Tensor,
        sigma: tf.Tensor = None,
        q_sample: tf.Tensor = None,
        channels: tf.Tensor = None,
    ):
        """Implement Jensen-Shannon (JS) divergence.

        This function returns the Jensen-Shannon divergence for two given sets
        of probabilities, ``p_true`` and ``q_test``. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of ``q_test``.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Jensen-Shannon is used as loss function.

        Arguments:
            p_true (tf.tensor): true probability
            q_test (tf.tensor): estimated probability
            logp (tf.tensor): logarithm of the true probability
            logq (tf.tensor): logarithm of the estimated probability
            sigma (tf.tensor): loss weights with shape (nsamples,). Defaults to None.
            q_sample (tf.tensor): sampling probability

        Returns:
            tf.tensor: computed Jensen-Shannon divergence

        """
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((tf.shape(q_test)[0],), dtype=self._dtype)

        if self.train_mcw:
            logm = tf.math.log(0.5 * (tf.stop_gradient(q_test) + p_true))
            return 0.5 * tf.reduce_mean(
                sigma * p_true * (logp - logm) / tf.stop_gradient(q_sample)
                + (q_test * (tf.stop_gradient(logq) - logm)),
                axis=0,
            )
        else:
            logm = tf.math.log(0.5 * (q_test + tf.stop_gradient(p_true)))
            return tf.reduce_mean(
                sigma
                * 0.5
                / tf.stop_gradient(q_sample)
                * (
                    (tf.stop_gradient(p_true) * (tf.stop_gradient(logp) - logm))
                    + (q_test * (logq - logm))
                ),
                axis=0,
            )

    def __call__(self, name):
        func = getattr(self, name, None)
        if func is not None:
            return func
        raise NotImplementedError(
            "The requested loss function {} "
            "is not implemented. Allowed "
            "options are {}.".format(name, self.divergences)
        )
