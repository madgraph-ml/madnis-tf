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
            sigma (tf.tensor): loss weights with shape (n_channels,). Defaults to None.
            q_sample (tf.tensor): sampling probability
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed variance loss

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        var = 0
        if self.train_mcw:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                mean2 = tf.reduce_mean(
                    pi ** 2 / (tf.stop_gradient(qti * qsi)),
                    axis=0,
                )
                mean = tf.reduce_mean(pi / tf.stop_gradient(qsi), axis=0)
                var += sigi * (mean2 - mean ** 2)
        else:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                mean2 = tf.reduce_mean(
                    tf.stop_gradient(pi) ** 2 / (qti * tf.stop_gradient(qsi)),
                    axis=0,
                )
                mean = tf.reduce_mean(tf.stop_gradient(pi / qsi), axis=0)
                var += sigi * (mean2 - mean ** 2)

        return var

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Neyman chi2 divergence

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        chi2 = 0
        if self.train_mcw:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                chi2i = tf.reduce_mean(
                    (pi - tf.stop_gradient(qti)) ** 2 / (tf.stop_gradient(qti * qsi)),
                    axis=0,
                )
                chi2 += sigi * chi2i
        else:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                chi2i = tf.reduce_mean(
                    (tf.stop_gradient(pi) - qti) ** 2 / (qti * tf.stop_gradient(qsi)),
                    axis=0,
                )
                chi2 += sigi * chi2i

        return chi2

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Pearson chi2 divergence

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        chi2 = 0
        if self.train_mcw:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                chi2i = tf.reduce_mean(
                    (tf.stop_gradient(qti) - pi) ** 2 / (pi * tf.stop_gradient(qsi)),
                    axis=0,
                )
                chi2 += sigi * chi2i
        else:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                chi2i = tf.reduce_mean(
                    (qti - tf.stop_gradient(pi)) ** 2 / (tf.stop_gradient(pi * qsi)),
                    axis=0,
                )
                chi2 += sigi * chi2i

        return chi2

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed KL divergence

        """
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        logps = tf.dynamic_partition(logp, channels, self.n_channels)
        logqs = tf.dynamic_partition(logq, channels, self.n_channels)
        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        kl = 0
        if self.train_mcw:
            for logpi, logqi, pi, qsi, sigi in zip(logps, logqs, ps, qs, sigmas):
                kli = tf.reduce_mean(
                    pi / tf.stop_gradient(qsi) * (logpi - tf.stop_gradient(logqi)),
                    axis=0,
                )
                kl += sigi * kli
        else:
            for logpi, logqi, pi, qsi, sigi in zip(logps, logqs, ps, qs, sigmas):
                kli = tf.reduce_mean(
                    tf.stop_gradient(pi / qsi) * (tf.stop_gradient(logpi) - logqi),
                    axis=0,
                )
                kl += sigi * kli

        return kl

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed RKL divergence

        """
        del p_true
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        logps = tf.dynamic_partition(logp, channels, self.n_channels)
        logqs = tf.dynamic_partition(logq, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        rkl = 0
        if self.train_mcw:
            for logpi, logqi, qsi, qti, sigi in zip(logps, logqs, qs, qt, sigmas):
                rkli = tf.reduce_mean(
                    tf.stop_gradient(qti / qsi) * (tf.stop_gradient(logqi) - logpi),
                    axis=0,
                )
                rkl += sigi * rkli
        else:
            for logpi, logqi, qsi, qti, sigi in zip(logps, logqs, qs, qt, sigmas):
                rkli = tf.reduce_mean(
                    qti / tf.stop_gradient(qsi) * (logqi - tf.stop_gradient(logpi)),
                    axis=0,
                )
                rkl += sigi * rkli

        return rkl

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Hellinger distance

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        hell = 0
        if self.train_mcw:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                helli = tf.reduce_mean(
                    2.0
                    * (tf.math.sqrt(pi) - tf.stop_gradient(tf.math.sqrt(qti))) ** 2
                    / tf.stop_gradient(qsi),
                    axis=0,
                )
                hell += sigi * helli
        else:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                helli = tf.reduce_mean(
                    2.0
                    * (tf.stop_gradient(tf.math.sqrt(pi)) - tf.math.sqrt(qti)) ** 2
                    / tf.stop_gradient(qsi),
                    axis=0,
                )
                hell += sigi * helli

        return hell

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Jeffreys divergence

        """
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        logps = tf.dynamic_partition(logp, channels, self.n_channels)
        logqs = tf.dynamic_partition(logq, channels, self.n_channels)
        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        jeff = 0
        if self.train_mcw:
            for logpi, logqi, pi, qsi, qti, sigi in zip(
                logps, logqs, ps, qs, qt, sigmas
            ):
                jeffi = tf.reduce_mean(
                    (pi - tf.stop_gradient(qti))
                    * (logpi - tf.stop_gradient(logqi))
                    / tf.stop_gradient(qsi),
                    axis=0,
                )
                jeff += sigi * jeffi
        else:
            for logpi, logqi, pi, qsi, qti, sigi in zip(
                logps, logqs, ps, qs, qt, sigmas
            ):
                jeffi = tf.reduce_mean(
                    (tf.stop_gradient(pi) - qti)
                    * (tf.stop_gradient(logpi) - logqi)
                    / tf.stop_gradient(qsi),
                    axis=0,
                )
                jeff += sigi * jeffi

        return jeff

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Chernoff divergence

        Raises:
           ValueError: If there is no alpha defined or alpha is not between 0 and 1

        """
        del logp, logq
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        if self.alpha is None:
            raise ValueError("Must give an alpha value to use Chernoff " "Divergence.")
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1.")

        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)
        prefactor = 4.0 / (1 - self.alpha ** 2)

        vlad = 0
        if self.train_mcw:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                int = tf.reduce_mean(
                    tf.pow(pi, (1.0 - self.alpha) / 2.0)
                    * tf.stop_gradient(tf.pow(qti, (1.0 + self.alpha) / 2.0))
                    / tf.stop_gradient(qsi),
                    axis=0,
                )
                vlad += sigi * prefactor * (1 - int)
        else:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                int = tf.reduce_mean(
                    tf.stop_gradient(tf.pow(pi, (1.0 - self.alpha) / 2.0))
                    * tf.pow(qti, (1.0 + self.alpha) / 2.0)
                    / tf.stop_gradient(qsi),
                    axis=0,
                )
                vlad += sigi * prefactor * (1 - int)

        return vlad

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Exponential divergence

        """
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        logps = tf.dynamic_partition(logp, channels, self.n_channels)
        logqs = tf.dynamic_partition(logq, channels, self.n_channels)
        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        exp = 0
        if self.train_mcw:
            for logpi, logqi, pi, qsi, sigi in zip(logps, logqs, ps, qs, sigmas):
                expi = tf.reduce_mean(
                    pi / tf.stop_gradient(qsi) * (logpi - tf.stop_gradient(logqi)) ** 2,
                    axis=0,
                )
                exp += sigi * expi
        else:
            for logpi, logqi, pi, qsi, sigi in zip(logps, logqs, ps, qs, sigmas):
                expi = tf.reduce_mean(
                    tf.stop_gradient(pi / qsi) * (tf.stop_gradient(logpi) - logqi) ** 2,
                    axis=0,
                )
                exp += sigi * expi

        return exp

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Exponential2 divergence

        """
        del p_true
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        logps = tf.dynamic_partition(logp, channels, self.n_channels)
        logqs = tf.dynamic_partition(logq, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        exp2 = 0
        if self.train_mcw:
            for logpi, logqi, qsi, qti, sigi in zip(logps, logqs, qs, qt, sigmas):
                exp2i = tf.reduce_mean(
                    tf.stop_gradient(qti / qsi)
                    * (tf.stop_gradient(logqi) - logpi) ** 2,
                    axis=0,
                )
                exp2 += sigi * exp2i
        else:
            for logpi, logqi, qsi, qti, sigi in zip(logps, logqs, qs, qt, sigmas):
                exp2i = tf.reduce_mean(
                    qti
                    / tf.stop_gradient(qsi)
                    * (logqi - tf.stop_gradient(logpi)) ** 2,
                    axis=0,
                )
                exp2 += sigi * exp2i

        return exp2

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

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
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

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

        logps = tf.dynamic_partition(logp, channels, self.n_channels)
        logqs = tf.dynamic_partition(logq, channels, self.n_channels)
        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)
        prefactor = 2.0 / ((1 - self.alpha) * (1 - self.beta))

        ab_prod = 0
        if self.train_mcw:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                ab_prodi = tf.reduce_mean(
                    (1 - tf.pow(tf.stop_gradient(qti) / pi, (1 - self.alpha) / 2.0))
                    * (1 - tf.pow(tf.stop_gradient(qti) / pi, (1 - self.beta) / 2.0))
                    * pi
                    / tf.stop_gradient(qsi),
                    axis=0,
                )
                ab_prod += sigi * ab_prodi
        else:
            for pi, qsi, qti, sigi in zip(ps, qs, qt, sigmas):
                ab_prodi = tf.reduce_mean(
                    (1 - tf.pow(qti / tf.stop_gradient(pi), (1 - self.alpha) / 2.0))
                    * (1 - tf.pow(qti / tf.stop_gradient(pi), (1 - self.beta) / 2.0))
                    * tf.stop_gradient(pi / qsi),
                    axis=0,
                )
                ab_prod += sigi * ab_prodi

        return prefactor * ab_prod

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
            channels (tf.tensor): encoding which channel to use with shape (nsamples,).

        Returns:
            tf.tensor: computed Jensen-Shannon divergence

        """
        if q_sample is None:
            q_sample = q_test
        if sigma is None:
            sigma = tf.ones((self.n_channels,), dtype=self._dtype)

        logps = tf.dynamic_partition(logp, channels, self.n_channels)
        logqs = tf.dynamic_partition(logq, channels, self.n_channels)
        ps = tf.dynamic_partition(p_true, channels, self.n_channels)
        qt = tf.dynamic_partition(q_test, channels, self.n_channels)
        qs = tf.dynamic_partition(q_sample, channels, self.n_channels)
        sigmas = tf.dynamic_partition(sigma, tf.range(self.n_channels), self.n_channels)

        js = 0
        if self.train_mcw:
            for logpi, logqi, pi, qsi, qti, sigi in zip(
                logps, logqs, ps, qs, qt, sigmas
            ):
                logm = tf.math.log(0.5 * (tf.stop_gradient(qti) + pi))
                jsi = tf.reduce_mean(
                    0.5
                    / tf.stop_gradient(qsi)
                    * (
                        pi * (logpi - logm)
                        + tf.stop_gradient(qti) * (tf.stop_gradient(logqi) - logm)
                    ),
                    axis=0,
                )
                js += sigi * jsi
        else:
            for logpi, logqi, pi, qsi, qti, sigi in zip(
                logps, logqs, ps, qs, qt, sigmas
            ):
                logm = tf.math.log(0.5 * (qti + tf.stop_gradient(pi)))
                jsi = tf.reduce_mean(
                    0.5
                    / tf.stop_gradient(qsi)
                    * (
                        tf.stop_gradient(pi) * (tf.stop_gradient(logpi) - logm)
                        + qti * (logqi - logm)
                    ),
                    axis=0,
                )
                js += sigi * jsi

        return js

    def __call__(self, name):
        func = getattr(self, name, None)
        if func is not None:
            return func
        raise NotImplementedError(
            "The requested loss function {} "
            "is not implemented. Allowed "
            "options are {}.".format(name, self.divergences)
        )
