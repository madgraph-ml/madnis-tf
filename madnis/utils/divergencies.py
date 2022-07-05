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

    Attributes:
        alpha (float): attribute needed for (alpha, beta)-product divergence
            and Chernoff divergence
        beta (float): attribute needed for (alpha, beta)-product divergence

    """

    def __init__(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.divergences = [
            x
            for x in dir(self)
            if ("__" not in x and "alpha" not in x and "beta" not in x)
        ]

    @staticmethod
    def variance(true, test, logp, logq):
        """Implement variance loss.

        This function returns the variance loss for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.
        The 'true' probability can also be used unnormalized.

        tf.stop_gradient is used such that the correct gradient is returned
        when the variance is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points. (dont has to be normalized!!)
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): not used in variance
            logq (tf.tensor or array(nbatch) of floats): not used in variance

        Returns:
            tf.tensor(float): computed variance loss

        """
        del logp, logq
        return (
            tf.reduce_mean(tf.stop_gradient(true) ** 2 / test / tf.stop_gradient(test))
            - tf.reduce_mean(tf.stop_gradient(true / test)) ** 2
        )

    @staticmethod
    def chi2(true, test, logp, logq):
        """Implement Neyman chi2 divergence.

        This function returns the Neyman chi2 divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the chi2 is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): not used in chi2
            logq (tf.tensor or array(nbatch) of floats): not used in chi2

        Returns:
            tf.tensor(float): computed Neyman chi2 divergence

        """
        del logp, logq
        return tf.reduce_mean(
            input_tensor=(tf.stop_gradient(true) - test) ** 2
            / test
            / tf.stop_gradient(test)
        )

    @staticmethod
    def pchi2(true, test, logp, logq):
        """Implement Pearson chi2 divergence.

        This function returns the Pearson chi2 divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the chi2 is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): not used in chi2
            logq (tf.tensor or array(nbatch) of floats): not used in chi2

        Returns:
            tf.tensor(float): computed Pearson chi2 divergence

        """
        del logp, logq
        return tf.reduce_mean(
            input_tensor=(test - tf.stop_gradient(true)) ** 2
            / true
            / tf.stop_gradient(test)
        )

    # pylint: disable=invalid-name
    @staticmethod
    def kl(true, test, logp, logq):
        """Implement Kullback-Leibler (KL) divergence.

        This function returns the Kullback-Leibler divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the KL is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): logarithm of the true probability
            logq (tf.tensor or array(nbatch) of floats): logarithm of the estimated probability

        Returns:
            tf.tensor(float): computed KL divergence

        """
        return tf.reduce_mean(
            input_tensor=tf.stop_gradient(true / test) * (tf.stop_gradient(logp) - logq)
        )

    # pylint: disable=invalid-name
    @staticmethod
    def rkl(true, test, logp, logq):
        """Implement reverse Kullback-Leibler (RKL) divergence.

        This function returns the reverse Kullback-Leibler divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the RKL is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): not used in RKL
            test (tf.tensor or array(nbatch) of floats): not used in RKL
            logp (tf.tensor or array(nbatch) of floats): logarithm of the true probability
            logq (tf.tensor or array(nbatch) of floats): logarithm of the estimated probability

        Returns:
            tf.tensor(float): computed RKL divergence

        """
        del true, test
        return tf.reduce_mean((1 + tf.stop_gradient(logq - logp)) * logq)

    # pylint: enable=invalid-name
    @staticmethod
    def hellinger(true, test, logp, logq):
        """Implement Hellinger divergence.

        This function returns the Hellinger distance for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Hellinger is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): not used in hellinger
            logq (tf.tensor or array(nbatch) of floats): not used in hellinger

        Returns:
            tf.tensor(float): computed Hellinger distance

        """
        del logp, logq
        return tf.reduce_mean(
            input_tensor=(
                2.0
                * (tf.stop_gradient(tf.math.sqrt(true)) - tf.math.sqrt(test)) ** 2
                / tf.stop_gradient(test)
            )
        )

    @staticmethod
    def jeffreys(true, test, logp, logq):
        """Implement Jeffreys divergence.

        This function returns the Jeffreys divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Jeffreys is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): logarithm of the true probability
            logq (tf.tensor or array(nbatch) of floats): logarithm of the estimated probability

        Returns:
            tf.tensor(float): computed Jeffreys divergence

        """
        return tf.reduce_mean(
            input_tensor=(
                (tf.stop_gradient(true) - test)
                * (tf.stop_gradient(logp) - logq)
                / tf.stop_gradient(test)
            )
        )

    def chernoff(self, true, test, logp, logq):
        """Implement Chernoff divergence.

        This function returns the Chernoff divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Chernoff is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): not used in chernoff
            logq (tf.tensor or array(nbatch) of floats): not used in chernoff

        Returns:
            tf.tensor(float): computed Chernoff divergence

        Raises:
           ValueError: If there is no alpha defined or alpha is not between 0 and 1

        """
        del logp, logq
        if self.alpha is None:
            raise ValueError("Must give an alpha value to use Chernoff " "Divergence.")
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1.")

        return (
            4.0
            / (1 - self.alpha ** 2)
            * (
                1
                - tf.reduce_mean(
                    input_tensor=(
                        tf.stop_gradient(tf.pow(true, (1.0 - self.alpha) / 2.0))
                        * tf.pow(test, (1.0 + self.alpha) / 2.0)
                        / tf.stop_gradient(test)
                    )
                )
            )
        )

    @staticmethod
    def exponential(true, test, logp, logq):
        """Implement Expoential divergence.

        This function returns the Exponential divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Exponential is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): logarithm of the true probability
            logq (tf.tensor or array(nbatch) of floats): logarithm of the estimated probability

        Returns:
            tf.tensor(float): computed Exponential divergence

        """
        return tf.reduce_mean(
            input_tensor=tf.stop_gradient(true / test)
            * (tf.stop_gradient(logp) - logq) ** 2
        )

    @staticmethod
    def exponential2(true, test, logp, logq):
        """Implement Expoential divergence with true and test interchanged.

        This function returns the Exponential2 divergence for two given sets
        of probabilities, true and test. In contrast to the Exponential
        divergence, it has true and test interchanged. It uses importance
        sampling, i.e. the estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Exponential2 is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): logarithm of the true probability
            logq (tf.tensor or array(nbatch) of floats): logarithm of the estimated probability

        Returns:
            tf.tensor(float): computed Exponential2 divergence

        """
        return tf.reduce_mean(
            input_tensor=tf.stop_gradient(true ** 2 / test)
            * (tf.stop_gradient(logp) - logq) ** 2
            / test
        )

    def ab_product(self, true, test, logp, logq):
        """Implement (alpha, beta)-product divergence.

        This function returns the (alpha, beta)-product divergence for two given
        sets of probabilities, true and test. It uses importance sampling,
        i.e. the estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the ab_product is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): not used in ab_product
            logq (tf.tensor or array(nbatch) of floats): not used in ab_product

        Returns:
            tf.tensor(float): computed (alpha, beta)-product divergence

        Raises:
           ValueError: If there is no alpha defined or alpha is not between 0 and 1
           ValueError: If there is no beta defined or beta is not between 0 and 1

        """
        del logp, logq
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

        return tf.reduce_mean(
            input_tensor=(
                2.0
                / ((1 - self.alpha) * (1 - self.beta))
                * (1 - tf.pow(test / tf.stop_gradient(true), (1 - self.alpha) / 2.0))
                * (1 - tf.pow(test / tf.stop_gradient(true), (1 - self.beta) / 2.0))
                * tf.stop_gradient(true / test)
            )
        )

    # pylint: disable=invalid-name
    @staticmethod
    def js(true, test, logp, logq):
        """Implement Jensen-Shannon divergence.

        This function returns the Jensen-Shannon divergence for two given
        sets of probabilities, true and test. It uses importance sampling,
        i.e. the estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Jenson-Shannon is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): logarithm of the true probability
            logq (tf.tensor or array(nbatch) of floats): logarithm of the estimated probability

        Returns:
            tf.tensor(float): computed Jensen-Shannon divergence

        """
        logm = tf.math.log(0.5 * (test + tf.stop_gradient(true)))
        return tf.reduce_mean(
            input_tensor=(
                tf.stop_gradient(0.5 / test)
                * (
                    (tf.stop_gradient(true) * (tf.stop_gradient(logp) - logm))
                    + (test * (logq - logm))
                )
            )
        )

    # pylint: enable=invalid-name
    def __call__(self, name):
        func = getattr(self, name, None)
        if func is not None:
            return func
        raise NotImplementedError(
            "The requested loss function {} "
            "is not implemented. Allowed "
            "options are {}.".format(name, self.divergences)
        )
