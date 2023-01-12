#TODO: I moved all the single pass stuff over here. At the moment it is not working

from ..models.mc_integrator import MultiChannelIntegrator

class MultiChannelSinglePassIntegrator(MultiChannelIntegrator):
    def __init__(
        self,
        *args,
        second_order_opt: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.second_order_opt = second_order_opt
        self.divergence = Divergence(
            n_channels=self.n_channels, single_pass_opt=True, **kwargs
        )

    @tf.function(reduce_retracing=True)
    def _get_single_pass_probs(
        self,
        nsamples: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable = None,
        return_integrand: bool = False,
        return_alphas: bool = False,
    ):
        x, y, q_test, logq, alphas, alphas_prior = self._get_single_pass_probs_alphas(
            nsamples, channels, weight_prior
        )

        if return_integrand:
            if return_alphas:
                if alphas_prior is not None:
                    alphas_prior = tf.gather(alphas_prior, channels, batch_dims=1)
                return alphas, alphas_prior, alphas * self._func(y, channels) / q_test
            else:
                return alphas * self._func(y, channels) / q_test

        p_true, logp, means, vars, counts = self._get_probs_integral(
            nsamples, alphas, q_test, self._func(y, channels), channels
        )
        return (
            p_true,
            q_test,
            logp,
            logq,
            means,
            vars,
            counts,
            x,
            self._func(y, channels),
        )

    @tf.function
    def _get_single_pass_probs_alphas(
        self,
        nsamples: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable = None,
    ):
        one_hot_channels = tf.one_hot(channels, self.n_channels, dtype=self._dtype)
        x, logq = self.dist.sample_and_log_prob(nsamples, condition=one_hot_channels)
        y, logq = self._compute_analytic_mappings(x, logq, channels)
        q_test = tf.math.exp(logq)

        alphas, alphas_prior = self._get_alphas(y, weight_prior)
        alphas = tf.gather(alphas, channels, batch_dims=1)

        return x, y, q_test, logq, alphas, alphas_prior

    @tf.function
    def _single_pass_optimization_step(
        self,
        nsamples: tf.Tensor,
        channels: tf.Tensor,
        weight_prior: Callable,
    ):
        if not self.train_flow and not self.train_mcw:
            raise ValueError("No network defined which can be optimized")

        with tf.GradientTape() as tape:
            (
                p_true,
                q_test,
                logp,
                logq,
                means,
                vars,
                counts,
                samples,
                func_vals,
            ) = self._get_single_pass_probs(nsamples, channels, weight_prior)
            loss = self.loss_func(p_true, q_test, logp, logq, channels, q_sample=q_test)

        trainable_weights = []
        if self.train_flow:
            trainable_weights.extend(self.dist.trainable_weights)
        if self.train_mcw:
            trainable_weights.extend(self.mcw_model.trainable_weights)
        grads = tape.gradient(loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        return (
            loss,
            means,
            vars,
            counts,
            tf.stop_gradient(samples),
            tf.stop_gradient(q_test),
            tf.stop_gradient(func_vals),
        )

    def train_one_step(
        self,
        nsamples: int,
        integral: bool = False,
    ):
        """Perform one step of integration and improve the sampling.

        Args:
            nsamples (int): Number of samples to be taken in a training step
            integral (bool, optional): return the integral value. Defaults to False.

        Returns:
            loss: Value of the loss function for this step
            integral (optional): Estimate of the integral value
            uncertainty (optional): Integral statistical uncertainty
        """

        # Sample from flow and update
        channels = self._get_channels(
            tf.constant(nsamples),
            self._get_variance_weights(),
            self.uniform_channel_ratio,
        )

        (
            loss,
            means,
            vars,
            counts,
            samples,
            q_sample,
            func_vals,
        ) = self._single_pass_optimization_step(
            tf.constant(nsamples), channels, weight_prior
        )
        self._store_samples(x, y, q_sample, func_vals, alphas_prior, channels)

        self.variance_history.append(vars)
        self.count_history.append(counts)
        del self.variance_history[: -self.variance_history_length]
        del self.count_history[: -self.variance_history_length]

        if integral:
            return (
                loss,
                tf.reduce_sum(means),
                tf.sqrt(tf.reduce_sum(vars / (counts - 1.0))),
            )

        return loss
