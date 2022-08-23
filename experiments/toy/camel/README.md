# 1-Dimensional Camel

In this simple one-dimensional example we use 2 channel of
integrations in which we use a cauchy distribution to map out the two
gaussian peaks. We train a neural network to learn the optimal
multi-channel weight $\alpha_i(x)$. This can be done in two ways.
Either, the weight is trained from scratch or the training starts from
a prior weight $\beta_i(x)$ which obeys the normalization

$$ \sum_i \beta_i(x) = 1, \qquad 0 < \beta_i(x) <1,$$

and thus only some residual weight $\Delta_i(x,\theta)$ is learned. Requiring the normalization

$$
\sum_i \alpha_i(x,\theta) = 1, \qquad 0<\alpha_i(x,\theta) <1,
$$

by applying a softmax activation in the last layer the full weight is then given by:

$$
\alpha_i(x,\theta) = \frac{\beta_i(x)\cdot e^{\omega_i\cdot\Delta_i(x,\theta)}}{\sum_j \beta_j(x)\cdot e^{\omega_j\cdot\Delta_j(x,\theta)}},\qquad \text{with}\quad \omega_i^{(0)} = 0.
$$

## Function and mapping

We consider a `camel` function which is defined as

$$
f(x)= \frac{a_1}{\sqrt{2\pi\sigma_1^2}}\,\mathrm{exp}\left(-\frac{1}{2}\frac{(x-\mu_1)^2}{\sigma_1^2}\right) + \frac{a_2}{\sqrt{2\pi\sigma_2^2}}\,\mathrm{exp}\left(-\frac{1}{2}\frac{(x-\mu_2)^2}{\sigma_2^2}\right)
$$

For the analytic mapping we choose a cauchy distribution for each peak as given by

$$
g_i(x) = \frac{1}{\sqrt{\pi}}\frac{1}{\sqrt{2\pi\sigma_i^2}}\frac{2\sigma_i^2}{(x-\mu_i)^2+2\sigma^2_i}
    \quad \text{with} \quad \left\vert\frac{\partial G_i(x)}{x}\right\vert=g_i(x)\,,
$$

The corresponding inverse mapping (quantile) is known and given as

$$
x=G^{-1}_i(z)=\sqrt{2}\sigma_i \\,\tan\\!\left(\pi (z-{\tfrac {1}{2}})\right)+\mu_i.
$$

such that using the channel splitting $\sum_i \alpha_i(x)=1$, the total integral is given by

$$
I = \int\limits_{-\infty}^{\infty}\mathrm{d} x\\,f(x)
  =\sum_i \int\limits_{-\infty}^{\infty}\mathrm{d} x\\,\alpha_i(x)\\,f(x)
  =\sum_i \int\limits_{0}^{1}\mathrm{d} z\\,\left.\alpha_i(x)\\,\frac{f(x)}{g_i(x)}\right\vert_{x=G^{-1}_i(z)}.
$$





## Training

Commands for ```camel```:

```python
# train the multi-channel weights (optinally adding arguments, see --help)
python train_camel.py (--arg ARG)
```

Commands for `truncated camel` (with cut):

```python
# train the multi-channel weights (optinally adding arguments, see --help)
python train_trunc_camel.py (--arg ARG)
```
   
   
