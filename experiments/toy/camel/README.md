# 1-Dimensional Camel

In this simple one-dimensional example we use 2 channel of
integrations in which we use a cauchy distribution to map out the two
gaussian peaks. We train a neural network to learn the optimal
multi-channel weight $\alpha_i(x)$. This can be done in two ways.

Either, the weight is trained from scratch or the training starts from
a prior weight $\beta_i(x)$ which obeys the normalization

```math
\sum_i \beta_i(x) = 1, \qquad 0 < \beta_i(x) <1,
```

and thus only some residual weight $\Delta_i(x,\theta)$ is learned. Requiring the normalization

```math
\sum_i \alpha_i(x,\theta) = 1, \qquad 0<\alpha_i(x,\theta) <1,
```
by applying a softmax activation in the last layer the full weight is then given by:

```math
\alpha_i(x,\theta) = \frac{\beta_i(x)\cdot e^{\omega_i\cdot\Delta_i(x,\theta)}}{\sum_j \beta_j(x)\cdot e^{\omega_j\cdot\Delta_j(x,\theta)}},\qquad \text{with}\quad \omega_i^{(0)} = 0.
```


## Training

Commands for ```camel```:

```python
# train the multi-channel weights (optinally adding arguments, see --help)
python train_camel.py (--arg ARG)
```
   
   
