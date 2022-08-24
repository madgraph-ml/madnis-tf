# $n$-Dimensional Camel

In this example we consider a $n$-dimensional camel with $m$ modes, as given by

$$
f_1(\vec{x})
    = \sum_{i=1}^m \frac{\omega_i}{\left(\sigma_i\sqrt{2\pi}\right)^{n}}\,\exp\left[-\frac{1}{2}\frac{ \vert\vec{x}-\vec{\mu}_i\vert^2}{\sigma^2_i}\right].
$$

which integrated on the unit-hypercube $[0,1]^n$ has the integral

$$
I_1 = \int\limits_{0}^{1}\mathrm{d}^n x\ f_1(\vec{x})=\sum_{i=1}^M\ \omega_i\ \prod_{j=1}^n\left(\frac{1}{2}\left[\mathrm{erf}\left(\frac{1-\mu^j_i}{\sqrt{2}\sigma^j_i}\right)+\mathrm{erf}\left(\frac{\mu^j_i}{\sqrt{2}\sigma^j_i}\right)\right]\right).
$$

## Training

Commands for `flow`:

```python
# train the flow only with fixed weights (1/nchannels) (optinally adding arguments, see --help)
python train_mcflow.py (--arg ARG)
```
When choosing ```--channels=1```, this is a normal i-flow integrator.

Commands for `mc-flow`:

```python
# train the flow simultanously with the multi-channel weights (optinally adding arguments, see --help)
python train_mcflow.py (--arg ARG)
```

One can possibly alter the complexity of the integration by modifying the follwing args:

```bash
--dims: dimensionality of the integral (int, default=2)
--channels: number of channels (int, default=2)
--modes: number of peaks (int, default=2)
```
