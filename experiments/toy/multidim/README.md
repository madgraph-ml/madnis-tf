# $n$-Dimensional Camel

In this example we consider a $n$-dimensional camel with $m$ modes, as given by

$$
f_{\text{multi}}(\vec{x})
= \sum_{i=1}^m \frac{\omega_i}{\left(\sigma_i\sqrt{2\pi}\right)^{n}}\ \exp\left\{-\frac{1}{2}\frac{ (\vec{x}-\vec{\mu}_i)^2}{\sigma^2_i}\right\}.
$$


## Training

Commands for `multi-dim-camel`:

```python
# train the flow simultanously with the multi-channel weights (optinally adding arguments, see --help)
python train_multi.py (--arg ARG)
```

One can possibly alter the complexity of the integration by modifying the follwing args:

```bash
--dims: dimensionality of the integral (int, default=2)
--channels: number of channels (int, default=2)
--modes: number of peaks (int, default=2)
```
