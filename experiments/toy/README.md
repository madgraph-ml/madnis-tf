# Toy Experiments

We test the performance of our machine learned multi-channel weight and importance weight deduced
from our conditional flow and use it to benchmark
the various techniques to improve precision.

## Examples

In this folder we test various multi-dimensional examples

- 1-dimensional Camel distribution in the folder `camel`
    - 2 channels with analytic mappings $G_i: \Phi \to U_i$
    and jacobian determinants $g_i(x)$.
    - trainable multi-channel weight $\alpha_i(x,\theta)$
- 2-dimensional distribution in the folder `ring`:
    - 2 channels with or without analytic mappings $G_i: \Phi \to U_i$ and jacobian determinants $g_i(x)$.
    - trainable multi-channel weight $\alpha_i(x;\theta)$
    - trainable conditional Flow $H(x,\varphi\vert i)$ with
    jacobian determinants $h(x,\varphi\vert i)$.
- $n$-dimensional Camel distribution on the unit hypercube in the folder `multidim`
    - variable dimension $n$ and variable number of peaks $m$
    - $k$ channels with or without analytic mappings $G_i: \Phi \to U_i$
    and jacobian determinants $g_i(x)$.
    - trainable multi-channel weight $\alpha_i(x;\theta)$
    - trainable conditional Flow $H(x,\varphi\vert i)$ with
    jacobian determinants $h(x,\varphi\vert i)$.
   
   
