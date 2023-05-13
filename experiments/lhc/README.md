# LHC Experiments

We test the performance of our machine learned multi-channel weight and importance weight deduced
from our conditional flow on real-world LHC examples

## Examples

In this folder we consider a toy LHC example to test various aspects of our framework

- Drell-Yan process in the folder `drell-yan`
    - Contains the fully differentiable matrix element for DY
    - trainable multi-channel weight $\alpha_i(x,\theta)$
    - trainable conditional Flow $H(x,\varphi\vert i)$ with
    jacobian determinants $h(x,\varphi\vert i)$.
