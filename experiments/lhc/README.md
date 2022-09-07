# LHC Experiments

We test the performance of our machine learned multi-channel weight and importance weight deduced
from our conditional flow on real-world LHC examples

## Examples

In this folder we consider multiple LHC examples to test various aspects of our framework

- Drell-Yan process in the folder `drell-yan`
    - Containing data for the $\mathrm{Z}/\gamma$ interference term
    - Data for 2 channels of integration.
    - trainable multi-channel weight $\alpha_i(x,\theta)$
- W + 2 jet production in the folder `wp2j`
    - Contains the API to acess the matrix-element for this process
    - trainable multi-channel weight $\alpha_i(x,\theta)$
    - trainable conditional Flow $H(x,\varphi\vert i)$ with
    jacobian determinants $h(x,\varphi\vert i)$.
