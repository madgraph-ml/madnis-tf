# Toy Experiments

We test the performance of our machine learned multi-channel weight and importance weight deduced
from our conditional flow and use it to benchmark
the various techniques to improve precision.

## Examples

In this folder we test various multi-dimensional examples

- 1-dimensional Camel distribution in the folder `camel`
    - 2 channels with analytic mappings
    - trainable multi-channel weight
- 2-dimensional distribution in the folder `ring`:
    - 2 channels with analytic mappings
    - trainable multi-channel weight
    - trainable conditional Flow
- $n$-dimensional Camel distribution on the unit hypercube in the folder `multidim`
    - variable dimension $n$ and variable number of peaks $m$
    - $k$ channels with or without analytic mappings
    - trainable multi-channel weight
    - trainable conditional flow
   
   
