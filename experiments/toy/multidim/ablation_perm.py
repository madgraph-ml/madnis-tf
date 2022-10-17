# pylint: disable=invalid-name
""" Script to test different permutation options in numerical integration.

    included options:
        - exchange (sets A and B are simply interchanged)
        - random (random perm. every second step, interchanges in between)
        - log (uses logarithmic bisection as in i-flow)
        - softperm (random, but fixed SO(n) matrix)
        - trainperm (trainable SO(n) matrix)

    Used for MadNIS.

    by Claudius Krause
"""

import os
import tensorflow as tf
import numpy as np
import argparse
import time

from madnis.utils.train_utils import integrate
from madnis.distributions.camel import NormalizedMultiDimCamel
from madnis.nn.nets.mlp import MLP
from vegasflow import RQSVegasFlow


# Use double precision
tf.keras.backend.set_floatx("float64")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#########
# Setup #
#########

parser = argparse.ArgumentParser()

# function specifications
parser.add_argument("--dims", type=int, default=8)
parser.add_argument("--modes", type=int, default=2)

# Data params
parser.add_argument("--train_batches", type=int, default=1000)
parser.add_argument("--int_samples", type=int, default=10000)


# Train params
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)

# model params
parser.add_argument("--units", type=int, default=16)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--blocks", type=int, default=6)
parser.add_argument("--activation", type=str, default="leakyrelu",
                    choices={"relu", "elu", "leakyrelu", "tanh"})
parser.add_argument("--initializer", type=str, default="glorot_uniform",
                    choices={"glorot_uniform", "he_uniform"})


args = parser.parse_args()


################################
# Define distribution
################################

DTYPE = tf.keras.backend.floatx()
DIMS_IN = args.dims  # dimensionality of data space
N_MODES = args.modes  # number of modes
N_CHANNELS = 1  # number of Channels, removed for permutation tests

# Define peak positions and heights
# Feel free to change
MEANS = []
SIGMAS = []
for i in range(N_MODES):
    MEANS.append(tf.constant([[(i+1)/(N_MODES+1)] * DIMS_IN], dtype=DTYPE))
    SIGMAS.append(0.1/N_MODES)

# Define truth distribution
multi_camel = NormalizedMultiDimCamel(MEANS, SIGMAS, DIMS_IN)

print(f"\n Integrand specifications:")
print("-----------------------------------------------------------")
print(f" Dimensions: {DIMS_IN}                                    ")
print(f" Modes: {N_MODES}                                         ")
print(f" Channels: {N_CHANNELS} (not for naive integration)       ")
print("-----------------------------------------------------------\n")


################################
# Naive integration
################################

INT_SAMPLES = args.int_samples

# Uniform sampling in range [0,1]^d
d = DIMS_IN
volume = 1.0 ** d
noise = tf.random.uniform((INT_SAMPLES, d), dtype=DTYPE) * volume
rho = 1 / volume
integrand = multi_camel.prob(noise) / rho  # divide by volume in this case
res, err = integrate(integrand)
relerr = err / res * 100

print(f"\n Naive integration ({INT_SAMPLES:.1e} samples):")
print("-----------------------------------------------------------")
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)")
print("-----------------------------------------------------------\n")

################################
# Define the flow network
################################

#PRIOR = args.use_prior_weights

FLOW_META = {
    "units": args.units,
    "layers": args.layers,
    "initializer": args.initializer,
    "activation": args.activation,
}

N_BLOCKS = args.blocks

flows = {}

for perm in ['exchange', 'random', 'log', 'soft', 'softlearn']:
    flows[perm] = RQSVegasFlow(
        [DIMS_IN],
        dims_c=[[N_CHANNELS]],
        n_blocks=N_BLOCKS,
        subnet_meta=FLOW_META,
        subnet_constructor=MLP,
        hypercube_target=True,
        permutations=perm
    )
