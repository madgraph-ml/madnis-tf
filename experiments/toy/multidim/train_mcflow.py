import os
import tensorflow as tf
import numpy as np
import argparse
import time

from mcw import mcw_model, residual_mcw_model
from utils import integrate, error
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.distributions.camel import MultiDimCamel, NormalizedMultiDimCamel
from madnis.nn.nets.mlp import MLP
from vegasflow import VegasFlow

import sys

# Use double precision
tf.keras.backend.set_floatx("float64")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#########
# Setup #
#########

parser = argparse.ArgumentParser()

# function specifications
parser.add_argument("--dims", type=int, default=2)
parser.add_argument("--modes", type=int, default=2)

# Data params
parser.add_argument("--train_samples", type=int, default=128 * 10000)
parser.add_argument("--int_samples", type=int, default=10000)

# model params
parser.add_argument("--use_prior_weights", action='store_true')
parser.add_argument("--units", type=int, default=16)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--blocks", type=int, default=6)
parser.add_argument("--activation", type=str, default="leakyrelu", choices={"relu", "elu", "leakyrelu", "tanh"})
parser.add_argument("--initializer", type=str, default="glorot_uniform", choices={"glorot_uniform", "he_uniform"})
parser.add_argument("--loss", type=str, default="variance", choices={"variance", "neyman_chi2", "kl_divergence"})

# mcw model params
parser.add_argument("--mcw_units", type=int, default=16)
parser.add_argument("--mcw_layers", type=int, default=2)

# Define the number of channels
parser.add_argument("--channels", type=int, default=2)

# Train params
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)

args = parser.parse_args()

################################
# Define distributions
################################

DTYPE = tf.keras.backend.floatx()
DIMS_IN = args.dims  # dimensionality of data space
N_MODES = args.modes  # number of modes
N_CHANNELS = args.channels  # number of Channels

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
res = integrate(integrand).numpy()
err = error(integrand).numpy()
relerr = err / res * 100

print(f"\n Naive integration ({INT_SAMPLES:.1e} samples):")
print("-----------------------------------------------------------")
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)")
print("-----------------------------------------------------------\n")


################################
# Define the flow network
################################

PRIOR = args.use_prior_weights

FLOW_META = {
    "units": args.units,
    "layers": args.layers,
    "initializer": args.initializer,
    "activation": args.activation,
}

N_BLOCKS = args.blocks

flow = VegasFlow(
    [DIMS_IN],
    dims_c=[[N_CHANNELS]],
    n_blocks=N_BLOCKS,
    subnet_meta=FLOW_META,
    subnet_constructor=MLP,
    hypercube_target=False, # CHECK?
)

################################
# Define the mcw network
################################

MCW_META = {
    "units": args.mcw_units,
    "layers": args.mcw_layers,
    "initializer": args.initializer,
    "activation": args.activation,
}

if PRIOR:
    mcw_net = residual_mcw_model(
        dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=MCW_META
    )
else:
    mcw_net = mcw_model(dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=MCW_META)

################################
# Define the prior
################################

madgraph_prior = None

################################
# Define the integrator
################################

# Define training params
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
LOSS = args.loss

# Number of samples
TRAIN_SAMPLES = args.train_samples
ITERS = TRAIN_SAMPLES // BATCH_SIZE

# Decay of learning rate
DECAY_RATE = 0.01
DECAY_STEP = ITERS

# Prepare scheduler and optimzer
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)

opt1 = tf.keras.optimizers.Adam(lr_schedule)
opt2 = tf.keras.optimizers.Adam(lr_schedule)

integrator = MultiChannelIntegrator(
    multi_camel, flow, [opt1, opt2], mcw_model=mcw_net, use_weight_init=PRIOR, n_channels=N_CHANNELS, loss_func=LOSS)

################################
# Pre train - integration
################################

res, err = integrator.integrate(INT_SAMPLES, weight_prior=madgraph_prior)
relerr = err / res * 100

print(f"\n Pre Multi-Channel integration ({INT_SAMPLES:.1e} samples):")
print("--------------------------------------------------------------")
print(f" Number of channels: {N_CHANNELS}                            ")
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %) ")
print("------------------------------------------------------------\n")


################################
# Train the network
################################

train_losses = []
start_time = time.time()
for e in range(EPOCHS):

    batch_train_losses = []
    # do multiple iterations.
    for _ in range(ITERS):
        batch_loss = integrator.train_one_step(BATCH_SIZE, weight_prior=madgraph_prior)
        batch_train_losses.append(batch_loss)

    train_loss = tf.reduce_mean(batch_train_losses)
    train_losses.append(train_loss)

    if (e + 1) % 1 == 0:
        # Print metrics
        print(
            "Epoch #{}: Loss: {}, Learning_Rate: {}".format(
                e + 1, train_losses[-1], opt1._decayed_lr(tf.float32)
            )
        )
end_time = time.time()
print("--- Run time: %s hour ---" % ((end_time - start_time) / 60 / 60))
print("--- Run time: %s mins ---" % ((end_time - start_time) / 60))
print("--- Run time: %s secs ---" % ((end_time - start_time)))

################################
# After train - integration
################################

res, err = integrator.integrate(INT_SAMPLES, weight_prior=madgraph_prior)
relerr = err / res * 100

print(f"\n Opt. Multi-Channel integration ({INT_SAMPLES:.1e} samples):")
print("---------------------------------------------------------------")
print(f" Number of channels: {N_CHANNELS}                             ")
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)  ")
print("-------------------------------------------------------------\n")
