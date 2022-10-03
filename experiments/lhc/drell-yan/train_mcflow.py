import os

from madnis.mappings.phasespace_2p import TwoParticlePhasespaceB
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import argparse
import time

from mcw import mcw_model, residual_mcw_model
from madnis.utils.train_utils import integrate
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.distributions.camel import NormalizedMultiDimCamel
from madnis.nn.nets.mlp import MLP
from dy_integrand import DrellYan, MZ, WZ
from vegasflow import VegasFlow, RQSVegasFlow

import sys

# Use double precision
tf.keras.backend.set_floatx("float64")

#########
# Setup #
#########

parser = argparse.ArgumentParser()

# Data params
parser.add_argument("--train_batches", type=int, default=1000)
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
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-3)

args = parser.parse_args()

################################
# Define integrand
################################

DTYPE = tf.keras.backend.floatx()
DIMS_IN = 4  # dimensionality of data space
N_CHANNELS = args.channels  # number of Channels. Default is 2
INT_SAMPLES = args.int_samples
RES_TO_PB = 0.389379 * 1e9 # Conversion factor

# Define truth integrand
integrand = DrellYan(["u", "d", "c", "s", "u", "d", "c", "s"], input_format="convpolar")
#integrand = lambda x: tf.constant(1.0, dtype=DTYPE) # For testing phase-space volume

print(f"\n Integrand specifications:")
print("-----------------------------------------------------------")
print(f" Dimensions: {DIMS_IN}                                    ")
print(f" Channels: {N_CHANNELS}                                   ")
print("-----------------------------------------------------------\n")

# Define the channel mappings
map_Z = TwoParticlePhasespaceB(s_mass=MZ, s_gamma=WZ)
map_y = TwoParticlePhasespaceB()

# # TODO: Make flat but consider cut m_inv > 50 GeV
# # Otherwise infinite cross section!
# map_flat = TwoParticlePhasespaceFlatB()

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

flow = RQSVegasFlow(
    [DIMS_IN],
    dims_c=[[N_CHANNELS]],
    n_blocks=N_BLOCKS,
    subnet_meta=FLOW_META,
    subnet_constructor=MLP,
    hypercube_target=True,
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

# TODO: Add parts of Matrix-Element as prior
# if PRIOR:
#     # Define prior weight
#     prior = WeightPrior([map_1,map_2], N_CHANNELS)
#     madgraph_prior = prior.get_prior_weights
# else:
#     madgraph_prior = None

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
# TRAIN_SAMPLES = args.train_batches
# ITERS = TRAIN_SAMPLES // BATCH_SIZE
ITERS = args.train_batches

# Decay of learning rate
DECAY_RATE = 0.01
DECAY_STEP = ITERS

# Prepare scheduler and optimzer
lr_schedule1 = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)
lr_schedule2 = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)

opt1 = tf.keras.optimizers.Adam(lr_schedule1)
opt2 = tf.keras.optimizers.Adam(lr_schedule2)

# Add mappings to integrator
MAPPINGS = [map_y, map_Z]
N_MAPS = len(MAPPINGS)
for i in range(N_CHANNELS-N_MAPS):
    MAPPINGS.append(map_y)

integrator = MultiChannelIntegrator(
    integrand, flow, [opt1, opt2],
    mcw_model=mcw_net,
    mappings=MAPPINGS,
    use_weight_init=PRIOR,
    n_channels=N_CHANNELS,
    loss_func=LOSS
)

################################
# Pre train - integration
################################

res, err = integrator.integrate(INT_SAMPLES, weight_prior=madgraph_prior)
relerr = err / res * 100

res *=RES_TO_PB
err *=RES_TO_PB

print(f"\n Pre Multi-Channel integration ({INT_SAMPLES:.1e} samples):  ")
print("----------------------------------------------------------------")
print(f" Number of channels: {N_CHANNELS}                              ")
print(f" Result: {res:.8f} +- {err:.8f} pb ( Rel error: {relerr:.4f} %)")
print("----------------------------------------------------------------\n")

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

res *=RES_TO_PB
err *=RES_TO_PB

print(f"\n Opt. Multi-Channel integration ({INT_SAMPLES:.1e} samples): ")
print("----------------------------------------------------------------")
print(f" Number of channels: {N_CHANNELS}                              ")
print(f" Result: {res:.8f} +- {err:.8f} pb ( Rel error: {relerr:.4f} %)")
print("----------------------------------------------------------------\n")
