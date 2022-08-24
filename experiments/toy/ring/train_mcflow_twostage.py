import os
import tensorflow as tf
import numpy as np
import argparse
import time
import sys

from mcw import mcw_model, residual_mcw_model
from utils import integrate, error

from madnis.distributions.gaussians_2d import TwoChannelLineRing
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.nn.nets.mlp import MLP
from vegasflow import VegasFlow

# Use double precision
tf.keras.backend.set_floatx("float64")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


#########
# Setup #
#########

parser = argparse.ArgumentParser()

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
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=5e-4)

args = parser.parse_args()

################################
# Define distributions
################################
DTYPE = tf.keras.backend.floatx()

# Define peak positions and heights
# Ring
RADIUS = 1.0
SIGMA0 = 0.01

# Line
MEAN1 = 0.0
MEAN2 = 0.0
SIGMA1 = 3.0
SIGMA2 = 0.01
ALPHA = np.pi/4

# Define truth distribution
line_ring = TwoChannelLineRing(RADIUS, SIGMA0, [MEAN1, MEAN2], [SIGMA1, SIGMA2], ALPHA)

################################
# Naive integration
################################

INT_SAMPLES = args.int_samples
DIMS_IN = 2  # dimensionality of data space

# Uniform sampling in area [-6,6]x[-6,6]
limits = [-6, 6]
volume = (limits[1] - limits[0]) ** 2
noise = limits[0] + (limits[1] - limits[0]) * tf.random.uniform((INT_SAMPLES, DIMS_IN), dtype=DTYPE)
phi = 1 / volume
integrand = line_ring.prob(noise) / phi  # divide by density which is 1/V

res = integrate(integrand).numpy()
err = error(integrand).numpy()
relerr = err / res * 100

print(f"\n Naive integration ({INT_SAMPLES:.1e} samples):")
print("-------------------------------------------------------------")
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)")
print("-----------------------------------------------------------\n")


################################
# Define the flow network
################################

N_CHANNELS = args.channels  # number of channels
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
    hypercube_target=False,
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
    line_ring, flow, [opt1, opt2], mcw_model=mcw_net, use_weight_init=PRIOR,
    n_channels=N_CHANNELS, loss_func=LOSS, sample_capacity=2500)

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
            "Epoch #{}.1: Loss: {}, Learning_Rate: {}".format(
                e + 1, train_losses[-1], opt1._decayed_lr(tf.float32)
            )
        )

    train_loss = integrator.train_on_stored_samples(1024, weight_prior=madgraph_prior)

    if (e + 1) % 1 == 0:
        # Print metrics
        print(
            "Epoch #{}.2: Loss: {}, Learning_Rate: {}".format(
                e + 1, train_loss, opt1._decayed_lr(tf.float32)
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