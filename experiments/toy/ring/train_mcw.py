import os
import tensorflow as tf
import numpy as np
import argparse
import time

from mcw import mcw_model, residual_mcw_model
from utils import integrate, error

# from plots import plot_alphas
from madnis.distributions.gaussians_2d import TwoChannelLineRing
from madnis.mappings.cauchy_2d import CauchyRingMap, CauchyLineMap
from madnis.models.multi_channel import MultiChannelWeight
from madnis.models.mc_prior import WeightPrior
import sys

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

# Model params
parser.add_argument("--use_prior_weights", action='store_true')
parser.add_argument("--units", type=int, default=32)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--activation", type=str, default="leakyrelu", choices={"relu", "elu", "leakyrelu", "tanh"})
parser.add_argument("--initializer", type=str, default="glorot_uniform", choices={"glorot_uniform", "he_uniform"})
parser.add_argument("--loss", type=str, default="variance", choices={"variance", "neyman_chi2", "kl_divergence"})

# Train params
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)

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

# Define the channel mappings
GAMMA0 = np.sqrt(2.) * SIGMA0
GAMMA1 = np.sqrt(2.) * SIGMA1
GAMMA2 = np.sqrt(2.) * SIGMA2

map_1 = CauchyRingMap(RADIUS, GAMMA0)
map_2 = CauchyLineMap([MEAN1, MEAN2], [GAMMA1, GAMMA2], ALPHA)

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
# Define the network
################################

N_CHANNELS = 2  # number of Mappings
PRIOR = args.use_prior_weights

META = {
    "units": args.units,
    "layers": args.layers,
    "initializer": args.initializer,
    "activation": args.activation,
}

if PRIOR:
    mcw_model = residual_mcw_model(
        dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=META
    )
    PREFIX = "residual"
else:
    mcw_model = mcw_model(dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=META)
    PREFIX = "scratch"
    
################################
# Define the prior
################################

if PRIOR:
    # Define prior weight 
    prior = WeightPrior([map_1,map_1], N_CHANNELS)
    madgraph_prior = prior.get_prior_weights
else:
    madgraph_prior = None
    
################################
# Define the integrator
################################

# Define training params
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
if PRIOR:
    LR /= 5
    EPOCHS /=2
LOSS = args.loss

# Number of samples
TRAIN_SAMPLES = args.train_samples
ITERS = TRAIN_SAMPLES // BATCH_SIZE

# Decay of learning rate
DECAY_RATE = 0.01
DECAY_STEP = ITERS

# Prepare scheduler and optimzer
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)
opt = tf.keras.optimizers.Adam(lr_schedule)

integrator = MultiChannelWeight(
    line_ring, mcw_model, [map_1, map_2], opt, use_weight_init=PRIOR, loss_func=LOSS)

################################
# Pre train - integration
################################

res, err = integrator.integrate(INT_SAMPLES, weight_prior=madgraph_prior)
relerr = err / res * 100

print(f"\n Pre Multi-Channel integration ({INT_SAMPLES:.1e} samples):")
print("--------------------------------------------------------------")
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
                e + 1, train_losses[-1], opt._decayed_lr(tf.float32)
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
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)  ")
print("-------------------------------------------------------------\n")
