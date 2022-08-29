import os
import tensorflow as tf
import numpy as np
import argparse
import time
import sys

from mcw import mcw_model, residual_mcw_model
from madnis.utils.train_utils import integrate
from madnis.plotting.plots import plot_alphas

from madnis.distributions.camel import Camel
from madnis.distributions.uniform import StandardUniform
from madnis.mappings.cauchy import CauchyDistribution
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.models.mc_prior import WeightPrior

# Use double precision
tf.keras.backend.set_floatx("float64")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


#########
# Setup #
#########

parser = argparse.ArgumentParser()

# Data params
parser.add_argument("--train_batches", type=int, default=1000)
parser.add_argument("--int_samples", type=int, default=10000)

# Model params
parser.add_argument("--use_prior_weights", action='store_true')
parser.add_argument("--units", type=int, default=16)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--activation", type=str, default="leakyrelu", choices={"relu", "elu", "leakyrelu", "tanh"})
parser.add_argument("--initializer", type=str, default="glorot_uniform", choices={"glorot_uniform", "he_uniform"})

# Train params
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--sample_capacity", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--loss", type=str, default="variance", choices={"variance", "neyman_chi2", "kl_divergence"})
parser.add_argument("--uniform_channel_ratio", type=float, default=1.0)
parser.add_argument("--variance_history_length", type=int, default=100)

args = parser.parse_args()

################################
# Define distributions
################################

# Define peak positions and heights
MEAN1 = 2.0
STD1 = 0.5
MEAN2 = 5.0
STD2 = 0.2

# Define truth distribution
camel = Camel([MEAN1, MEAN2], [STD1, STD2], peak_ratios=[0.5, 0.5])

# Define the channel mappings
GAMMA1 = np.sqrt(2.) * STD1
GAMMA2 = np.sqrt(2.) * STD2

map_1 = CauchyDistribution(mean=MEAN1, gamma=GAMMA1)
map_2 = CauchyDistribution(mean=MEAN2, gamma=GAMMA2)

################################
# Naive integration
################################

INT_SAMPLES = args.int_samples

# Uniform sampling in range [0,6]
volume = 6
noise = tf.random.uniform((INT_SAMPLES, 1), dtype=tf.keras.backend.floatx()) * volume
phi = 1 / volume
integrand = camel.prob(noise) / phi  # divide by volume in this case

res, err = integrate(integrand)
relerr = err / res * 100

print(f"\n Naive integration ({INT_SAMPLES:.1e} samples):")
print("-------------------------------------------------------------")
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)")
print("-----------------------------------------------------------\n")


################################
# Define the network
################################

DIMS_IN = 1  # dimensionality of data space (Camel = 1D)
N_CHANNELS = 2  # number of Mappings
PRIOR = args.use_prior_weights

META = {
    "units": args.units,
    "layers": args.layers,
    "initializer": args.initializer,
    "activation": args.activation,
}

if PRIOR:
    mcw_net = residual_mcw_model(
        dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=META
    )
    PREFIX = "residual"
else:
    mcw_net = mcw_model(dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=META)
    PREFIX = "scratch"


################################
# Define the prior
################################

# Define prior functions
# LOGST1 = tf.math.log(STD1) * tf.ones((1,1))
# LOGST2 = tf.math.log(STD2) * tf.ones((1,1))
# f1 = Normal((1,), mean=MEAN1, log_std=LOGST1)
# f2 = Normal((1,), mean=MEAN2, log_std=LOGST2)

if PRIOR:
    # Define prior weight 
    prior = WeightPrior([map_1,map_2], N_CHANNELS)
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
LOSS = args.loss
SAMPLE_CAPACITY = args.sample_capacity
UNIFORM_CHANNEL_RATIO = args.uniform_channel_ratio
VARIANCE_HISTORY_LENGTH = args.variance_history_length

# Number of samples
#TRAIN_SAMPLES = args.train_samples
ITERS = args.train_batches

# Decay of learning rate
DECAY_RATE = 0.01
DECAY_STEP = ITERS

# Prepare scheduler and optimzer
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)
opt = tf.keras.optimizers.Adam(lr_schedule)

base_dist = StandardUniform((DIMS_IN,))

integrator = MultiChannelIntegrator(
    camel,
    base_dist,
    [opt],
    mappings=[map_1, map_2],
    mcw_model=mcw_net,
    use_weight_init=PRIOR,
    n_channels=N_CHANNELS,
    loss_func=LOSS,
    sample_capacity=SAMPLE_CAPACITY,
    uniform_channel_ratio=UNIFORM_CHANNEL_RATIO,
    variance_history_length=VARIANCE_HISTORY_LENGTH
)


################################
# Pre train - plot alphas
################################

p = tf.cast(tf.linspace([0], [6], 1000, axis=0), tf.keras.backend.floatx())
if PRIOR:
    res = madgraph_prior(p)
    alphas = mcw_net([p, res])
else:
    alphas = mcw_net(p)
truth = camel.prob(p)
m1 = map_1.prob(p)
m2 = map_2.prob(p)
plot_alphas(p, alphas, truth, [m1, m2], prefix=f"pre_{PREFIX}")

################################
# Pre train - integration
################################

# Uniform sampling in range [0,1]
# integrand = integrator._get_integrand(INT_SAMPLES, weight_prior=madgraph_prior)
# print(integrand)
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
# After train - plot alphas
################################

p = tf.cast(tf.linspace([0], [6], 1000, axis=0), tf.keras.backend.floatx())
if PRIOR:
    res = madgraph_prior(p)
    alphas = mcw_net([p, res])
else:
    alphas = mcw_net(p)
truth = camel.prob(p)
m1 = map_1.prob(p)
m2 = map_2.prob(p)
plot_alphas(p, alphas, truth, [m1, m2], prefix=f"after_{PREFIX}")


################################
# After train - integration
################################

res, err = integrator.integrate(INT_SAMPLES, weight_prior=madgraph_prior)
relerr = err / res * 100

print(f"\n Opt. Multi-Channel integration ({INT_SAMPLES:.1e} samples):")
print("---------------------------------------------------------------")
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)  ")
print("-------------------------------------------------------------\n")
