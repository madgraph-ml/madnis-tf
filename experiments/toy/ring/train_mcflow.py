import os
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import pickle

from madnis.models.mcw import mcw_model, residual_mcw_model
from madnis.utils.train_utils import integrate

from madnis.distributions.gaussians_2d import TwoChannelLineRing
from madnis.mappings.cauchy_2d import CauchyRingMap, CauchyLineMap
from madnis.mappings.unit_hypercube import RealsToUnit
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.models.mc_prior import WeightPrior
from madnis.nn.nets.mlp import MLP
from madnis.models.vegasflow import AffineVegasFlow
from madnis.mappings.multi_flow import MultiFlow
from madnis.plotting.distributions import DistributionPlot

# Use double precision
tf.keras.backend.set_floatx("float64")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#tf.config.run_functions_eagerly(True)

pickle_data = {}

#########
# Setup #
#########

parser = argparse.ArgumentParser()

# Data params
parser.add_argument("--train_batches", type=int, default=100)
parser.add_argument("--int_samples", type=int, default=10000)

# prior and mapping params
parser.add_argument("--maps", type=str, default="ff")
parser.add_argument("--prior", type=str, default="flat",
                    choices={"mg5", "sherpa", "flat"})

# model params
parser.add_argument("--use_prior_weights", action='store_true')
parser.add_argument("--units", type=int, default=16)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--blocks", type=int, default=6)
parser.add_argument("--activation", type=str, default="leakyrelu", choices={"relu", "elu", "leakyrelu", "tanh"})
parser.add_argument("--initializer", type=str, default="glorot_uniform", choices={"glorot_uniform", "he_uniform"})
parser.add_argument("--loss", type=str, default="variance", choices={"variance", "neyman_chi2", "kl_divergence"})
parser.add_argument("--separate_flows", action="store_true")

# mcw model params
parser.add_argument("--mcw_units", type=int, default=16)
parser.add_argument("--mcw_layers", type=int, default=2)

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

res, err = integrate(integrand)
relerr = err / res * 100

print(f"\n Naive integration ({INT_SAMPLES:.1e} samples):")
print("-------------------------------------------------------------")
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)")
print("-----------------------------------------------------------\n")


################################
# Define the flow network
################################

N_CHANNELS = len(args.maps)  # number of channels

FLOW_META = {
    "units": args.units,
    "layers": args.layers,
    "initializer": args.initializer,
    "activation": args.activation,
}

N_BLOCKS = args.blocks

make_flow = lambda dims_c: AffineVegasFlow(
    [DIMS_IN],
    dims_c=dims_c,
    n_blocks=N_BLOCKS,
    subnet_meta=FLOW_META,
    subnet_constructor=MLP,
    hypercube_target=True,
)

if args.separate_flows:
    flow = MultiFlow([make_flow(None) for i in range(N_CHANNELS)])
else:
    flow = make_flow([[N_CHANNELS]])

################################
# Define the mappings
################################

GAMMA0 = np.sqrt(40.) * SIGMA0
GAMMA1 = np.sqrt(40.) * SIGMA1
GAMMA2 = np.sqrt(40.) * SIGMA2

map_r = CauchyRingMap(RADIUS, GAMMA0)
map_l = CauchyLineMap([MEAN1, MEAN2], [GAMMA1, GAMMA2], ALPHA)
map_f = RealsToUnit((2,))

map_dict = {"r": map_r, "l": map_l, "f": map_f}
mappings = [map_dict[m] for m in args.maps]

################################
# Define the prior
################################

if args.prior == "mg5" and args.maps == "rl":
    r_mg_prior = line_ring.ring.prob
    l_mg_prior = line_ring.line.prob
    prior = WeightPrior([r_mg_prior, l_mg_prior], N_CHANNELS)
elif args.prior == "sherpa":
    prior = WeightPrior([m.prob for m in mappings], N_CHANNELS)
else:
    prior = None

madgraph_prior = None if prior is None else prior.get_prior_weights

################################
# Define the mcw network
################################

MCW_META = {
    "units": args.mcw_units,
    "layers": args.mcw_layers,
    "initializer": args.initializer,
    "activation": args.activation,
}

if madgraph_prior is not None:
    mcw_net = residual_mcw_model(
        dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=MCW_META
    )
else:
    mcw_net = mcw_model(dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=MCW_META)

################################
# Define the integrator
################################

# Define training params
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
LOSS = args.loss

# Number of samples
ITERS = args.train_batches

# Decay of learning rate
DECAY_RATE = 0.01
DECAY_STEP = ITERS

# Prepare scheduler and optimzer
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)

opt = tf.keras.optimizers.Adam(lr_schedule)

integrator = MultiChannelIntegrator(
    line_ring,
    flow,
    opt,
    mappings=mappings,
    mcw_model=mcw_net,
    use_weight_init=madgraph_prior is not None,
    n_channels=N_CHANNELS,
    loss_func=LOSS
)

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
                e + 1, train_losses[-1], opt._decayed_lr(tf.float32)
            )
        )
end_time = time.time()
print("--- Run time: %s hour ---" % ((end_time - start_time) / 60 / 60))
print("--- Run time: %s mins ---" % ((end_time - start_time) / 60))
print("--- Run time: %s secs ---" % ((end_time - start_time)))

################################
# After train - plot sampling
################################

log_dir = f'./plots/map_{args.maps}_prior_{args.prior}{"_sep" if args.separate_flows else ""}'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

dist = DistributionPlot(log_dir, "ring", which_plots=[0,0,0,1])
for i in range(N_CHANNELS):
    x0, weight0 = integrator.sample_per_channel(10*INT_SAMPLES, i, weight_prior=madgraph_prior)
    dist.plot(x0, x0, f'post-channel-{i}')

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
