import os

from madnis.mappings.phasespace_2p import TwoParticlePhasespaceB
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import argparse
import time

from mcw import mcw_model, residual_mcw_model
from madnis.utils.train_utils import integrate
from madnis.utils.bayesian import BayesianHelper
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.distributions.camel import NormalizedMultiDimCamel
from madnis.nn.nets.mlp import MLP
from dy_integrand import DrellYan, MZ, WZ
from madnis.plotting.distributions import DistributionPlot
from madnis.plotting.plots import plot_weights
from madnis.distributions import StandardUniform

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

# Bayesian
parser.add_argument("--bayes_samples", type=int, default=25)

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
# Define the mcw network
################################

DATASET_SIZE = args.epochs * args.batch_size * args.train_batches
PRIOR = args.use_prior_weights

mcw_bayesian_helper = BayesianHelper(dataset_size=DATASET_SIZE)

MCW_META = {
    "units": args.mcw_units,
    "layers": args.mcw_layers,
    "initializer": args.initializer,
    "activation": args.activation,
    "layer_constructor": mcw_bayesian_helper.construct_dense
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

base_dist = StandardUniform(DIMS_IN)

# Prepare scheduler and optimzer
lr_schedule2 = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)

opt2 = tf.keras.optimizers.Adam(lr_schedule2)

# Add mappings to integrator
MAPPINGS = [map_y, map_Z]
N_MAPS = len(MAPPINGS)
for i in range(N_CHANNELS-N_MAPS):
    MAPPINGS.append(map_y)

integrator = MultiChannelIntegrator(
    integrand, base_dist, [opt2],
    mcw_model=mcw_net,
    mappings=MAPPINGS,
    use_weight_init=PRIOR,
    n_channels=N_CHANNELS,
    loss_func=LOSS,
    mcw_bayesian_helper=mcw_bayesian_helper
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

mcw_bayesian_helper.set_training(True)

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
                e + 1, train_losses[-1], opt2._decayed_lr(tf.float32)
            )
        )
end_time = time.time()
print("--- Run time: %s hour ---" % ((end_time - start_time) / 60 / 60))
print("--- Run time: %s mins ---" % ((end_time - start_time) / 60))
print("--- Run time: %s secs ---" % ((end_time - start_time)))

log_dir = f'./plots/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
#integrator.save_weights(log_dir)
#integrator.load_weights(log_dir + "model/")

mcw_bayesian_helper.set_training(False)

################################
# After train - plot sampling
################################

def to_four_mom(x):
    e_beam = 6500
    x1, x2, costheta, phi = tf.unstack(x, axis=1)
    s = 4 * e_beam**2 * x1 * x2
    r3 = (costheta + 1) / 2
    pz1 = e_beam * (x1*r3 + x2*(r3-1))
    pz2 = e_beam * (x1*(1-r3) - x2*r3)
    pt = tf.math.sqrt(s*r3*(1-r3))
    px1 = pt * tf.math.cos(phi)
    py1 = pt * tf.math.sin(phi)
    e1 = tf.math.sqrt(px1**2 + py1**2 + pz1**2)
    e2 = tf.math.sqrt(px1**2 + py1**2 + pz2**2)
    return tf.stack((e1, px1, py1, pz1, e2, -px1, -py1, pz2), axis=-1)

BAYES_SAMPLES = args.bayes_samples

#tf.config.run_functions_eagerly(True)
dist = DistributionPlot(log_dir, 'drell_yan', which_plots=[True, False, False, True])
channel_data = []
for i in range(N_CHANNELS):
    print(f'Sampling from channel {i}')
    ps, weights, alphas, alphas_prior = [], [], [], []
    for j in range(BAYES_SAMPLES):
        print(f'  Bayesian net {j}')
        mcw_bayesian_helper.sample_weights()
        x, weight, alpha, alpha_prior = integrator.sample_per_channel(
            10*INT_SAMPLES, i, weight_prior=madgraph_prior, return_alphas=True)
        p = to_four_mom(x).numpy()
        alpha_prior = None if alpha_prior is None else alpha_prior.numpy()
        ps.append(p)
        weights.append(weight.numpy())
        alphas.append(alpha.numpy())
        alphas_prior.append(alpha_prior)
    ps = np.stack(ps, axis=0)
    channel_data.append((
        ps,
        np.stack(weights, axis=0),
        np.stack(alphas, axis=0),
        np.stack(alphas_prior, axis=0) if alphas_prior[0] is not None else alphas_prior
    ))
    print(f'Plotting distributions for channel {i}')
    dist.plot(p, ps, f'after-channel-{i}')

print('Plotting channel weights')
dist.plot_channel_weights(channel_data, 'channel-weights')

print('Plotting weight distribution')
plot_weights(channel_data, log_dir, 'drell_yan_weight-dist')

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
