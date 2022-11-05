import os
from madnis.mappings.phasespace_2p import TwoParticlePhasespaceB

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import argparse
import time

from mcw import mcw_model, residual_mcw_model
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.distributions.uniform import StandardUniform
from madnis.nn.nets.mlp import MLP
from fudge_integrand import FudgeDrellYan, MZ, MZP, WZP
from madnis.plotting.distributions import DistributionPlot
from madnis.plotting.plots import plot_weights
from vegasflow import VegasFlow, RQSVegasFlow
from madnis.mappings.multi_flow import MultiFlow
from madnis.models.mc_prior import WeightPrior
from utils import to_four_mom

import sys

# Use double precision
tf.keras.backend.set_floatx("float64")

#########
# Setup #
#########

parser = argparse.ArgumentParser()

# Data params
parser.add_argument("--train_batches", type=int, default=1000)
parser.add_argument("--int_samples", type=int, default=1000000)

# model params
parser.add_argument("--use_prior_weights", action="store_true")
parser.add_argument("--units", type=int, default=16)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--blocks", type=int, default=6)
parser.add_argument("--activation", type=str, default="leakyrelu", choices={"relu", "elu", "leakyrelu", "tanh"})
parser.add_argument("--initializer", type=str, default="glorot_uniform", choices={"glorot_uniform", "he_uniform"})
parser.add_argument("--loss", type=str, default="variance", choices={"variance", "neyman_chi2", "exponential"})
parser.add_argument("--separate_flows", action="store_true")

# physics model-parameters
parser.add_argument("--z_width_scale", type=float, default=1)
parser.add_argument("--zp_width_scale", type=float, default=1)
parser.add_argument("--cut", type=float, default=15)

# mcw model params
parser.add_argument("--mcw_units", type=int, default=16)
parser.add_argument("--mcw_layers", type=int, default=2)

# prior and mapping setting
parser.add_argument("--prior", type=str, default="mg5", choices={"mg5", "sherpa", "flat"})
parser.add_argument("--maps", type=str, default="y", choices={"y", "z", "p", "zy", "py", "pz", "pzy"})

# Train params
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--train_mcw", action="store_true")
parser.add_argument("--fixed_mcw", dest="train_mcw", action="store_false")
parser.set_defaults(train_mcw=True)

# Plot
parser.add_argument("--pre_plotting", action="store_true")
parser.add_argument("--post_plotting", action="store_true")

args = parser.parse_args()


################################
# Define integrand
################################

DTYPE = tf.keras.backend.floatx()
DIMS_IN = 4  # dimensionality of data space
N_CHANNELS = len(args.maps)
INT_SAMPLES = args.int_samples
PLOT_SAMPLES = INT_SAMPLES
RES_TO_PB = 0.389379 * 1e9  # Conversion factor from GeV^-2 to pb
CUT = args.cut
MAPS = args.maps
PRIOR = args.prior

Z_SCALE = args.z_width_scale
WZ = 2.441404e-00 * Z_SCALE

ZP_SCALE = args.zp_width_scale
WZP = 5.000000e-01 * ZP_SCALE

if args.separate_flows:
    mode = "seperate"
else:
    mode = "cond"

if args.train_mcw:
    alphas = "trained"
else:
    alphas = "fixed"

LOG_DIR = f"./plots/zprime/{N_CHANNELS}channels_{MAPS}map_{mode}_{PRIOR}_{alphas}/"

# Define truth integrand
integrand = FudgeDrellYan(
    ["u", "d", "c", "s"],
    input_format="convpolar",
    wz=WZ,
    wzp=WZP,
    z_scale=Z_SCALE,
    zp_scale=ZP_SCALE,
)

print(f"\n Integrand specifications:")
print("-----------------------------------------------------------")
print(f" Dimensions : {DIMS_IN}                                   ")
print(f" Channels   : {N_CHANNELS}                                ")
print(f" Z-Width    : {WZ} GeV                                    ")
print(f" Z'-Width   : {WZP} GeV                                    ")
print("-----------------------------------------------------------\n")

# Define the channel mappings
map_p = TwoParticlePhasespaceB(s_mass=MZP, s_gamma=WZP, sqrt_s_min=CUT)
map_Z = TwoParticlePhasespaceB(s_mass=MZ, s_gamma=WZ, sqrt_s_min=CUT)
map_y = TwoParticlePhasespaceB(sqrt_s_min=CUT, nu=2)

################################
# Define the flow network
################################

FLOW_META = {
    "units": args.units,
    "layers": args.layers,
    "initializer": args.initializer,
    "activation": args.activation,
}

N_BLOCKS = args.blocks

if args.separate_flows:
    flow = MultiFlow(
        [
            RQSVegasFlow(
                [DIMS_IN],
                dims_c=None,
                n_blocks=N_BLOCKS,
                subnet_meta=FLOW_META,
                subnet_constructor=MLP,
                hypercube_target=True,
            )
            for i in range(N_CHANNELS)
        ]
    )
else:
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

mcw_net = residual_mcw_model(dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=MCW_META)

################################
# Define the prior
################################

def y_mg_prior(p: tf.Tensor):
    return integrand.single_channel(p, 0)

def z_mg_prior(p: tf.Tensor):
    return integrand.single_channel(p, 1)

def p_mg_prior(p: tf.Tensor):
    return integrand.single_channel(p, 2)

def y_map_prior(p: tf.Tensor):
    return map_y.prob(p)

def z_map_prior(p: tf.Tensor):
    return map_Z.prob(p)

def p_map_prior(p: tf.Tensor):
    return map_p.prob(p)

if PRIOR == "mg5":
    # Define prior weight
    if MAPS == "zy":
        prior = WeightPrior([z_mg_prior, y_mg_prior], N_CHANNELS)
        madgraph_prior = prior.get_prior_weights
    elif MAPS == "py":
        prior = WeightPrior([p_mg_prior, y_mg_prior], N_CHANNELS)
        madgraph_prior = prior.get_prior_weights
    elif MAPS == "pz":
        prior = WeightPrior([p_mg_prior, z_mg_prior], N_CHANNELS)
        madgraph_prior = prior.get_prior_weights
    elif MAPS == "pzy":
        prior = WeightPrior([p_mg_prior, z_mg_prior, y_mg_prior], N_CHANNELS)
        madgraph_prior = prior.get_prior_weights
    else:
        madgraph_prior = None
elif PRIOR == "sherpa":
    # Define prior weight
    if MAPS == "zy":
        prior = WeightPrior([z_map_prior, y_map_prior], N_CHANNELS)
        madgraph_prior = prior.get_prior_weights
    elif MAPS == "py":
        prior = WeightPrior([p_map_prior, y_map_prior], N_CHANNELS)
        madgraph_prior = prior.get_prior_weights
    elif MAPS == "pz":
        prior = WeightPrior([p_map_prior, z_map_prior], N_CHANNELS)
        madgraph_prior = prior.get_prior_weights
    elif MAPS == "pzy":
        prior = WeightPrior([p_map_prior, z_map_prior, z_map_prior], N_CHANNELS)
        madgraph_prior = prior.get_prior_weights
    else:
        madgraph_prior = None
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
TRAIN_MCW = args.train_mcw
UNIFORM_CHANNEL_RATIO = 1.0 #N_CHANNELS / 10

# Number of samples
# TRAIN_SAMPLES = args.train_batches
# ITERS = TRAIN_SAMPLES // BATCH_SIZE
ITERS = args.train_batches

# Decay of learning rate
DECAY_RATE = 0.01
DECAY_STEP = ITERS

# Prepare scheduler and optimzer
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)
opt = tf.keras.optimizers.Adam(lr_schedule)

# Add mappings to integrator
map_dict = {"y": map_y, "z": map_Z, "p": map_p}
MAPPINGS = []
for i in range(N_CHANNELS):
    MAPPINGS.append(map_dict[MAPS[i]])

base_dist = StandardUniform((DIMS_IN,))

if TRAIN_MCW:
    integrator = MultiChannelIntegrator(
        integrand,
        flow,
        opt,
        mcw_model=mcw_net,
        mappings=MAPPINGS,
        use_weight_init=PRIOR,
        n_channels=N_CHANNELS,
        uniform_channel_ratio=UNIFORM_CHANNEL_RATIO,
        loss_func=LOSS,
    )
else:
    integrator = MultiChannelIntegrator(
        integrand,
        flow,
        opt,
        mcw_model=None,
        mappings=MAPPINGS,
        use_weight_init=PRIOR,
        n_channels=N_CHANNELS,
        uniform_channel_ratio=UNIFORM_CHANNEL_RATIO,
        loss_func=LOSS,
    )

################################
# Pre train - plot sampling
################################

PLOTTING_PRE = args.pre_plotting

if PLOTTING_PRE:
    log_dir = LOG_DIR

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    dist = DistributionPlot(
        log_dir, "fudge_drell_yan", which_plots=[True, False, False, False]
    )
    channel_data = []
    for i in range(N_CHANNELS):
        print(f"Sampling from channel {i}")
        x, weight, alphas, alphas_prior = integrator.sample_per_channel(
            PLOT_SAMPLES, i, weight_prior=madgraph_prior, return_alphas=True
        )
        p = to_four_mom(x).numpy()
        alphas_prior = None if alphas_prior is None else alphas_prior.numpy()
        channel_data.append((p, weight.numpy(), alphas.numpy(), alphas_prior))

    weight_truth, events_truth = integrator.sample_weights(PLOT_SAMPLES, yield_samples=True, weight_prior=madgraph_prior)
    p_truth = to_four_mom(events_truth).numpy()
    true_data = (p_truth, weight_truth.numpy())

    print("Plotting channel weights")
    dist.plot_channels_stacked(channel_data, true_data, "pre_stacked")

    print("Plotting weight distribution")
    plot_weights(channel_data, log_dir, "pre_weight_dist")

################################
# Pre train - integration
################################

res, err = integrator.integrate(INT_SAMPLES, weight_prior=madgraph_prior)
relerr = err / res * 100

res *= RES_TO_PB
err *= RES_TO_PB

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
                e + 1, train_losses[-1], opt._decayed_lr(tf.float32)
            )
        )
end_time = time.time()
print("--- Run time: %s hour ---" % ((end_time - start_time) / 60 / 60))
print("--- Run time: %s mins ---" % ((end_time - start_time) / 60))
print("--- Run time: %s secs ---" % ((end_time - start_time)))

# integrator.save_weights(log_dir)
# integrator.load_weights(log_dir + "model/")

################################
# After train - plot sampling
################################

PLOTTING = args.post_plotting

if PLOTTING:
    log_dir = LOG_DIR

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    dist = DistributionPlot(
        log_dir, "fudge_drell_yan", which_plots=[True, False, False, False]
    )
    channel_data = []
    for i in range(N_CHANNELS):
        print(f"Sampling from channel {i}")
        x, weight, alphas, alphas_prior = integrator.sample_per_channel(
            PLOT_SAMPLES, i, weight_prior=madgraph_prior, return_alphas=True
        )
        p = to_four_mom(x).numpy()
        alphas_prior = None if alphas_prior is None else alphas_prior.numpy()
        channel_data.append((p, weight.numpy(), alphas.numpy(), alphas_prior))

    if not PLOTTING_PRE:
        weight_truth, events_truth = integrator.sample_weights(PLOT_SAMPLES, yield_samples=True)
        p_truth = to_four_mom(events_truth).numpy()
        true_data = (p_truth, weight_truth.numpy())

    print("Plotting channel weights")
    dist.plot_channels_stacked(channel_data, true_data, "post_stacked")

    print("Plotting weight distribution")
    plot_weights(channel_data, log_dir, "post_weight_dist")


################################
# After train - integration
################################

res, err = integrator.integrate(INT_SAMPLES, weight_prior=madgraph_prior)
relerr = err / res * 100

res *= RES_TO_PB
err *= RES_TO_PB

print(f"\n Opt. Multi-Channel integration ({INT_SAMPLES:.1e} samples): ")
print("----------------------------------------------------------------")
print(f" Number of channels: {N_CHANNELS}                              ")
print(f" Result: {res:.8f} +- {err:.8f} pb ( Rel error: {relerr:.4f} %)")
print("----------------------------------------------------------------\n")

########################################
# After train - unweighting efficiency
########################################

n_opt = INT_SAMPLES // 100
uwgt_eff, uwgt_eff_part, over_weights = integrator.acceptance(n_opt)
over_weights *= 100
uwgt_eff_part *= 100
uwgt_eff *= 100

print(f"\n Unweighting efficiency using ({n_opt:.1e} samples): ")
print("----------------------------------------------------------------")
print(f" Number of channels: {N_CHANNELS}                              ")
print(f" Efficiency 1 : {uwgt_eff:.8f} %                               ")
print(f" Efficiency 2 : {uwgt_eff_part:.8f} %                          ")
print(f" Over_weights : {over_weights:.8f} %                           ")
print("----------------------------------------------------------------\n")
