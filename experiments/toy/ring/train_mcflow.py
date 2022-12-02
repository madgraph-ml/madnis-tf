import os
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import pickle
import matplotlib.pyplot as plt

from madnis.models.mcw import mcw_model, residual_mcw_model
from madnis.utils.train_utils import integrate

from madnis.distributions.gaussians_2d import TwoChannelLineRing, GaussianLine, GaussianRing
from madnis.mappings.cauchy_2d import CauchyRingMap, CauchyLineMap
from madnis.mappings.unit_hypercube import RealsToUnit
from madnis.mappings.identity import Identity
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.models.mc_prior import WeightPrior
from madnis.nn.nets.mlp import MLP
from madnis.models.vegasflow import AffineVegasFlow, RQSVegasFlow
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
parser.add_argument("--train_batches", type=int, default=500)
parser.add_argument("--int_samples", type=int, default=10000)

# prior and mapping params
parser.add_argument("--maps", type=str, default="ff")
parser.add_argument("--prior", type=str, default="flat",
                    choices={"mg5", "sherpa", "flat"})

# model params
parser.add_argument("--couplings", type=str, default="affine",
                    choices={"affine", "rqs"})
parser.add_argument("--units", type=int, default=16)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--blocks", type=int, default=6)
parser.add_argument("--activation", type=str, default="leakyrelu",
                    choices={"relu", "elu", "leakyrelu", "tanh"})
parser.add_argument("--permutations", type=str, default="soft")
parser.add_argument("--initializer", type=str, default="glorot_uniform",
                    choices={"glorot_uniform", "he_uniform"})
parser.add_argument("--loss", type=str, default="variance",
                    choices={"variance", "neyman_chi2", "kl_divergence"})
parser.add_argument("--separate_flows", action="store_true")

# mcw model params
parser.add_argument("--mcw_units", type=int, default=16)
parser.add_argument("--mcw_layers", type=int, default=2)

# Train params
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_decay", type=float, default=0.02)

parser.add_argument("--run_name", type=str)

args = parser.parse_args()

################################
# Define distributions
################################
DTYPE = tf.keras.backend.floatx()

# Define peak positions and heights
# Ring
RADIUS = 1.0
SIGMA0 = 0.05

# Line
MEAN1 = 0.0
MEAN2 = 0.0
SIGMA1 = 3.0
SIGMA2 = 0.05
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

pickle_data.update({
    "naive_integral": float(res),
    "naive_integral_err": float(err),
    "naive_integral_relerr": float(relerr),
})

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

CouplingBlock = {"affine": AffineVegasFlow, "rqs": RQSVegasFlow}[args.couplings]
make_flow = lambda dims_c, use_hyper: CouplingBlock(
    [DIMS_IN],
    dims_c=dims_c,
    n_blocks=N_BLOCKS,
    subnet_meta=FLOW_META,
    subnet_constructor=MLP,
    hypercube_target=use_hyper,
    clamp=0.5,
    permutations=args.permutations,
)

if args.separate_flows:
    hyper_bool = lambda map: False if map == "f" else True
    flow = MultiFlow([make_flow(None, hyper_bool(map)) for map in args.maps])
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

if args.couplings == "affine" and args.separate_flows:
    map_f = Identity((2,))
else:
    map_f = RealsToUnit((2,))

map_dict = {"r": map_r, "l": map_l, "f": map_f}
mappings = [map_dict[m] for m in args.maps]

################################
# Define the prior
################################

# Maps for the MG prior (possibly change its widths if wanted)
prior_r = CauchyRingMap(RADIUS, np.sqrt(2.) * SIGMA0)
prior_l = CauchyLineMap([MEAN1, MEAN2], [np.sqrt(2.)*SIGMA1, np.sqrt(2.)*SIGMA2], ALPHA)
prior_l2 = CauchyLineMap([-1, -1], [GAMMA1, GAMMA2], ALPHA)
prior_l3 = CauchyLineMap([1, 1], [GAMMA1, GAMMA2], ALPHA)

if args.prior == "mg5" and len(args.maps) == 2:
    # r_mg_prior = line_ring.ring.prob
    # l_mg_prior = line_ring.line.prob
    # # prior = WeightPrior([r_mg_prior, l_mg_prior], N_CHANNELS)
    prior = WeightPrior([prior_r.prob, prior_l.prob], N_CHANNELS)
elif args.prior == "mg5" and len(args.maps) == 3:
    prior = WeightPrior([prior_r.prob, prior_r.prob, prior_l.prob], N_CHANNELS)
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

#if madgraph_prior is not None:
mcw_net = residual_mcw_model(
    dims_in=DIMS_IN, n_channels=N_CHANNELS, meta=MCW_META
)

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
DECAY_RATE = args.lr_decay
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
    use_weight_init=True, #madgraph_prior is not None,
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

pickle_data.update({
    "pre_integral": float(res),
    "pre_integral_err": float(err),
    "pre_integral_relerr": float(relerr),
})

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
                e + 1, train_losses[-1],
                opt._decayed_lr(tf.float32) if hasattr(opt,"_decayed_lr") else "?"
            )
        )
train_time = time.time() - start_time
print("--- Run time: %s hour ---" % (train_time / 60 / 60))
print("--- Run time: %s mins ---" % (train_time / 60))
print("--- Run time: %s secs ---" % (train_time))

pickle_data["train_losses"] = np.array(train_losses)
pickle_data["train_time"] = train_time

################################
# After train - integration
################################

res, err, chan_means, chan_errs = integrator.integrate(
        INT_SAMPLES, weight_prior=madgraph_prior, return_channels=True)
relerr = err / res * 100

print(f"\n Opt. Multi-Channel integration ({INT_SAMPLES:.1e} samples):")
print("---------------------------------------------------------------")
print(f" Number of channels: {N_CHANNELS}                             ")
print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)  ")
print("-------------------------------------------------------------\n")

chan_means = np.array(chan_means)
chan_errs = np.array(chan_errs)
pickle_data.update({
    "post_integral": float(res),
    "post_integral_err": float(err),
    "post_integral_relerr": float(relerr),
    "post_integral_chan_means": chan_means,
    "post_integral_chan_errs": chan_errs,
})

################################
# After train - plot sampling
################################

if args.run_name is None:
    sep_str = "_sep" if args.separate_flows else ""
    rqs_str = "_rqs" if args.couplings == "rqs" else ""
    log_dir = f'./plots/map_{args.maps}_prior_{args.prior}{sep_str}{rqs_str}'
else:
    log_dir = f'./plots/{args.run_name}'

def hist_all(d, chan_weights):
    pcd = d["post_channel_data"]
    x = np.concatenate([c[0] for c in pcd], axis=0)
    alpha = np.concatenate([w*c[2] for w, c in zip(chan_weights, pcd)], axis=0)
    
    bins = np.linspace(-2,2,100)
    plt.figure(figsize=(5,4))
    plt.title("All channels")
    plt.hist2d(x[:,0], x[:,1], weights=alpha, bins=bins, density=True, rasterized=True)
    plt.colorbar()
    plt.savefig(os.path.join(log_dir, "all_channels.pdf"))
    plt.close()

def alphas_2c(d):
    x = d["post_x"]
    y = d["post_y"]
    xx, yy = np.meshgrid(x, y)
    alphas = d["post_alphas"].reshape(len(x), len(y), 2)
    plt.figure(figsize=(5,4))
    plt.title(r"$\alpha = 1$: channel 0, $\alpha = 0$: channel 1")
    plt.pcolormesh(xx, yy, alphas[:,:,0], vmin=0., vmax=1., rasterized=True)
    plt.colorbar()
    plt.savefig(os.path.join(log_dir, "alphas.pdf"))
    plt.close()

def alphas_3c(d):
    x = d["post_x"]
    y = d["post_y"]
    xx, yy = np.meshgrid(x, y)
    alphas = d["post_alphas"].reshape(len(x), len(y), 3)
    plt.figure(figsize=(4,4))
    plt.title(r"red: chan 0, green: chan 1, blue: chan 2")
    plt.imshow(alphas, origin="lower", extent=(-2,2,-2,2), rasterized=True)
    plt.savefig(os.path.join(log_dir, "alphas.pdf"))
    plt.close()

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

dist = DistributionPlot(log_dir, "ring", which_plots=[0,0,0,1])
channel_data = []
for i in range(N_CHANNELS):
    x, weight, alphas, alphas_prior = integrator.sample_per_channel(
        10*INT_SAMPLES, i, weight_prior=madgraph_prior, return_alphas=True
    )
    dist.plot(x, x, f'post-channel-{i}')
    alphas_prior = None if alphas_prior is None else alphas_prior.numpy()
    channel_data.append((x.numpy(), weight.numpy(), alphas.numpy(), alphas_prior))
pickle_data["post_channel_data"] = channel_data

grid = tf.cast(tf.linspace(-2, 2, 101), tf.keras.backend.floatx())
xx = tf.reshape(tf.stack(tf.meshgrid(grid, grid), axis=-1), (-1,2))
if madgraph_prior is not None:
    res = madgraph_prior(xx)
else:
    res = 1. / N_CHANNELS * tf.ones(
        (xx.shape[0], N_CHANNELS), dtype=tf.keras.backend.floatx()
    )
    #alphas = mcw_net(xx)
alphas = mcw_net([xx, res])
pickle_data["post_x"] = grid.numpy()
pickle_data["post_y"] = grid.numpy()
pickle_data["post_alphas"] = alphas.numpy()

hist_all(pickle_data, chan_means / np.sum(chan_means))
if N_CHANNELS == 2:
    alphas_2c(pickle_data)
elif N_CHANNELS == 3:
    alphas_3c(pickle_data)

with open(os.path.join(log_dir, "results.pkl"), "wb") as f:
    pickle.dump(pickle_data, f)
