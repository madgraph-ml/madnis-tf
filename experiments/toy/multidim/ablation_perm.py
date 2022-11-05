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
import matplotlib.pyplot as plt

from madnis.utils.train_utils import integrate
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.distributions.camel import NormalizedMultiDimCamel
from madnis.nn.nets.mlp import MLP
import madnis
from madnis.models.vegasflow import AffineVegasFlow, RQSVegasFlow


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
parser.add_argument("--use_RQS", action='store_true', help='use RQS instead of affine trafo')
parser.add_argument("--RQS_bins", type=int, default=8)
parser.add_argument("--units", type=int, default=128)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--blocks", type=int, default=6)
parser.add_argument("--activation", type=str, default="leakyrelu",
                    choices={"relu", "elu", "leakyrelu", "tanh"})
parser.add_argument("--initializer", type=str, default="glorot_uniform",
                    choices={"glorot_uniform", "he_uniform"})
parser.add_argument("--loss", type=str, default="variance",
                    choices={"variance", "neyman_chi2", "kl_divergence"})

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

flows_dic = {}
if args.use_RQS:
    FLOW_CONST = RQSVegasFlow
    kwargs = {'bins': args.RQS_bins}
else:
    FLOW_CONST = AffineVegasFlow
    kwargs = {}

for perm in ['exchange', 'random', 'log', 'soft', 'softlearn']:
    flows_dic[perm] = FLOW_CONST(
        [DIMS_IN],
        dims_c=[[N_CHANNELS]],
        n_blocks=N_BLOCKS,
        subnet_meta=FLOW_META,
        subnet_constructor=MLP,
        hypercube_target=True,
        permutations=perm,
        **kwargs
    )

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
# TRAIN_SAMPLES = args.train_batches
# ITERS = TRAIN_SAMPLES // BATCH_SIZE
ITERS = args.train_batches

# Decay of learning rate
DECAY_RATE = 0.01
DECAY_STEP = ITERS

# Prepare scheduler, optimzer, and integrator
lr_schedule_dic = {}
opt_dic = {}
integrator_dic = {}

################################
# Define the training
################################

def train_flow(integrator, opt, num_epochs):
    """ trains a given flow """
    train_losses = []
    start_time = time.time()
    for e in range(num_epochs):

        batch_train_losses = []
        # do multiple iterations.
        for _ in range(ITERS):
            batch_loss = integrator.train_one_step(BATCH_SIZE, weight_prior=madgraph_prior)
            batch_train_losses.append(batch_loss)

            #for idx, trafo in enumerate(integrator.dist.transforms):
            #    print("inside train", idx, trafo)
            #    if isinstance(trafo, (madnis.transforms.permutation.PermuteSoftLearn,
            #                          madnis.transforms.permutation.Permutation)):
            #        print(trafo.w_perm)


        train_loss = tf.reduce_mean(batch_train_losses)
        train_losses.append(train_loss)

        if (e + 1) % 1 == 0:
            # Print metrics
            print(
                "Epoch #{}/{}: Loss: {}, Learning_Rate: {}".format(
                    e + 1, num_epochs, train_losses[-1], opt._decayed_lr(tf.float32)
                )
            )

    end_time = time.time()
    print("--- Run time: %s hour ---" % ((end_time - start_time) / 60 / 60))
    print("--- Run time: %s mins ---" % ((end_time - start_time) / 60))
    print("--- Run time: %s secs ---" % ((end_time - start_time)))
    return train_losses

################################
# Define the integration
################################

def integrate_with_flow(integrator, num_samples, weight_prior=None):
    """ integrates based on the flow sampling """
    res, err = integrator.integrate(num_samples, weight_prior=weight_prior)
    relerr = err / res * 100
    return res, err, relerr

################################
# Pre train - integration
################################
loss_dic = {}
#tf.config.run_functions_eagerly(True)

plt.figure()

#for perm in ['softlearn']:
for perm in ['exchange', 'random', 'log', 'soft', 'softlearn']:
    print("Optimizing flow for permutation {} now ... ".format(perm))
    lr_schedule_dic[perm] = tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP,
                                                                           DECAY_RATE)
    opt_dic[perm] = tf.keras.optimizers.Adam(lr_schedule_dic[perm])
    integrator_dic[perm] = MultiChannelIntegrator(
        multi_camel, flows_dic[perm], [opt_dic[perm]], use_weight_init=False,
        n_channels=N_CHANNELS, loss_func=LOSS)

    res, err, relerr = integrate_with_flow(integrator_dic[perm], INT_SAMPLES,
                                           weight_prior=madgraph_prior)
    print(f"\n Pre Multi-Channel integration ({INT_SAMPLES:.1e} samples):")
    print(f"-----------------Choice of Permutation: {perm}----------------")
    print("--------------------------------------------------------------")
    print(f" Number of channels: {N_CHANNELS}                            ")
    print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %) ")
    print("------------------------------------------------------------\n")

    # show perms:
    for trafo in flows_dic[perm].transforms:
        #print("before ", trafo)
        #if isinstance(trafo, (madnis.transforms.permutation.PermuteSoftLearn,
        #                      madnis.transforms.permutation.Permutation)):
        #    print(trafo.w_perm)
        if isinstance(trafo, (madnis.transforms.permutation.PermuteSoftLearn)):
            print("before: ", trafo.trainable_variables)
    # train
    loss_dic[perm] = train_flow(integrator_dic[perm], opt_dic[perm], EPOCHS)
    plt.plot(loss_dic[perm], label='permutation: {}'.format(perm))
    # show perms:
    for trafo in flows_dic[perm].transforms:
        #print("after ", trafo)
        #if isinstance(trafo, (madnis.transforms.permutation.PermuteSoftLearn,
        #                      madnis.transforms.permutation.Permutation)):
        #   print(trafo.w_perm)
        if isinstance(trafo, (madnis.transforms.permutation.PermuteSoftLearn)):
            print("after: ", trafo.trainable_variables)

    # after train integration
    res, err, relerr = integrate_with_flow(integrator_dic[perm], INT_SAMPLES,
                                           weight_prior=madgraph_prior)
    print(f"\n Opt. Multi-Channel integration ({INT_SAMPLES:.1e} samples):")
    print(f"-----------------Choice of Permutation: {perm}----------------")
    print("---------------------------------------------------------------")
    print(f" Number of channels: {N_CHANNELS}                             ")
    print(f" Result: {res:.8f} +- {err:.8f} ( Rel error: {relerr:.4f} %)  ")
    print("-------------------------------------------------------------\n")

plt.legend()
plt.ylabel("NLL loss")
plt.xlabel("epoch")
plt.yscale('log')
plt.ylim([1e-3, 1e2])
plt.show()
# to-do: plot corner of learned dist.
