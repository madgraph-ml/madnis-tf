import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import pickle

from madnis.models.mcw import mcw_model, residual_mcw_model, additive_residual_mcw_model
from madnis.models.mc_integrator import MultiChannelIntegrator
from madnis.distributions.uniform import StandardUniform
from madnis.nn.nets.mlp import MLP
from madnis.plotting.distributions import DistributionPlot
from madnis.plotting.plots import plot_weights, plot_variances
from madnis.models.vegasflow import AffineVegasFlow, RQSVegasFlow
from madnis.mappings.multi_flow import MultiFlow
from utils import to_four_mom
from madnis.utils.train_utils import parse_schedule

class MadnisTraining:
    def tf_setup(self):
        tf.keras.backend.set_floatx("float64")

    def parse_arguments(self):
        parser = argparse.ArgumentParser()

        # Data params
        parser.add_argument("--train_batches", type=int, default=1000)
        parser.add_argument("--int_samples", type=int, default=1000000)

        # model params
        #parser.add_argument("--use_prior_weights", action="store_true")
        parser.add_argument("--units", type=int, default=16)
        parser.add_argument("--layers", type=int, default=2)
        parser.add_argument("--blocks", type=int, default=6)
        parser.add_argument("--activation", type=str, default="leakyrelu",
                            choices={"relu", "elu", "leakyrelu", "tanh"})
        parser.add_argument("--initializer", type=str, default="glorot_uniform",
                            choices={"glorot_uniform", "he_uniform"})
        parser.add_argument("--loss", type=str, default="variance",
                            choices={"variance", "neyman_chi2", "exponential"})
        parser.add_argument("--separate_flows", action="store_true")

        # mcw model params
        parser.add_argument("--mcw_units", type=int, default=16)
        parser.add_argument("--mcw_layers", type=int, default=2)
        parser.add_argument("--mcw_res_additive", action="store_true", default=False,
                            help="use additive residual layers")

        # Train params
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--schedule", type=str)
        parser.add_argument("--batch_size", type=int, default=1000)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--lr_decay", type=float, default=0.01)
        parser.add_argument("--train_mcw", action="store_true")
        parser.add_argument("--fixed_mcw", dest="train_mcw", action="store_false")
        parser.set_defaults(train_mcw=True)
        parser.add_argument("--sample_capacity", type=int, default=0)
        parser.add_argument("--uniform_channel_ratio", type=float, default=1.)
        parser.add_argument("--variance_history_length", type=int, default=1000)

        # Plot
        parser.add_argument("--pre_plotting", action="store_true")
        parser.add_argument("--post_plotting", action="store_true")
        parser.add_argument("--plot_var_scatter", action="store_true")

        self.define_physics_arguments(parser)
        self.args = parser.parse_args()

    def define_physics_arguments(self, parser):
        raise NotImplementedError("Process-specific arguments not implemented")

    def define_integrand(self):
        raise NotImplementedError("Integrand not implemented")

    def define_mappings(self):
        raise NotImplementedError("Mappings not implemented")

    def define_prior(self):
        raise NotImplementedError("Prior not implemented")

    def define_output(self):
        raise NotImplementedError("Output settings not implemented")

    def define_flow_network(self):
        FLOW_META = {
            "units": self.args.units,
            "layers": self.args.layers,
            "initializer": self.args.initializer,
            "activation": self.args.activation,
        }

        make_flow = lambda dims_c: RQSVegasFlow(
            [self.dims_in],
            dims_c=dims_c,
            n_blocks=self.args.blocks,
            subnet_meta=FLOW_META,
            subnet_constructor=MLP,
            hypercube_target=True,
        )

        if self.args.separate_flows:
            self.flow = MultiFlow([make_flow(None) for i in range(self.n_channels)])
        else:
            self.flow = make_flow([[self.n_channels]])

    def define_mcw_network(self):
        MCW_META = {
            "units": self.args.mcw_units,
            "layers": self.args.mcw_layers,
            "initializer": self.args.initializer,
            "activation": self.args.activation,
        }

        if self.prior is None:
            self.mcw_net = mcw_model(
                    dims_in=self.dims_in, n_channels=self.n_channels, meta=MCW_META)
        else:
            if self.args.mcw_res_additive:
                self.mcw_net = additive_residual_mcw_model(
                        dims_in=self.dims_in, n_channels=self.n_channels, meta=MCW_META)
            else:
                self.mcw_net = residual_mcw_model(
                        dims_in=self.dims_in, n_channels=self.n_channels, meta=MCW_META)

    def define_integrator(self):
        # Define training params
        if self.args.schedule is None:
            self.schedule = ["g"] * self.args.epochs
        else:
            self.schedule = parse_schedule(self.args.schedule)

        # Prepare scheduler and optimzer
        #lr_schedule = [tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE),
        #               tf.keras.optimizers.schedules.InverseTimeDecay(LR, DECAY_STEP, DECAY_RATE)]
        #opt_list = [tf.keras.optimizers.Adam(lr_schedule[0]), tf.keras.optimizers.Adam(lr_schedule[1])]
        #opt_list = [tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(LR)]
        #opt_and_lay = [(opt_list[0], mcw_net.layers), (opt_list[1], flow.layers)]
        #opt = tfa.optimizers.MultiOptimizer(opt_and_lay)

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                self.args.lr, self.args.train_batches, self.args.lr_decay)
        self.opt = tf.keras.optimizers.Adam(lr_schedule)

        base_dist = StandardUniform((self.dims_in,))

        self.integrator = MultiChannelIntegrator(
            self.integrand,
            self.flow,
            self.opt,
            mcw_model=self.mcw_net if self.args.train_mcw else None,
            mappings=self.mappings,
            use_weight_init=self.prior is not None,
            n_channels=self.n_channels,
            uniform_channel_ratio=self.args.uniform_channel_ratio,
            loss_func=self.args.loss,
            sample_capacity=self.args.sample_capacity,
            variance_history_length=self.args.variance_history_length
        )

    def run_plots(self, prefix):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        dist = DistributionPlot(
            self.log_dir, self.plot_name, which_plots=[True, False, False, False]
        )
        channel_data = []
        for i in range(self.n_channels):
            print(f"Sampling from channel {i}")
            x, weight, alphas, alphas_prior = self.integrator.sample_per_channel(
                self.args.int_samples, i, weight_prior=self.prior, return_alphas=True
            )
            p = to_four_mom(x).numpy()
            alphas_prior = None if alphas_prior is None else alphas_prior.numpy()
            channel_data.append((p, weight.numpy(), alphas.numpy(), alphas_prior))

        if not hasattr(self, "true_data"):
            weight_truth, events_truth = self.integrator.sample_weights(
                    self.args.int_samples, yield_samples=True)
            p_truth = to_four_mom(events_truth).numpy()
            self.true_data = (p_truth, weight_truth.numpy())

        with open(os.path.join(self.log_dir, f"{prefix}_data.pkl"), "wb") as f:
            pickle.dump({
                "channel_data": channel_data,
                "true_data": self.true_data
            }, f)

        print("Plotting channel weights")
        dist.plot_channels_stacked(channel_data, self.true_data, f"{prefix}_stacked")

        print("Plotting weight distribution")
        plot_weights(channel_data, self.log_dir, f"{prefix}_weight_dist")

        if hasattr(self, "train_variance"):
            print("Plotting variances during training")
            plot_variances(self.train_variance, prefix=self.plot_name, log_axis=True,
                           log_dir=self.log_dir)

    def run_integration(self, prefix):
        res, err = self.integrator.integrate(self.args.int_samples, weight_prior=self.prior)
        relerr = err / res * 100

        RES_TO_PB = 0.389379 * 1e9  # Conversion factor from GeV^-2 to pb
        res *= RES_TO_PB
        err *= RES_TO_PB

        name = {"pre": "Pre", "post": "Opt."}[prefix]

        print(f"\n {name} Multi-Channel integration ({self.args.int_samples:.1e} samples): ")
        print("----------------------------------------------------------------")
        print(f" Number of channels: {self.n_channels}                         ")
        print(f" Result: {res:.8f} +- {err:.8f} pb ( Rel error: {relerr:.4f} %)")
        print("----------------------------------------------------------------\n")

    def run_training(self):
        train_losses = []
        start_time = time.time()
        if self.args.plot_var_scatter:
            self.train_variance = []
        batch_size = self.args.batch_size
        for e, etype in enumerate(self.schedule):
            if self.args.plot_var_scatter:
                self.train_variance.append([])
                for _ in range(25):
                    _, err = self.integrator.integrate(batch_size, weight_prior=self.prior)
                    var_int = err**2 * (batch_size - 1.)
                    self.train_variance[-1].append(var_int.numpy())
            if etype == "g":
                batch_train_losses = []
                # do multiple iterations.
                for _ in range(self.args.train_batches):
                    batch_loss = self.integrator.train_one_step(
                            batch_size, weight_prior=self.prior)
                    batch_train_losses.append(batch_loss)

                train_loss = tf.reduce_mean(batch_train_losses)
                train_losses.append(train_loss)

                print(
                    f"Epoch #{e+1}: generating, Loss: {train_loss}, " +
                    f"Learning_Rate: {self.opt._decayed_lr(tf.float32)}"
                )

            elif etype == "r":
                train_loss = self.integrator.train_on_stored_samples(
                        batch_size, weight_prior=self.prior)

                print(
                    f"Epoch #{e+1}: on samples, Loss: {train_loss}, " +
                    f"Learning_Rate: {self.opt._decayed_lr(tf.float32)}"
                )

            elif etype == "d":
                self.integrator.delete_samples()

                print(f"Epoch #{e+1}: delete samples")
        end_time = time.time()
        print("--- Run time: %s hour ---" % ((end_time - start_time) / 60 / 60))
        print("--- Run time: %s mins ---" % ((end_time - start_time) / 60))
        print("--- Run time: %s secs ---" % ((end_time - start_time)))

    def run_unweighting(self):
        n_opt = self.args.int_samples // 100
        uwgt_eff, uwgt_eff_part, over_weights = self.integrator.acceptance(n_opt)
        over_weights *= 100
        uwgt_eff_part *= 100
        uwgt_eff *= 100

        print(f"\n Unweighting efficiency using ({n_opt:.1e} samples): ")
        print("----------------------------------------------------------------")
        print(f" Number of channels: {self.n_channels}                         ")
        print(f" Efficiency 1 : {uwgt_eff:.8f} %                               ")
        print(f" Efficiency 2 : {uwgt_eff_part:.8f} %                          ")
        print(f" Over_weights : {over_weights:.8f} %                           ")
        print("----------------------------------------------------------------\n")

    def run(self):
        self.tf_setup()
        self.parse_arguments()
        self.define_integrand()
        self.define_mappings()
        self.define_prior()
        self.define_flow_network()
        self.define_mcw_network()
        self.define_integrator()
        self.define_output()

        if self.args.pre_plotting:
            self.run_plots("pre")
        self.run_integration("pre")
        self.run_training()
        if self.args.post_plotting:
            self.run_plots("post")
        self.run_integration("post")
        self.run_unweighting()
