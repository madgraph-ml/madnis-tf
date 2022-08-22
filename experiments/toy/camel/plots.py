"""
Plotting routines and functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys

##########
# Setup ##
##########


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=16.0)
plt.rc("axes", labelsize="large")
plt.rc("pdf", compression=9)

##############
# Plot Loss ##
##############


def plot_loss(loss, log_dir=".", name="", log_axis=True):
    """Plot the traings curve"""
    fig, ax1 = plt.subplots(1, figsize=(10, 4))
    epoch = np.arange(len(loss))
    loss = np.array(loss)

    if log_axis:
        ax1.set_yscale("log")

    loss_name = name + " Loss"
    plt.plot(epoch, loss[:], color="red", markersize=12, label=r"%s" % loss_name)

    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=9,
        fancybox=True,
        shadow=True,
        prop={"size": 10},
    )
    ax1.set_xlabel(r"Epochs")
    ax1.set_ylabel(r"Loss")
    fig.savefig(log_dir + "/%s.pdf" % name, dpi=120, bbox_inches="tight")
    plt.close("all")


################
# Plot Alphas ##
################


def plot_alphas(p_values, alphas, truth, mappings, prefix=""):
    """Plot the alphas"""

    plt.rc("figure", figsize=(6.6, 6))

    # Define as numpy
    p_values = p_values.numpy()
    alphas = alphas.numpy()
    truth = truth.numpy()
    m1 = mappings[0].numpy()
    m2 = mappings[1].numpy()

    # Define Points
    n_curves = len(mappings)
    xs = p_values[:, 0]
    ys = []
    for i in range(n_curves):
        ys.append(alphas[:, i])

    amg1 = m1 / (m1 + m2)
    amg2 = m2 / (m1 + m2)

    fig, axs = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.00}
    )
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.1,0.05,0.99,0.96))

    axs[0].tick_params(
        axis="both",
        left=True,
        right=True,
        top=True,
        bottom=True,
        which="both",
        direction="in",
        width=0.5,
        zorder=10.0,
    )
    axs[1].tick_params(
        axis="both",
        left=True,
        right=True,
        top=True,
        bottom=True,
        which="both",
        direction="in",
        width=0.5,
        zorder=10.0,
    )
    axs[0].minorticks_on()
    axs[1].minorticks_on()

    # axs[0].set_yscale('log')

    # Upper panel
    axs[0].plot(xs, truth, color="black", markersize=12, label=r"$\mathrm{Camel}(p)$")
    axs[0].plot(xs, m1, color="red", markersize=12, label=r"$g_1(p)$")
    axs[0].plot(xs, m2, color="blue", markersize=12, label=r"$g_2(p)$")
    # axs[0].plot(
    #     xs,
    #     m1[:, 0] / ys[0],
    #     color="black",
    #     markersize=12,
    #     label=r"$\frac{\varphi_1(p)}{\alpha_1(p)}$",
    #     linestyle="dashed",
    # )

    axs[0].legend(loc="upper left", frameon=False, prop={"size": 12})
    axs[0].set_ylabel(r"Probability density")
    maxi = np.max([np.max(m1), np.max(m2), np.max(truth)])
    axs[0].set_ylim(top=maxi * 1.1)

    axs[1].plot(
        xs,
        ys[0],
        color="red",
        markersize=12,
        label=r"$\alpha_1(p)$",
        linestyle="dashed",
    )
    axs[1].plot(
        xs,
        ys[1],
        color="blue",
        markersize=12,
        label=r"$\alpha_2(p)$",
        linestyle="dashed",
    )

    axs[1].plot(
        xs,
        amg1,
        color="red",
        markersize=12,
        label=r"$\alpha_{\mathrm{opt},1}(p)$",
        linestyle="dotted",
    )
    axs[1].plot(
        xs,
        amg2,
        color="blue",
        markersize=12,
        label=r"$\alpha_{\mathrm{opt},2}(p)$",
        linestyle="dotted",
    )

    axs[1].legend(loc="upper center", ncol=4, frameon=False, prop={"size": 10})
    axs[1].set_xlabel(r"$p$")
    axs[1].set_ylabel(r"Channel weights")
    axs[1].set_ylim(bottom=-0.05, top=1.22)
    name = f"{prefix}_alphas"
    fig.savefig(f"{name}.pdf", format="pdf")
    plt.close("all")
