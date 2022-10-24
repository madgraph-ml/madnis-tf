"""
Plotting routines and functions.
"""

import numpy as np
from scipy.stats import binned_statistic
np.seterr(invalid='ignore', divide='ignore')

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.ticker import ScalarFormatter
from matplotlib.legend_handler import HandlerBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

gcolor = '#3b528b'
dcolor = '#e41a1c'
teal = '#10AB75'

try:
    plt.rc("text", usetex=True)
    FONTSIZE = 18
except:
    FONTSIZE = 12

##########
# Setup ##
##########

class ScalarFormatterForceFormat(ScalarFormatter):
	def _set_format(self):	# Override function that finds format to use.
		self.format = "%1.1f"  # Give format her

class AnyObjectHandler(HandlerBase):
	def create_artists(self, legend, orig_handle,
					   x0, y0, width, height, fontsize, trans):
		l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
						   linestyle=orig_handle[1], color=orig_handle[0])
		l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
						   linestyle = orig_handle[2], color = orig_handle[0])
		return [l1, l2]

def dup_last(a):
    return np.append(a, a[-1])


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
        prop={"size": FONTSIZE-4},
    )
    ax1.set_xlabel(r"Epochs")
    ax1.set_ylabel(r"Loss")
    fig.savefig(log_dir + "/%s.pdf" % name, dpi=120, bbox_inches="tight")
    plt.close("all")


#################
# Plot Weights ##
#################


def plot_weights(channel_data, log_dir=".", name=""):
    """Plot histogram of weights"""
    fig, ax1 = plt.subplots(1, figsize=(6.6, 6))
    all_weights = np.stack([weights for _, weights, _, _ in channel_data])
    m_weight = np.mean(all_weights)
    w_min = np.min(all_weights/m_weight)
    w_max = np.max(all_weights/m_weight)
    bins = np.logspace(np.log10(1e-12), np.log10(1e03), 40)
    
    for label in ( [ax1.yaxis.get_offset_text()] +
                    ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(FONTSIZE-2)

    for i, (_, weights, _, _) in enumerate(channel_data):
        color = f"C{i}"

        if len(weights.shape) == 2:
            wh_all = np.stack([np.histogram(w, bins=bins)[0] for w in weights], axis=0)
            wh = np.mean(wh_all, axis=0)
            wh_err = np.std(wh_all, axis=0)
            ax1.fill_between(bins, dup_last(wh - wh_err), dup_last(wh + wh_err),
                             facecolor=color, alpha=0.3, step="post")
        else:
            m_weight = np.mean(weights)
            wh, _ = np.histogram(weights/m_weight, bins=bins)

        ax1.stairs(
            wh, edges=bins, color=color, label=f"chan {i}", linewidth=1.0, baseline=None)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Weight", fontsize = FONTSIZE-2)
    ax1.set_ylabel("Number of events", fontsize = FONTSIZE-2)
    ax1.legend(frameon=False, prop={"size": int(FONTSIZE-5)}, loc='upper left')
    fig.savefig(log_dir + "/%s.pdf" % name, dpi=120, bbox_inches="tight")
    plt.close("all")


################
# Plot Alphas ##
################


def plot_alphas(p_values, alphas, truth, mappings, prefix=""):
    """Plot the alphas"""

    plt.rc("figure", figsize=(6.6, 6))
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    FONTSIZE = 16

    # Define as numpy
    p_values = p_values
    alphas = alphas
    truth = truth
    m1 = mappings[0]
    m2 = mappings[1]

    # Define data
    n_curves = len(mappings)
    xs = p_values[:, 0]
    ys = []
    for i in range(n_curves):
        ys.append(alphas[:, i])

    amg1 = m1 / (m1 + m2)
    amg2 = m2 / (m1 + m2)

    # Fig layout
    #
    
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
    
    
    for j in range(2):
        for label in ( [axs[j].yaxis.get_offset_text()] +
                        axs[j].get_xticklabels() + axs[j].get_yticklabels()):
            label.set_fontsize(FONTSIZE)


    # Upper panel
    #
    
    axs[0].plot(xs, truth, color="black", markersize=12, label=r"$\mathrm{Camel}(x)$")
    axs[0].plot(xs, m1, color="red", markersize=12, label=r"$g_1(x)$")
    axs[0].plot(xs, m2, color="blue", markersize=12, label=r"$g_2(x)$")


    axs[0].legend(loc="upper left", frameon=False, prop={"size": int(FONTSIZE-4)})
    axs[0].set_ylabel(r"Probability density", fontsize = FONTSIZE)
    maxi = np.max([np.max(m1), np.max(m2), np.max(truth)])
    axs[0].set_ylim(top=maxi * 1.1)
    
    # Lower panel
    #
    
    axs[1].plot(xs, ys[0], color="red", markersize=12, label=r"$\alpha_1(x)$", linestyle="dashed")
    axs[1].plot(xs, ys[1], color="blue", markersize=12, label=r"$\alpha_2(x)$", linestyle="dashed")
    axs[1].plot(xs, amg1, color="red", markersize=12, label=r"$\alpha_{\mathrm{opt},1}(x)$", linestyle="dotted")
    axs[1].plot(xs, amg2, color="blue", markersize=12, label=r"$\alpha_{\mathrm{opt},2}(x)$", linestyle="dotted")

    axs[1].legend(loc="upper center", ncol=4, frameon=False, prop={"size": int(FONTSIZE-8)})
    axs[1].set_xlabel(r"$x$", fontsize = FONTSIZE)
    axs[1].set_ylabel(r"Channel weights", fontsize = FONTSIZE)
    axs[1].set_ylim(bottom=-0.05, top=1.22)
    name = f"{prefix}_alphas"
    fig.savefig(f"{name}.pdf", format="pdf")
    plt.close("all")


def plot_alphas_multidim(axs, channel_data, args):
    """Plot the alphas for multidimensional data"""
    if args[6]:
        axs[0].set_yscale('log')
    else:
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0,0))
        axs[0].yaxis.set_major_formatter(yfmt)

    for i, (ys, weights, alphas, alpha_priors) in enumerate(channel_data):
        if len(ys.shape) == 2:
            ys = ys[None,:,:]
            weights = weights[None,:]
            alphas = alphas[None,:]
            if alpha_priors is not None:
                alpha_priors = alpha_priors[None,:]
            else:
                alpha_priors = [None] * ys.shape[0]
            plot_errors = False
        else:
            plot_errors = True
        has_prior = alpha_priors[0] is not None

        y_p_all = []
        alpha_binned_all = []
        alpha_prior_binned_all = []
        for y, weight, alpha, alpha_prior in zip(ys, weights, alphas, alpha_priors):
            y = args[1](y, args[0])
            y_p, x_p = np.histogram(y, args[2], density=True, range=args[3])
            y_p_all.append(y_p)
            weight_norm, _, _ = binned_statistic(y, weight/alpha, statistic='sum', bins=x_p)
            alpha_binned_all.append(
                binned_statistic(y, weight, statistic='sum', bins=x_p)[0] / weight_norm)
            if has_prior:
                alpha_prior_binned_all.append(
                    binned_statistic(y, weight/alpha * alpha_prior, statistic='sum',
                                     bins=x_p)[0] / weight_norm
                )

        y_p_all = np.stack(y_p_all, axis=0)
        y_p = np.mean(y_p_all, axis=0)
        y_p_err = np.std(y_p_all, axis=0)
        alpha_binned_all = np.stack(alpha_binned_all, axis=0)
        alpha_binned = np.mean(alpha_binned_all, axis=0)
        alpha_binned_err = np.std(alpha_binned_all, axis=0)
        if has_prior:
            alpha_prior_binned_all = np.stack(alpha_prior_binned_all, axis=0)
            alpha_prior_binned = np.mean(alpha_prior_binned_all, axis=0)
            alpha_prior_binned_err = np.std(alpha_prior_binned_all, axis=0)

        color = f'C{i}'

        if plot_errors:
            axs[0].fill_between(x_p, dup_last(y_p - y_p_err), dup_last(y_p + y_p_err),
                                facecolor=color, alpha=0.3, step="post")
            axs[1].fill_between(x_p, dup_last(alpha_binned - alpha_binned_err),
                                dup_last(alpha_binned + alpha_binned_err),
                                facecolor=color, alpha=0.3, step="post")
            if has_prior:
                axs[1].fill_between(x_p,
                                    dup_last(alpha_prior_binned - alpha_prior_binned_err),
                                    dup_last(alpha_prior_binned + alpha_prior_binned_err),
                                    facecolor=color, alpha=0.3, step="post")

        axs[0].stairs(
            y_p, edges=x_p, color=color, label=f'chan {i}', linewidth=1.0, baseline=None)
        if i == 0:
            lbl1 = "Learned"
            lbl2 = "Prior"
        else:
            lbl1 = None
            lbl2 = None
        axs[1].stairs(
            alpha_binned, edges=x_p, color=color, linewidth=1.0, label=lbl1, baseline=None)
        if has_prior:
            axs[1].stairs(alpha_prior_binned, edges=x_p, color=color, ls="dashed",
                          linewidth=1.0, label=lbl2, baseline=None)

    for j in range(2):
        for label in ( [axs[j].yaxis.get_offset_text()] +
                        axs[j].get_xticklabels() + axs[j].get_yticklabels()):
            label.set_fontsize(FONTSIZE)

    axs[0].set_ylabel('Probability density', fontsize = FONTSIZE)
    axs[0].legend(loc='upper right', prop={'size': int(FONTSIZE-4)}, frameon=False)

    axs[1].set_ylabel(r'$\alpha$', fontsize = FONTSIZE-2)
    axs[1].set_ylim(bottom=-0.05, top=1.22)
    for yy in [0., 0.5, 1.]:
        axs[1].axhline(y=yy,linewidth=1, linestyle='--', color='grey')
    axs[1].set_xlabel(args[4], fontsize = FONTSIZE)
    if has_prior:
        axs[1].legend(loc="upper center", ncol=2, frameon=False,
                      prop={"size": int(FONTSIZE-8)})
        
        
def plot_alphas_stack(axs, channel_data, true_data, args):
    """Plot the alphas for multidimensional data"""
    if args[6]:
        axs[0].set_yscale('log')
    else:
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0,0))
        axs[0].yaxis.set_major_formatter(yfmt)

    y_all = []
    labels = []
    weight_all = []
    alpha_all = []
    for i, (y, weight, alpha, alpha_prior) in enumerate(channel_data):
        has_prior = alpha_prior is not None
        y = args[1](y, args[0])
        y_all.append(y)
        _, x_p = np.histogram(y, args[2], density=True, range=args[3])
        # if i == 0:
        #     y_t, x_t = np.histogram(y, args[2], density=True, range=args[3], weights=weight)
        #     axs[0].stairs(y_t, edges=x_t, ls="dashed", color='black', label="Truth")
        weight_all.append(weight/alpha)
        alpha_all.append(alpha)
        weight_norm, _, _ = binned_statistic(y, weight/alpha, statistic='sum', bins=x_p)
        alpha_binned = binned_statistic(y, weight, statistic='sum', bins=x_p)[0] / weight_norm
        if has_prior:
            alpha_prior_binned = binned_statistic(y, weight/alpha * alpha_prior, statistic='sum', bins=x_p)[0] / weight_norm

        color = f'C{i}'
        labels.append(f'chan {i}')
        # axs[0].stairs(
        #     y_p, edges=x_p, color=color, label=f'chan {i}', linewidth=1.0, baseline=None)

        if i == 0:
            lbl1 = "Learned"
            lbl2 = "Prior"
        else:
            lbl1 = None
            lbl2 = None
        axs[1].stairs(
            alpha_binned, edges=x_p, color=color, linewidth=1.0, label=lbl1, baseline=None)
        if has_prior:
            axs[1].stairs(alpha_prior_binned, edges=x_p, color=color, ls="dashed",
                          linewidth=1.0, label=lbl2, baseline=None)
    
    y_stack = np.stack(y_all, axis=-1)
    alpha_stack = np.stack(alpha_all, axis=-1)    
    axs[0].hist(y_stack, args[2], density=True, histtype='bar', stacked=True, label=labels, range=args[3], weights=alpha_stack)
    
    # Plot truth data
    y_comb = args[1](true_data[0], args[0])
    weight_comb = true_data[1]
    y_t, x_t = np.histogram(y_comb, args[2], density=True, range=args[3], weights=weight_comb)
    axs[0].stairs(y_t, edges=x_t, ls="dashed", color='black', label="Truth")
    
    for j in range(2):
        for label in ( [axs[j].yaxis.get_offset_text()] +
                        axs[j].get_xticklabels() + axs[j].get_yticklabels()):
            label.set_fontsize(FONTSIZE)

    axs[0].set_ylabel('Probability density', fontsize = FONTSIZE)
    axs[0].legend(loc='upper right', prop={'size': int(FONTSIZE-4)}, frameon=False)

    axs[1].set_ylabel(r'$\alpha$', fontsize = FONTSIZE-2)
    axs[1].set_ylim(bottom=-0.05, top=1.22)
    for yy in [0., 0.5, 1.]:
        axs[1].axhline(y=yy,linewidth=1, linestyle='--', color='grey')
    axs[1].set_xlabel(args[4], fontsize = FONTSIZE)
    if has_prior:
        axs[1].legend(loc="upper center", ncol=2, frameon=False,
                      prop={"size": int(FONTSIZE-8)})
    
#######################
# Plot Distributions ##
#######################

def plot_distribution_ratio(axs, y_train, y_predict, weights, args):
    """Plot the distributions including ratio"""
    y_train = args[1](y_train, args[0])

    if args[6]:
        axs[0].set_yscale('log')
    else:
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0,0))
        axs[0].yaxis.set_major_formatter(yfmt)

    y_t, x_t = np.histogram(y_train, args[2], density=True, range=args[3])
    if len(y_predict.shape) == 2:
        y_predict = y_predict[None,:,:]
        plot_errors = False
    else:
        plot_errors = True

    y_ps = []
    for y_pred in y_predict:
        y_pred = args[1](y_pred, args[0])
        y_ps.append(np.histogram(y_pred, args[2], density=True, range=args[3],
                                 weights=weights)[0])
    y_ps = np.stack(y_ps, axis=0)
    y_p = np.mean(y_ps, axis=0)
    y_p_err = np.std(y_ps, axis=0)

    # Upper panel
    #

    if plot_errors:
        axs[0].fill_between(x_t, dup_last(y_p - y_p_err), dup_last(y_p + y_p_err),
                            facecolor=dcolor, alpha=0.3, step="post")

    axs[0].stairs(y_t, edges=x_t, color=gcolor, label='Truth', linewidth=1.0, baseline=None)
    axs[0].stairs(y_p, edges=x_t, color=dcolor, label='MadNIS', linewidth=1.0, baseline=None)

    for j in range(2):
        for label in ( [axs[j].yaxis.get_offset_text()] +
                        axs[j].get_xticklabels() + axs[j].get_yticklabels()):
            label.set_fontsize(FONTSIZE)


    axs[0].set_ylabel('Normalized', fontsize = FONTSIZE)
    axs[0].legend(loc='upper right', prop={'size': int(FONTSIZE-4)}, frameon=False)

    # Lower panel
    #

    axs[1].set_ylabel(r'$\text{Ratio}$', fontsize = FONTSIZE)

    y_r = (y_p)/y_t
    y_r [np.isnan(y_r )==True]=1
    y_r [y_r==np.inf]=1
    y_r_err = y_p_err / y_t

    if plot_errors:
        axs[1].fill_between(x_t, dup_last(y_r - y_r_err), dup_last(y_r + y_r_err),
                            facecolor=dcolor, alpha=0.3, step="post")
    axs[1].stairs(y_r, edges=x_t, color=dcolor, linewidth=1.0, baseline=None)
    axs[1].set_ylim((0.82,1.18))
    axs[1].set_yticks([0.9, 1.0, 1.1])
    axs[1].set_yticklabels([r'$0.9$', r'$1.0$', "$1.1$"])

    axs[1].axhline(y=1,linewidth=1, linestyle='--', color='grey')
    axs[1].axhline(y=2,linewidth=1, linestyle='--', color='grey')
    axs[1].set_xlabel(args[4], fontsize = FONTSIZE)
 
 
def plot_distribution_diff_ratio(axs, y_train, y_predict, weights, args):
    """Plot the distributions including ratio ind absolute difference"""

    y_train = args["observable"](y_train, args["particle_id"])

    if args["log_scale"]:
        axs[0].set_yscale('log')
    else:
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0,0))
        axs[0].yaxis.set_major_formatter(yfmt)

    y_t, x_t = np.histogram(y_train, args["bins"], density=True, range=args["range"])
    if len(y_predict.shape) == 2:
        y_predict = y_predict[None,:,:]
        plot_errors = True
    else:
        plot_errors = False

    y_ps = []
    for y_pred in y_predict:
        y_pred = args[1](y_pred, args[0])
        y_ps.append(np.histogram(y_pred, args[2], density=True, range=args[3],
                                 weights=weights)[0])
    y_ps = np.stack(y_ps, axis=0)
    y_p = np.mean(y_ps, axis=0)
    y_p_err = np.std(y_ps, axis=0)

    line_dat, = axs[0].stairs(
        y_t, edges=x_t, color=dcolor, label='True', linewidth=1.0, baseline=None)
    line_gen, = axs[0].stairs(
        y_p, edges=x_t, color=gcolor, label='GAN', linewidth=1.0, baseline=None)

    if args["range"] == (-3.14,3.14):
        axs[0].set_ylim((-0.02,0.3))

    for j in range(3):
        for label in ( [axs[j].yaxis.get_offset_text()] +
                      axs[j].get_xticklabels() + axs[j].get_yticklabels()):
            label.set_fontsize(FONTSIZE-2)

    axs[0].set_ylabel(r'$\frac{1}{\sigma}\frac{\mathrm{d}\sigma}{\mathrm{d}%s}$' %args["label"], fontsize = FONTSIZE)

    axs[0].legend(
        [line_gen, line_dat],
        ['GAN', 'Truth'],
        #title = "GAN vs Data",
        loc='upper right',
        prop={'size':(FONTSIZE-4)},
        frameon=False)

    # middle panel

    axs[1].set_ylabel(r'$\frac{\text{GAN}}{\text{Truth}}$', fontsize = FONTSIZE)

    dummy = 1e-6
    y_r = (y_p)/((y_t + dummy))

    #statistic
    r_stat = np.sqrt(y_p * (y_p + y_t)/((y_t+dummy)**3))
    r_statp = y_r + r_stat
    r_statm = y_r - r_stat

    axs[1].stairs(y_r, edges=x_t, color='black', linewidth=1.0, baseline=None)
    axs[1].stairs(
        r_statp, edges=x_t, color='grey', label='$+- stat$', linewidth=0.5, baseline=None)
    axs[1].stairs(
        r_statm, edges=x_t, color='grey', linewidth=0.5, baseline=None)
    axs[1].fill_between(x_t[:args["bins"]], r_statm, r_statp, facecolor='grey', alpha = 0.5, step = 'mid')

    axs[1].set_ylim((0.85,1.15))
    axs[1].set_yticks([0.9, 1.0, 1.1])
    axs[1].set_yticklabels([r'$0.9$', r'$1.0$', "$1.1$"])
    axs[1].axhline(y=1,linewidth=1, linestyle='--', color='grey')

    # Lowest panel
    axs[2].set_ylabel(r'$\delta [\%]$', fontsize = FONTSIZE)
    y_diff = np.fabs((y_r - 1)) * 100
    diff_stat = 100 * np.sqrt(y_p * (y_p + y_t)/((y_t+dummy)**3))

    axs[2].errorbar(x_t[:args["bins"]], y_diff, yerr=diff_stat, ecolor='grey', color='black', elinewidth=0.5, linewidth=0,  fmt='.', capsize=2)

    axs[2].set_ylim((0.05,20))
    axs[2].set_yscale('log')
    axs[2].set_yticks([0.1, 1.0, 10.0])
    axs[2].set_yticklabels([r'$0.1$', r'$1.0$', "$10.0$"])
    axs[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2., 3., 4., 5., 6., 7., 8., 9.], minor=True)

    axs[2].axhline(y=1.0,linewidth=0.5, linestyle='--', color='grey')
    axs[2].axhspan(0, 1.0, facecolor='#cccccc', alpha=0.3)
    axs[2].set_xlabel(r'${label}$ {unit}'.format(label = args["label"], unit = args["si_unit"]), fontsize = FONTSIZE)

def plot_2d_distribution(fig, axs, y_train, y_predict, weights, args1, args2):
    """Plot the distributions"""
    fontsize=FONTSIZE+2
    data = [[0.,0.], [0.,0.]]
    h = [0.] * 3

    # Fill (x,y) with data
    data[0][0] = args1[1](y_train, args1[0])
    data[1][0] = args2[1](y_train, args2[0])

    data[0][1] = args1[1](y_predict, args1[0])
    data[1][1] = args2[1](y_predict, args2[0])

    h[0], xedges, yedges = np.histogram2d(data[0][0], data[1][0], bins=args1[2], range=(args1[3],args2[3]), density=True)

    if weights is not None:
        h[1], xedges, yedges = np.histogram2d(data[0][1], data[1][1], bins=args1[2], range=(args1[3],args2[3]), density=True, weights=weights)
    else:
        h[1], xedges, yedges = np.histogram2d(data[0][1], data[1][1], bins=args1[2], range=(args1[3],args2[3]), density=True)

    h[2] = (h[1]-h[0])/(h[1]+h[0])
    h[2][np.isnan(h[2])==True]=0 #1
    h[2][h[2]==np.inf]=0 #1

    for j in range(3):
        for label in ( [axs[j].yaxis.get_offset_text()] +
                        axs[j].get_xticklabels() + axs[j].get_yticklabels()):
            label.set_fontsize(fontsize)
   
        Z = h[j].T
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0,0))
        im = axs[j].pcolormesh(xedges, yedges, Z, rasterized=True)
        divider = make_axes_locatable(axs[j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        axs[j].set_xlabel(args1[4], fontsize = fontsize)
        axs[j].set_ylabel(args2[4], fontsize = fontsize)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.set_major_formatter(yfmt)

def plot_2d_distribution_single(fig, axs, y_train, y_predict, weights, args1, args2):
    """Plot the 2d-distributions of the learned function only"""
    del y_train

    if len(y_predict.shape) == 2:
        y_predict = y_predict[None,:,:]
        plot_errors = False
    else:
        plot_errors = True

    hs = []
    for y_pred in y_predict:
        data = [0., 0.]
        data[0] = args1[1](y_pred, args1[0])
        data[1] = args2[1](y_pred, args2[0])
        h, xedges, yedges = np.histogram2d(data[0], data[1], bins=args1[2],
                range=(args1[3],args2[3]), density=True, weights=weights)
        hs.append(h)

    hs = np.stack(hs, axis=0)
    h = np.mean(hs, axis=0)
    h_err = np.std(hs, axis=0)

    if args1[6]:
        axs.set_xscale('log')
    if args2[6]:
        axs.set_yscale('log')

    for label in ([axs.yaxis.get_offset_text()] + axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(FONTSIZE)

    yfmt = ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0,0))
    im = axs.pcolormesh(xedges, yedges, h.T, rasterized=True)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    axs.set_xlabel(args1[4], fontsize = FONTSIZE)
    axs.set_ylabel(args2[4], fontsize = FONTSIZE)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=FONTSIZE)
    cbar.ax.yaxis.set_major_formatter(yfmt)
    for label in ([cbar.ax.yaxis.get_offset_text()]):
        label.set_fontsize(FONTSIZE)

