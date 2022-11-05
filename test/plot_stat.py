import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        if len(orig_handle) == 2:
            l1 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height],
                                linestyle=orig_handle[0], color=orig_handle[1])

            return [l1]
        else:
            l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                                linestyle=orig_handle[0], color=orig_handle[2])
            l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
                                linestyle=orig_handle[1], color = orig_handle[2])
            return [l1, l2]

t_gen = 40 * 1e-6
t_samp = 33 * 1e-6
stat_dec = np.linspace(1,6,1000)
integ_times = [1e-6, 1e-5, 1e-4, 1e-3]
t_labels = ["1\\mu s", "10\\mu s", "100\\mu s", "1ms"]

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.style.use('dark_background')

FONTSIZE = 18
fig, ax1 = plt.subplots(figsize=(6.6,6))

for label in ( [ax1.yaxis.get_offset_text()] +
                ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontsize(FONTSIZE)


colors = ['#3182bd', '#31a354', "#c6ad2a", '#de2d26']
for i, integ_time in enumerate(integ_times):
    t_inc = (t_gen+integ_time)/((1-1/stat_dec) * t_samp + 1/stat_dec * (t_gen+integ_time))
    wu_inc = (1-1/stat_dec)*(t_gen+integ_time)/t_samp + 1/stat_dec
    # ax1.plot(stat_dec, wu_inc, colors[i])
    ax1.plot(stat_dec, 1/t_inc, colors[i])
ax1.set_yscale("log")
#ax2.set_yscale("log")
ax1.set_ylim(ymin=9e-2)
#ax1.set_ylim(ymin=9e-1)
#ax1.set_ylim(ymax=5e+01)
#ax2.set_ylim(ymin=1e-2)
labellist = [r"$t_f =1\,\mu \mathrm{s}$", r"$t_f =10\,\mu \mathrm{s}$", r"$t_f =100\,\mu \mathrm{s}$", r"$t_f =1\,\mathrm{ms}$"]
# hist_style_list = [
#     ( "-", "dashed", colors[0]),
#     ( "-", "dashed", colors[1]), # "#e6cc3d"
#     ( "-", "dashed", colors[2]), # '#74c476'
#     ( "-", "dashed", colors[3]), # '#6baed6'
# ]

hist_style_list = [
    ( "-", colors[0]),
    ( "-", colors[1]), # "#e6cc3d"
    ( "-", colors[2]), # '#74c476'
    ( "-", colors[3]), # '#6baed6'
]

ax1.legend(
    hist_style_list,
    labellist,
    handler_map={tuple: AnyObjectHandler()},
    frameon=False,          
    handletextpad=0.4,
    prop={'size': int(FONTSIZE-4)},
    loc="lower right")
    
# ax1.text(0.25, 0.5, r"$t_g = 40\,\mu \mathrm{s}$", transform = plt.gca().transAxes, fontsize=FONTSIZE-4)
# ax1.text(0.25, 0.45, r"$t_s = 30\,\mu \mathrm{s}$", transform = plt.gca().transAxes, fontsize=FONTSIZE-4)
ax1.text(0.25, 0.15, r"$t_g = 40\,\mu \mathrm{s}$", transform = plt.gca().transAxes, fontsize=FONTSIZE-4)
ax1.text(0.25, 0.10, r"$t_s = 30\,\mu \mathrm{s}$", transform = plt.gca().transAxes, fontsize=FONTSIZE-4)
ax1.set_xlabel(r"reduction factor of training statistics", fontsize = FONTSIZE)
ax1.set_ylabel(r"reduction factor in training time", fontsize = FONTSIZE)
#ax1.set_ylabel(r"increase factor in weight updates", fontsize = FONTSIZE)
fig.savefig("times_vs_stat_dark.pdf", bbox_inches="tight",  transparent=True)