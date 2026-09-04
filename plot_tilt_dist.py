"""Octahedral tilt-angle distribution of Cs2SnI6 at 300 K, in the style of the
perovskite tilt-classification figure (three pseudo-cubic axes a/b/c, filled
histogram + Gaussian fit, -45..45 deg).  Data: 666_mlff/tilt_300K.npy, shape
(nframes, n_octahedra, 3) from PDynA.  Single peak centred at 0 on all three
axes = a^0 a^0 a^0 (untilted cubic).

Run: python 666_mlff/plot_tilt_dist.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

BANNER = "#6cc3ac"
FILL = "#c9cdd1"
LINE = "#2f3a4a"

a = np.load("666_mlff/tilt_300K.npy")            # (nframes, n_oct, 3)
axes_lbl = ["a", "b", "c"]
data = [a[:, :, i].ravel() for i in range(3)]
data = [d[np.isfinite(d)] for d in data]

bins = np.arange(-25, 25.01, 0.5)
xg = np.linspace(-45, 45, 600)

fig, axs = plt.subplots(3, 1, figsize=(4.3, 5.4), sharex=True,
                        gridspec_kw={"hspace": 0.14})
fig.suptitle(r"Cs$_2$SnI$_6$   300 K :   $a^{0}a^{0}a^{0}$",
             fontsize=12.5, fontstyle="italic", y=0.98,
             bbox=dict(boxstyle="round,pad=0.4", fc=BANNER, ec="none"))

for ax, d, lab in zip(axs, data, axes_lbl):
    ax.hist(d, bins=bins, density=True, color=FILL, edgecolor="none", zorder=2)
    mu, sd = d.mean(), d.std()
    ax.plot(xg, norm.pdf(xg, mu, sd), color=LINE, lw=1.7, zorder=3)
    ax.set_xlim(-45, 45)
    ax.set_xticks([-45, -30, -15, 0, 15, 30, 45])
    ax.set_yticks([])
    ax.set_ylabel("Counts (a.u.)", fontsize=9.5)
    ax.text(0.975, 0.85, lab, transform=ax.transAxes, ha="right", va="top",
            fontstyle="italic", fontsize=14)
    ax.tick_params(labelsize=9)
    ax.axvline(0, color="0.6", lw=0.6, ls=":", zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axs[-1].set_xlabel(r"Tilt angle (deg)", fontsize=10.5)
fig.savefig("666_mlff/tilt_dist_300K.png", dpi=200, bbox_inches="tight")
print("saved 666_mlff/tilt_dist_300K.png")
print("per-axis (mean, std) deg:",
      [(round(float(d.mean()), 3), round(float(d.std()), 3)) for d in data])
