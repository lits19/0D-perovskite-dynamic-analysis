"""Cs2SnI6 300 K tilt distribution, PDynA style but FILLED.

Data handling is PDynA's own (pdyna.structural.periodicity_fold, same bins /
range / colours / labels as pdyna.analysis.draw_tilt_density); the only change
is that each histogram curve is filled under.

Run in the `pmg` env:
    MPLBACKEND=Agg /opt/anaconda3/envs/pmg/bin/python 666_mlff/run_pdyna_tilt.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pdyna.structural import periodicity_fold

os.chdir(os.path.dirname(os.path.abspath(__file__)))

n_bins = 100
symm_n_fold = 4
T = np.load("tilt_300K.npy")                     # (nframes, n_oct, 3)
T = periodicity_fold(T, n_fold=symm_n_fold)      # PDynA's own folding

hrange = [-45, 45]
tlabel = [-45, -30, -15, 0, 15, 30, 45]
tup_T = [T[:, :, i].reshape(-1) for i in range(3)]

figs, axs = plt.subplots(3, 1)
labels = [r'$\mathit{a}$', r'$\mathit{b}$', r'$\mathit{c}$']
colors = ["C0", "C1", "C2"]
for i in range(3):
    y, binEdges = np.histogram(tup_T[i], bins=n_bins, range=hrange)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    axs[i].fill_between(bincenters, y, color=colors[i], alpha=0.35, zorder=2)
    axs[i].plot(bincenters, y, color=colors[i], linewidth=2, zorder=3)
    axs[i].text(0.03, 0.82, labels[i], horizontalalignment='center', fontsize=15,
                verticalalignment='center', transform=axs[i].transAxes)

for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(hrange)
    ax.set_xticks(tlabel)
    ax.set_yticks([])
    ax.set_ylim(bottom=0)

axs[2].set_xlabel(r'Tilt Angle ($\degree$)', fontsize=15)
axs[1].set_ylabel('Counts (a.u.)', fontsize=15)
axs[0].xaxis.set_ticklabels([])
axs[1].xaxis.set_ticklabels([])
axs[0].set_title(r"Cs$_2$SnI$_6$  300 K", fontsize=16)

plt.savefig("traj_tilt_density_Cs2SnI6_300K.png", dpi=350, bbox_inches='tight')
print("saved 666_mlff/traj_tilt_density_Cs2SnI6_300K.png")
