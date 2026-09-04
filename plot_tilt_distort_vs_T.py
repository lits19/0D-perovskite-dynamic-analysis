"""Temperature trend of octahedral tilt and distortion in Cs2SnI6.

Reads 666_mlff/pdyna_{T}K.npz  (keys: 'tilt' (nframes, n_oct, 3),
'distort' (nframes, n_oct, 4)).  Averages over all octahedra and frames and
plots a 2x1 figure:
    (top)    three tilt axes  a / b / c
    (bottom) four distortion modes  Eg / T2g / T1u / T2u
vs temperature.

Tilt is signed and averages to ~0 in the cubic phase, so the tilt panel shows
the mean ABSOLUTE tilt <|theta|> (amplitude) by default; --signed plots the raw
signed mean.  Distortion modes are non-negative -> plain mean.

Run:
    python plot_tilt_distort_vs_T.py
    python plot_tilt_distort_vs_T.py --temps 300 350 450 600 --signed
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


NPZ = "/Users/tianshu/Documents/MLFF/Cs2SnI6/666_mlff/pdyna_{T}K.npz"
TILT_AXES = [r"$\theta_x$", r"$\theta_y$", r"$\theta_z$"]
DIST_MODES = [r"$E_g$", r"$T_{2g}$", r"$T_{1u}$", r"$T_{2u}$"]
TILT_COL = ["C0", "C1", "C2"]
DIST_COL = ["C3", "C4", "C5", "C6"]


def draw_glazer_stacked(axes, T, Corr, n_bins=100, colors=("C0", "C1", "C2"),
                        labels=(r"$\mathit{a}$", r"$\mathit{b}$", r"$\mathit{c}$")):
    """PDynA-style Glazer shade, one tilt axis per sub-panel (stacked a/b/c).
    line = normalised tilt distribution; shade = (tilt dist) x (correlation dist).
    axes: 3 stacked matplotlib axes;  T: (nf,n_oct,3);  Corr: (nf,n_oct,3)."""
    for i, ax in enumerate(axes):
        yt, be = np.histogram(T[:, :, i].ravel(), bins=n_bins, range=[-45, 45])
        bc = 0.5 * (be[1:] + be[:-1])
        yt = yt / yt.max()
        yc, _ = np.histogram(Corr[:, :, i].ravel(), bins=n_bins, range=[-45, 45])
        yc = yc / yc.max()
        ax.fill_between(bc, yt * yc, 0, facecolor=colors[i], alpha=0.35, interpolate=True)
        ax.plot(bc, yt, color=colors[i], lw=2.0)
        ax.text(0.03, 0.78, labels[i], transform=ax.transAxes, fontsize=14,
                style="italic", va="center", ha="center")
        ax.set_xlim([-45, 45])
        ax.set_xticks([-45, -30, -15, 0, 15, 30, 45])
        ax.set_yticks([])
        ax.set_ylim(bottom=0)
        if i < len(axes) - 1:                       # only bottom sub-panel keeps x ticks
            plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel(r"Tilt angle ($\degree$)", fontsize=12)


def _err(a, mode):
    """a: (nframes, n_oct, ncomp).  Returns (ncomp,) error bar of the mean."""
    per_oct = np.nanmean(a, axis=0)                          # (n_oct, ncomp): time mean per oct
    oct_std = np.nanstd(per_oct, axis=0)                     # spread across octahedra
    if mode == "octstd":
        return oct_std
    if mode == "sem":                                       # standard error of the overall mean
        n = np.sum(np.all(np.isfinite(per_oct), axis=1))
        return oct_std / np.sqrt(max(int(n), 1))
    if mode == "fluct":                                     # instantaneous fluctuation amplitude
        return np.nanstd(a.reshape(-1, a.shape[-1]), axis=0)
    return oct_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=int, nargs="+", default=[300, 350, 450, 600])
    ap.add_argument("--signed", action="store_true",
                    help="plot signed mean tilt instead of mean |tilt|")
    ap.add_argument("--err", choices=["sem", "octstd", "fluct"], default="octstd",
                    help="error bar: sem=standard error of the mean (default), "
                         "octstd=spread across octahedra, fluct=instantaneous std")
    ap.add_argument("--out",
                    default="/Users/tianshu/Documents/MLFF/Cs2SnI6/666_mlff/tilt_distort_vs_T.png")
    args = ap.parse_args()

    temps = sorted(args.temps)
    tilt_mean = np.full((len(temps), 3), np.nan)
    tilt_err = np.full((len(temps), 3), np.nan)
    dist_mean = np.full((len(temps), 4), np.nan)
    dist_err = np.full((len(temps), 4), np.nan)

    for i, T in enumerate(temps):
        f = NPZ.format(T=T)
        if not os.path.isfile(f):
            print(f"[skip] missing {f}")
            continue
        d = np.load(f, allow_pickle=True)
        tilt = np.asarray(d["tilt"], float)          # (nf, noct, 3)
        dist = np.asarray(d["distort"], float)       # (nf, noct, 4)
        tt = tilt if args.signed else np.abs(tilt)
        tilt_mean[i] = np.nanmean(tt, axis=(0, 1))
        dist_mean[i] = np.nanmean(dist, axis=(0, 1))
        tilt_err[i] = _err(tt, args.err)
        dist_err[i] = _err(dist, args.err)
        print(f"T={T}K  tilt={np.round(tilt_mean[i],4)}  dist={np.round(dist_mean[i],4)}  "
              f"err({args.err})={np.round(tilt_err[i],4)}")

    x = np.array(temps)
    # top: Glazer a/b/c stacked (x=tilt angle); bottom two: vs-T (x=temperature).
    # These two groups must not share x, so use nested gridspecs.
    fig = plt.figure(figsize=(5.2, 9.2))
    # two groups so the gaps can differ: Glazer<->ax1 gap = outer hspace (0.1),
    # ax1<->ax2 gap = inner hspace (0.02)
    outer = fig.add_gridspec(2, 1, height_ratios=[2.1, 3.6], hspace=0.18)
    gg = outer[0].subgridspec(3, 1, hspace=0.0)     # 3 stacked Glazer sub-panels
    gax = [fig.add_subplot(gg[k]) for k in range(3)]
    vv = outer[1].subgridspec(2, 1, hspace=0.02)    # vs-T panels
    ax1 = fig.add_subplot(vv[0])
    ax2 = fig.add_subplot(vv[1], sharex=ax1)

    dg = np.load(NPZ.format(T=300), allow_pickle=True)
    Tg = np.asarray(dg["tilt"], float)            # (nf, n_oct, 3)
    Cg = np.asarray(dg["tilt_corr"], float)       # (nf, n_oct, 1, 3)  1 = shell index
    if Cg.ndim == 4 and Cg.shape[2] == 1:
        Cg = Cg[:, :, 0, :]                       # -> (nf, n_oct, 3)
    draw_glazer_stacked(gax, Tg, Cg, colors=TILT_COL)
    gax[1].set_ylabel("Counts (a.u.)")
    #gax[0].set_title("300 K  tilt + correlation shade", fontsize=10)

    tlabel = "signed mean tilt" if args.signed else r"Avg. $|\theta|$"
    for j, (lab, c) in enumerate(zip(TILT_AXES, TILT_COL)):
        ax1.errorbar(x, tilt_mean[:, j], yerr=tilt_err[:, j], marker="o", ms=6,
                     lw=1.8, capsize=3, color=c, label=lab)
    ax1.set_ylabel(f"Tilting")
    ax1.legend(title="", fontsize=9, ncol=3, handletextpad=0.3, columnspacing=0.6)
    ax1.grid(alpha=0.3)
    ax1.set_xlabel("")
    plt.setp(ax1.get_xticklabels(), visible=False)   # temperature axis shown on ax2 only
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.25))

    for j, (lab, c) in enumerate(zip(DIST_MODES, DIST_COL)):
        ax2.errorbar(x, dist_mean[:, j], yerr=dist_err[:, j], marker="s", ms=6,
                     lw=1.8, capsize=3, color=c, label=lab)
    ax2.set_ylabel("Distortion")
    ax2.set_xlabel("Temperature (K)")
    ax2.legend(title="", loc="upper left", fontsize=9, ncol=4)
    ax2.set_ylim(0.02, 0.13)
    ax2.grid(alpha=0.3)
    ax2.yaxis.set_major_locator(MultipleLocator(0.03))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.015))
    ax2.set_xticks(x)

    fig.savefig(args.out, dpi=400, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
