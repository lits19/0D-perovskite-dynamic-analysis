"""Spatial/temporal analysis of octahedral tilt correlations in Cs2SnI6.

Input  : a tilt array from PDynA of shape (nframes, n_oct, 3)
         -- either tilt_<T>K.npy  or a pdyna_<T>K.npz bundle (key 'tilt').
         Each trajectory frame, each octahedron carries 3 tilt components
         (rotation about pseudo-cubic x/y/z); they are read as the 3 channels.
         Sn equilibrium positions come from SUPERCELL_666.vasp (same octahedron
         order as the tilt array -- verified by the fcc shell pair counts below).

Usage  : python tilt_analysis.py [tilt_npy] [--poscar SUPERCELL_666.vasp]
                                  [--dt-fs 10.0] [--out-dir .]
         python tilt_analysis.py                       # first tilt*.npy in cwd
         python tilt_analysis.py ./666_mlff/tilt_300K.npy

Outputs (tagged by temperature): fig1_CR_shells_<T>.png, fig2_Sq_maps_<T>.png,
         fig2b_Sq_path_<T>.png, fig3_time_corr_<T>.png, fig4_static_check_<T>.png
"""
import argparse
import glob
import itertools
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
COLORS = ["#1f77b4", "#2ca02c", "#d62728", "k"]
LABELS = ["x", "y", "z", "avg"]


def find_poscar(explicit):
    if explicit:
        return explicit
    for cand in ("SUPERCELL_666.vasp", "../SUPERCELL_666.vasp",
                 "../../666_mlff/SUPERCELL_666.vasp"):
        if os.path.isfile(cand):
            return cand
    raise SystemExit("[error] SUPERCELL_666.vasp not found; pass --poscar")


def load_tilt(path):
    """Return tilt as (3, nt, N) from a (nt, N, 3) .npy or a .npz bundle."""
    if path.endswith(".npz"):
        d = np.load(path)
        arr = d["tilt"]
        dt_hint = 10.0 * int(d["read_every"]) if "read_every" in d else None
    else:
        arr = np.load(path)
        dt_hint = None
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise SystemExit(f"[error] expected (nframes, n_oct, 3), got {arr.shape}")
    return arr.transpose(2, 0, 1), dt_hint          # -> (3, nt, N)


def read_supercell(poscar):
    with open(poscar) as f:
        lines = f.readlines()
    scale = float(lines[1])
    cell = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    counts = [int(x) for x in lines[6].split()]             # Cs Sn I
    mode = lines[7].strip().lower()
    pos = np.array([[float(x) for x in lines[8 + i].split()[:3]]
                    for i in range(sum(counts))])
    frac = pos if mode.startswith("d") else pos @ np.linalg.inv(cell)
    sn_frac = frac[counts[0]:counts[0] + counts[1]]
    return cell, sn_frac


def star_average_Sq(dtilt, sn_frac, L=6):
    """Cubic-star-averaged S_alpha(q) on the commensurate L^3 grid (grid units)."""
    N = sn_frac.shape[0]
    X = sn_frac * L
    g = np.arange(2 * L)
    grid = np.array(list(itertools.product(g, g, g)))
    phase = np.exp(-2j * np.pi * (grid / L) @ X.T)          # (nq, N)
    var = dtilt.var(axis=(1, 2))
    S3 = np.stack([(np.abs(phase @ dtilt[c].T) ** 2).mean(axis=1)
                   .reshape(2 * L, 2 * L, 2 * L) / N / var[c] for c in range(3)])
    Ss = np.zeros_like(S3)
    for p in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            Rg = np.stack([(s[k] * grid[:, p[k]]) % (2 * L) for k in range(3)], axis=1)
            for a in range(3):
                Ss[a] += S3[p[a]][Rg[:, 0], Rg[:, 1], Rg[:, 2]].reshape(2 * L, 2 * L, 2 * L)
    return Ss / 48.0                                        # (3, 2L, 2L, 2L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tilt_npy", nargs="?", default=None,
                    help="tilt_<T>K.npy or pdyna_<T>K.npz (default: first tilt*.npy in cwd)")
    ap.add_argument("--poscar", default=None)
    ap.add_argument("--dt-fs", type=float, default=10.0,
                    help="frame spacing in the array (fs), for the time axis in ps")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    npy = args.tilt_npy
    if npy is None:
        hits = sorted(glob.glob("tilt*.npy"))
        if not hits:
            raise SystemExit("[error] no tilt*.npy in cwd; give a path")
        npy = hits[0]
    m = re.search(r"(\d+)K", os.path.basename(npy))
    T = m.group(1) if m else "?"
    os.makedirs(args.out_dir, exist_ok=True)

    def out(name):
        return os.path.join(args.out_dir, name)

    # ---------------- load ----------------
    tilt, dt_hint = load_tilt(npy)
    dt_fs = dt_hint if dt_hint is not None else args.dt_fs
    ncomp, nt, N = tilt.shape
    cell, sn_frac = read_supercell(find_poscar(args.poscar))
    L = cell[0, 0]
    a_conv = L / 6.0
    print(f"loaded {npy}: {ncomp} components, {nt} frames, {N} octahedra; "
          f"T={T} K, dt={dt_fs:.1f} fs, box L={L:.3f} A")

    mean_t = tilt.mean(axis=1)                              # (3, N) static tilt
    dtilt = tilt - mean_t[:, None, :]                       # (3, nt, N)
    var = dtilt.var(axis=(1, 2))

    # ---------------- pair shells (PBC minimal image) ----------------
    d = sn_frac[None, :, :] - sn_frac[:, None, :]
    d -= np.round(d)
    dist = np.linalg.norm(d @ cell, axis=-1)
    iu = np.triu_indices(N, k=1)
    dr = dist[iu]
    shell_d = np.array([a_conv / np.sqrt(2), a_conv, a_conv * np.sqrt(1.5),
                        a_conv * np.sqrt(2), a_conv * np.sqrt(2.5), a_conv * np.sqrt(3)])
    shell_pairs = []
    for sd in shell_d:
        mask = np.abs(dr - sd) < 0.35
        shell_pairs.append((iu[0][mask], iu[1][mask]))
        print(f"  shell d={sd:.2f} A: {mask.sum()} pairs")

    # ---------------- Fig 1: C(R) with block-averaged error bars ----------------
    nblock = 5
    bs = nt // nblock
    fig, ax = plt.subplots(figsize=(5., 4.2))
    for c in range(4):
        means, errs = [], []
        for (ii, jj) in shell_pairs:
            vals = []
            for b in range(nblock):
                sl = slice(b * bs, (b + 1) * bs)
                if c < 3:
                    x = dtilt[c, sl]
                    r = (x[:, ii] * x[:, jj]).mean() / x.var()
                else:
                    num = sum((dtilt[k, sl][:, ii] * dtilt[k, sl][:, jj]).mean()
                              for k in range(3))
                    den = sum(dtilt[k, sl].var() for k in range(3))
                    r = num / den
                vals.append(r)
            vals = np.array(vals)
            means.append(vals.mean())
            errs.append(vals.std(ddof=1) / np.sqrt(nblock))
        ax.errorbar(shell_d, means, yerr=errs, marker="o", ms=5, capsize=3,
                    color=COLORS[c], label=LABELS[c],
                    lw=1.5 if c == 3 else 1, alpha=1 if c == 3 else 0.75)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel(r"Neighbour Distance ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Tilting Correlation")
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    #ax.set_title(rf"Cs$_2$SnI$_6$ {T} K — equal-time tilt correlation vs fcc shell")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out(f"fig1_CR_shells_{T}.png"), dpi=200)
    plt.close(fig)
    print("fig1 done")

    # ---------------- Fig 2: cubic-star-averaged S(q) map, qz=0 slice ----------------
    Ss = star_average_Sq(dtilt, sn_frac, L=6)
    hk = np.arange(-6, 7)
    wrap = hk % 12
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.1), constrained_layout=True)
    ext = [-1 - 1 / 12, 1 + 1 / 12] * 2
    vmin, vmax = 0.75, 1.30
    for c in range(3):
        sl = Ss[c][np.ix_(wrap, wrap, [0])][:, :, 0]
        im = axes[c].imshow(sl.T, origin="lower", extent=ext, cmap="magma",
                            vmin=vmin, vmax=vmax)
        axes[c].set_title(rf"$S_{LABELS[c]}(q)/\langle\delta\theta^2\rangle$")
        axes[c].yaxis.set_major_locator(MultipleLocator(0.5))
        axes[c].yaxis.set_minor_locator(MultipleLocator(0.25))
        axes[c].xaxis.set_major_locator(MultipleLocator(0.5))
        axes[c].xaxis.set_minor_locator(MultipleLocator(0.25))
    tot = sum(Ss[c][np.ix_(wrap, wrap, [0])][:, :, 0] for c in range(3)) / 3
    im = axes[3].imshow(tot.T, origin="lower", extent=ext, cmap="magma",
                        vmin=vmin, vmax=vmax)
    axes[3].set_title(r"$\frac{1}{3}\sum_\alpha S_\alpha(q)$ (total)")
    axes[3].yaxis.set_major_locator(MultipleLocator(0.5))
    axes[3].yaxis.set_minor_locator(MultipleLocator(0.25))
    axes[3].xaxis.set_major_locator(MultipleLocator(0.5))
    axes[3].xaxis.set_minor_locator(MultipleLocator(0.25))

    for axx in axes:
        axx.set_box_aspect(1)
        axx.set_xlabel(r"$q_x$ ($2\pi/a$)")
        axx.scatter([0], [0], marker="+", c="cyan", s=70)
        axx.scatter([1, -1, 0, 0], [0, 0, 1, -1], marker="x", c="lime", s=40)
    axes[0].set_ylabel(r"$q_y$ ($2\pi/a$)")
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    #fig.suptitle(rf"Cs$_2$SnI$_6$ {T} K — tilt structure factor, ($q_x,q_y$,0), "
    #             r"cubic-star averaged.  + = $\Gamma$,  x = X", y=1.05)
    fig.savefig(out(f"fig2_Sq_maps_{T}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")

    # ---------------- Fig 2b: S(q) along high-symmetry directions ----------------
    def Ssym_path(pts):
        return [sum(Ss[c][h % 12, k % 12, l % 12] for c in range(3)) / 3
                for (h, k, l) in pts]
    paths = {r"$\Gamma\to$X (00$\xi$)": [(0, 0, l) for l in range(7)],
             r"$\Gamma\to$K (ξξ0)": [(k, k, 0) for k in range(7)],
             r"$\Gamma\to$L (ξξξ)": [(l, l, l) for l in range(4)]}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (lbl, pts), col in zip(paths.items(), ["#1f77b4", "#d62728", "#2ca02c"]):
        xs = [np.linalg.norm(p) / 6 for p in pts]
        ax.plot(xs, Ssym_path(pts), "o-", color=col, label=lbl, ms=5)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlabel(r"$|q|$ ($2\pi/a$)")
    ax.set_ylabel(r"$S_{\rm tilt}(q)/\langle\delta\theta^2\rangle$ (total, symmetrized)")
    ax.set_title(rf"Cs$_2$SnI$_6$ {T} K — tilt structure factor, high-sym directions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out(f"fig2b_Sq_path_{T}.png"), dpi=200)
    plt.close(fig)
    print("fig2b done")

    # ---------------- Fig 3: time-lagged correlation, 1st shell ----------------
    maxlag = min(nt // 3, 500)
    ii1, jj1 = shell_pairs[0]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ii1), size=min(1500, len(ii1)), replace=False)
    ii1s, jj1s = ii1[sel], jj1[sel]
    lags = np.arange(maxlag)
    tps = lags * dt_fs / 1000.0                             # ps
    auto = np.zeros((3, maxlag))
    cross = np.zeros((3, maxlag))
    for c in range(3):
        x = dtilt[c]
        v = x.var()
        for k, lag in enumerate(lags):
            a = x[: nt - lag]
            b = x[lag:]
            auto[c, k] = (a * b).mean() / v
            cross[c, k] = (a[:, ii1s] * b[:, jj1s]).mean() / v
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for c in range(3):
        ax[0].plot(tps, auto[c], color=COLORS[c], label=f"auto {LABELS[c]}")
        ax[1].plot(tps, cross[c], color=COLORS[c], label=f"cross-1NN {LABELS[c]}")
    ax[1].plot(tps, cross.mean(0), "k", lw=2, label="cross-1NN avg")
    for a_ in ax:
        a_.axhline(0, color="gray", lw=0.8)
        a_.set_xlabel("lag (ps)")
        a_.legend(fontsize=9)
    ax[0].yaxis.set_major_locator(MultipleLocator(0.5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.25))
    ax[0].set_ylabel(r"$\frac{\langle\delta\theta_i(0)\,\delta\theta_i(t)\rangle}{\langle\delta\theta^2\rangle}$", fontsize=12)
    ax[1].set_ylabel(r"$\frac{\langle\delta\theta_i(0)\,\delta\theta_j(t)\rangle}{\langle\delta\theta^2\rangle}$", fontsize=12)
    #ax[0].set_title("on-site autocorrelation")
    #ax[1].set_title("1st-shell cross correlation")
    fig.tight_layout()
    fig.savefig(out(f"fig3_time_corr_{T}.png"), dpi=400)
    plt.close(fig)
    print("fig3 done")

    # ---------------- Fig 4: static (frozen) tilt check + convergence ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for c in range(3):
        ax[0].hist(mean_t[c], bins=60, alpha=0.55, color=COLORS[c],
                   label=f"{LABELS[c]}  (std={mean_t[c].std():.2f}°)")
    ax[0].set_xlabel("time-averaged tilt per octahedron (deg)")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"static tilt vs dynamic σ≈{np.sqrt(var.mean()):.2f}°")
    ax[0].legend(fontsize=9)
    for c in range(3):
        r_halves = []
        for sl in (slice(0, nt // 2), slice(nt // 2, nt)):
            x = tilt[c, sl] - tilt[c, sl].mean(axis=0)
            r_halves.append((x[:, ii1] * x[:, jj1]).mean() / x.var())
        ax[1].plot([1, 2], r_halves, "o-", color=COLORS[c], label=LABELS[c])
    ax[1].set_xticks([1, 2], ["1st half", "2nd half"])
    ax[1].set_ylabel("1st-shell C")
    ax[1].set_title("convergence check")
    ax[1].legend(fontsize=9)
    lay = np.abs(sn_frac[:, 2] - sn_frac[:, 2].min()) < 0.02
    sc = ax[2].scatter(sn_frac[lay, 0], sn_frac[lay, 1], c=mean_t[1][lay],
                       cmap="coolwarm", vmin=-3, vmax=3, s=90)
    ax[2].set_title("static tilt$_y$ map, bottom (001) layer")
    ax[2].set_xlabel("frac x")
    ax[2].set_ylabel("frac y")
    plt.colorbar(sc, ax=ax[2], label="deg")
    fig.tight_layout()
    fig.savefig(out(f"fig4_static_check_{T}.png"), dpi=200)
    plt.close(fig)
    print("fig4 done")

    print("\n--- summary ---")
    print(f"sigma_dyn per comp (deg): {np.sqrt(var)}")
    print(f"static std per comp (deg): {mean_t.std(axis=1)}")


if __name__ == "__main__":
    main()
