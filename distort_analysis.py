"""Spatial/temporal analysis of octahedral DISTORTION correlations in Cs2SnI6.

Companion to tilt_analysis.py. Serves as the "negative control": distortion is
an intra-octahedral degree of freedom, expected to be far less spatially
correlated than the rigid-body tilt.

Input : distortion array from PDynA of shape (nframes, n_oct, K)
        -- distort_<T>K.npy or a pdyna_<T>K.npz bundle (key 'distort').
        K distortion amplitudes per octahedron (Eg/T2g-type modes). Unlike tilt,
        these are NON-NEGATIVE magnitudes, not a signed vector -- so we analyse
        correlations of distortion *magnitude* (large-here vs large-there), and
        cannot see sign/direction coupling. State this limitation in the paper:
        "no correlation in distortion magnitudes".

Two consequences vs tilt:
  * S(q) at Gamma is large (2-3): for a non-negative field the box-averaged
    magnitude "breathes" with temperature -- a trivial thermodynamic fluctuation,
    NOT spatial organisation. Gamma is annotated, not colour-scaled.
  * The cubic star average rotates q only (each amplitude treated as a scalar
    field); it does NOT permute components the way the tilt vector does. The
    rigorous point-group-invariant object is the total (sum over components).

Usage : python distort_analysis.py [distort_npy] [--poscar SUPERCELL_666.vasp]
                                    [--dt-fs 10.0] [--out-dir .]
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


def find_poscar(explicit):
    if explicit:
        return explicit
    for cand in ("SUPERCELL_666.vasp", "../SUPERCELL_666.vasp",
                 "../../666_mlff/SUPERCELL_666.vasp"):
        if os.path.isfile(cand):
            return cand
    raise SystemExit("[error] SUPERCELL_666.vasp not found; pass --poscar")


def load_distort(path):
    """Return distortion as (K, nt, N) from a (nt, N, K) .npy or a .npz bundle."""
    if path.endswith(".npz"):
        d = np.load(path)
        arr = d["distort"]
        dt_hint = 10.0 * int(d["read_every"]) if "read_every" in d else None
    else:
        arr = np.load(path)
        dt_hint = None
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise SystemExit(f"[error] expected (nframes, n_oct, K), got {arr.shape}")
    return arr.transpose(2, 0, 1), dt_hint          # -> (K, nt, N)


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


def star_average_Sq_scalar(dd, sn_frac, L=6):
    """Per-component cubic-star-averaged S_c(q) on the commensurate L^3 grid.
    Each distortion amplitude is treated as an independent SCALAR field: the
    star average rotates q only (no component permutation)."""
    K = dd.shape[0]
    N = sn_frac.shape[0]
    X = sn_frac * L
    g = np.arange(2 * L)
    grid = np.array(list(itertools.product(g, g, g)))
    phase = np.exp(-2j * np.pi * (grid / L) @ X.T)          # (nq, N)
    var = dd.var(axis=(1, 2))
    S3 = np.stack([(np.abs(phase @ dd[c].T) ** 2).mean(axis=1)
                   .reshape(2 * L, 2 * L, 2 * L) / N / var[c] for c in range(K)])
    Ss = np.zeros_like(S3)
    nops = 0
    for p in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            Rg = np.stack([(s[k] * grid[:, p[k]]) % (2 * L) for k in range(3)], axis=1)
            for c in range(K):                              # scalar: rotate q, keep c
                Ss[c] += S3[c][Rg[:, 0], Rg[:, 1], Rg[:, 2]].reshape(2 * L, 2 * L, 2 * L)
            nops += 1
    return Ss / nops, var


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("distort_npy", nargs="?", default=None,
                    help="distort_<T>K.npy or pdyna_<T>K.npz (default: first distort*.npy in cwd)")
    ap.add_argument("--poscar", default=None)
    ap.add_argument("--dt-fs", type=float, default=10.0)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    npy = args.distort_npy
    if npy is None:
        hits = sorted(glob.glob("distort*.npy"))
        if not hits:
            raise SystemExit("[error] no distort*.npy in cwd; give a path")
        npy = hits[0]
    m = re.search(r"(\d+)K", os.path.basename(npy))
    T = m.group(1) if m else "?"
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)

    # ---------------- load ----------------
    dd_raw, dt_hint = load_distort(npy)
    dt_fs = dt_hint if dt_hint is not None else args.dt_fs
    K, nt, N = dd_raw.shape
    cell, sn_frac = read_supercell(find_poscar(args.poscar))
    L = cell[0, 0]
    a_conv = L / 6.0
    labels = [r"$\mathrm{E_g}$", r"$\mathrm{T_{2g}}$",r"$\mathrm{T_{1u}}$", r"$\mathrm{T_{2u}}$"] + ["total"]
    cmap_lines = plt.cm.viridis(np.linspace(0, 0.85, K))
    colors = list(cmap_lines) + ["k"]
    print(f"loaded {npy}: {K} distortion modes, {nt} frames, {N} octahedra; "
          f"T={T} K, dt={dt_fs:.1f} fs")

    mean_t = dd_raw.mean(axis=1)                            # (K, N) mean magnitude
    dd = dd_raw - mean_t[:, None, :]                        # fluctuation about own mean
    var = dd.var(axis=(1, 2))

    # ---------------- pair shells ----------------
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
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for c in range(K + 1):
        means, errs = [], []
        for (ii, jj) in shell_pairs:
            vals = []
            for b in range(nblock):
                sl = slice(b * bs, (b + 1) * bs)
                if c < K:
                    x = dd[c, sl]
                    r = (x[:, ii] * x[:, jj]).mean() / x.var()
                else:
                    num = sum((dd[k, sl][:, ii] * dd[k, sl][:, jj]).mean() for k in range(K))
                    den = sum(dd[k, sl].var() for k in range(K))
                    r = num / den
                vals.append(r)
            vals = np.array(vals)
            means.append(vals.mean())
            errs.append(vals.std(ddof=1) / np.sqrt(nblock))
        ax.errorbar(shell_d, means, yerr=errs, marker="o", ms=5, capsize=3,
                    color=colors[c], label=labels[c],
                    lw=1.8 if c == K else 1, alpha=1 if c == K else 0.7)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel(r"Neighbour Distance ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Distortion Correlation")
    ax.yaxis.set_major_locator(MultipleLocator(0.002))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))
    #ax.set_title(rf"Cs$_2$SnI$_6$ {T} K — equal-time distortion correlation vs fcc shell")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out(f"distort_fig1_CR_shells_{T}.png"), dpi=200)
    plt.close(fig)
    print("fig1 done")

    # ---------------- Fig 2: cubic-star-averaged S(q) map, qz=0 slice ----------------
    Ss, _ = star_average_Sq_scalar(dd, sn_frac, L=6)
    hk = np.arange(-6, 7)
    wrap = hk % 12
    ncol = K # + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.1 * ncol, 4.1), constrained_layout=True)
    ext = [-1 - 1 / 12, 1 + 1 / 12] * 2
    vmin, vmax = 0.80, 1.25
    for c in range(K):
        sl = Ss[c][np.ix_(wrap, wrap, [0])][:, :, 0]
        im = axes[c].imshow(sl.T, origin="lower", extent=ext, cmap="magma",
                            vmin=vmin, vmax=vmax)
        axes[c].set_title(rf"$S(q)/\langle\delta d^2\rangle$ — {labels[c]}")
        axes[c].annotate(rf"$\Gamma$={sl[6, 6]:.2f}", (0.03, 0.03),
                         xycoords="axes fraction", color="w", fontsize=9)
        axes[c].yaxis.set_major_locator(MultipleLocator(0.5))
        axes[c].yaxis.set_minor_locator(MultipleLocator(0.25))
        axes[c].xaxis.set_major_locator(MultipleLocator(0.5))
        axes[c].xaxis.set_minor_locator(MultipleLocator(0.25))
    """
    tot = sum(Ss[c][np.ix_(wrap, wrap, [0])][:, :, 0] for c in range(K)) / K
    im = axes[K].imshow(tot.T, origin="lower", extent=ext, cmap="magma",
                        vmin=vmin, vmax=vmax)
    axes[K].set_title(r"total $\frac{1}{K}\Sigma_c S_c(q)$")
    axes[K].annotate(rf"$\Gamma$={tot[6, 6]:.2f}", (0.03, 0.03),
                     xycoords="axes fraction", color="w", fontsize=9)
    axes[K].yaxis.set_major_locator(MultipleLocator(0.5))
    axes[K].yaxis.set_minor_locator(MultipleLocator(0.25))
    axes[K].xaxis.set_major_locator(MultipleLocator(0.5))
    axes[K].xaxis.set_minor_locator(MultipleLocator(0.25))
    """

    for axx in axes:
        axx.set_xlabel(r"$q_x$ ($2\pi/a$)")
        axx.scatter([0], [0], marker="+", c="cyan", s=70)
        axx.scatter([1, -1, 0, 0], [0, 0, 1, -1], marker="x", c="lime", s=40)
    axes[0].set_ylabel(r"$q_y$ ($2\pi/a$)")

    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap="magma"
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.01)
    cbar.set_ticks([0.8, 0.9, 1.0, 1.1, 1.2])
    #fig.suptitle(rf"Cs$_2$SnI$_6$ {T} K — distortion-magnitude structure factor "
    #             r"($q_z$=0, cubic-star averaged; $\Gamma$ saturates, value annotated)", y=1.05)
    fig.savefig(out(f"distort_fig2_Sq_maps_{T}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")

    # ---------------- Fig 2b: total S(q) along high-symmetry directions ----------------
    def Ssym_path(pts):
        return [sum(Ss[c][h % 12, k % 12, l % 12] for c in range(K)) / K
                for (h, k, l) in pts]
    paths = {r"$\Gamma\to$X (00$\xi$)": [(0, 0, l) for l in range(7)],
             r"$\Gamma\to$K (ξξ0)": [(k, k, 0) for k in range(7)],
             r"$\Gamma\to$L (ξξξ)": [(l, l, l) for l in range(4)]}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (lbl, pts), col in zip(paths.items(), ["#1f77b4", "#d62728", "#2ca02c"]):
        xs = [np.linalg.norm(p) / 6 for p in pts]
        ys = Ssym_path(pts)
        ax.plot(xs, ys, "o-", color=col, label=lbl, ms=5)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlabel(r"$|q|$ ($2\pi/a$)")
    ax.set_ylabel(r"$S_{\rm distort}(q)/\langle\delta d^2\rangle$ (total, symmetrized)")
    ax.set_title(rf"Cs$_2$SnI$_6$ {T} K — distortion structure factor, high-sym "
                 r"(note: $\Gamma$ off-scale = breathing, not correlation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out(f"distort_fig2b_Sq_path_{T}.png"), dpi=200)
    plt.close(fig)
    print("fig2b done")

    # ---------------- Fig 3: time-lagged correlation, 1st shell ----------------
    maxlag = min(nt // 3, 500)
    ii1, jj1 = shell_pairs[0]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ii1), size=min(1500, len(ii1)), replace=False)
    ii1s, jj1s = ii1[sel], jj1[sel]
    lags = np.arange(maxlag)
    tps = lags * dt_fs / 1000.0
    auto = np.zeros((K, maxlag))
    cross = np.zeros((K, maxlag))
    for c in range(K):
        x = dd[c]
        v = x.var()
        for k, lag in enumerate(lags):
            a = x[: nt - lag]
            b = x[lag:]
            auto[c, k] = (a * b).mean() / v
            cross[c, k] = (a[:, ii1s] * b[:, jj1s]).mean() / v
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for c in range(K):
        ax[0].plot(tps, auto[c], color=colors[c], label=f"auto {labels[c]}")
        ax[1].plot(tps, cross[c], color=colors[c], label=f"cross-1NN {labels[c]}")
    ax[1].plot(tps, cross.mean(0), "k", lw=2, label="cross-1NN avg")
    for a_ in ax:
        a_.axhline(0, color="gray", lw=0.8)
        a_.set_xlabel("lag (ps)")
        a_.legend(fontsize=8, ncol=2)
    ax[0].yaxis.set_major_locator(MultipleLocator(0.2))
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.1))

    ax[0].set_ylabel(r"$\frac{\langle\delta d_i(0)\,\delta d_i(t)\rangle}{\langle\delta d^2\rangle}$", fontsize=12)
    ax[1].set_ylabel(r"$\frac{\langle\delta d_i(0)\,\delta d_j(t)\rangle}{\langle\delta d^2\rangle}$", fontsize=12)
    #ax[0].set_title("on-site distortion autocorrelation")
    #ax[1].set_title("1st-shell distortion cross correlation")
    fig.tight_layout()
    fig.savefig(out(f"distort_fig3_time_corr_{T}.png"), dpi=400)
    plt.close(fig)
    print("fig3 done")

    # ---------------- Fig 4: magnitude distribution + convergence ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c in range(K):
        ax[0].hist(mean_t[c], bins=50, alpha=0.5, color=colors[c],
                   label=f"{labels[c]} (mean={mean_t[c].mean():.3f})")
    ax[0].set_xlabel("time-averaged distortion magnitude per octahedron")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"per-mode distortion magnitude (dynamic σ≈{np.sqrt(var.mean()):.3f})")
    ax[0].legend(fontsize=8, ncol=2)
    for c in range(K):
        r_halves = []
        for sl in (slice(0, nt // 2), slice(nt // 2, nt)):
            x = dd_raw[c, sl] - dd_raw[c, sl].mean(axis=0)
            r_halves.append((x[:, ii1] * x[:, jj1]).mean() / x.var())
        ax[1].plot([1, 2], r_halves, "o-", color=colors[c], label=labels[c])
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set_xticks([1, 2], ["1st half", "2nd half"])
    ax[1].set_ylabel("1st-shell C")
    ax[1].set_title("convergence check")
    ax[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out(f"distort_fig4_check_{T}.png"), dpi=400)
    plt.close(fig)
    print("fig4 done")

    print("\n--- summary ---")
    print(f"first-shell C per mode: "
          + " ".join(f"{(dd[c][:, ii1] * dd[c][:, jj1]).mean() / dd[c].var():+.4f}"
                     for c in range(K)))
    print(f"sigma_dyn per mode: {np.sqrt(var)}")


if __name__ == "__main__":
    main()
