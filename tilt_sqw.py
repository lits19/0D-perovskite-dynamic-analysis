"""Dynamical structure factor of the octahedral TILT field, S_tilt(q, omega).

This is the tilt-projected analogue of a phonon dispersion measured from MD: at
each q it shows the frequency spectrum of the collective libration wave, so its
intensity peaks trace the libration branch(es) that can be overlaid directly on
the SCPH phonon dispersion (same Fm-3m FCC path).

Method (no atomic trajectory needed -- reads the small tilt npy):
  A_a(q,t) = sum_i dtheta_i^a(t) exp(-2pi i g.s_i)     [g = 6q/(2pi/a), s_i = Sn frac.]
  S(q,w)  = < |FFT_t[ w(t) A_a(q,t) ]|^2 >  (Welch-averaged, summed over a=x,y,z)
Peaks in S(q,w) give omega_lib(q).

For the FULL phonon dispersion (all branches) from MD, use dynasor's SED on the
atomic trajectory instead -- that is a separate, complementary calculation.

Usage : python tilt_sqw.py [tilt_npy] [--poscar ..] [--dt-fs 20] [--fmax 7]
                           [--out-dir .]
"""
import argparse
import glob
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})


def find_poscar(explicit):
    if explicit:
        return explicit
    for c in ("SUPERCELL_666.vasp", "../SUPERCELL_666.vasp",
              "../../666_mlff/SUPERCELL_666.vasp"):
        if os.path.isfile(c):
            return c
    raise SystemExit("[error] SUPERCELL_666.vasp not found; pass --poscar")


def load_tilt(path):
    if path.endswith(".npz"):
        d = np.load(path)
        arr, dt_hint = d["tilt"], (10.0 * int(d["read_every"]) if "read_every" in d else None)
    else:
        arr, dt_hint = np.load(path), None
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise SystemExit(f"[error] expected (nframes, n_oct, 3), got {arr.shape}")
    return arr.transpose(2, 0, 1), dt_hint          # (3, nt, N)


def read_sn_frac(poscar):
    with open(poscar) as f:
        lines = f.readlines()
    counts = [int(x) for x in lines[6].split()]
    mode = lines[7].strip().lower()
    scale = float(lines[1])
    cell = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    pos = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(sum(counts))])
    frac = pos if mode.startswith("d") else pos @ np.linalg.inv(cell)
    return frac[counts[0]:counts[0] + counts[1]]     # Sn fractional (supercell)


def build_path(L=6):
    """Commensurate FCC path Gamma-X-W-L-Gamma in grid units (q = g/L * 2pi/a)."""
    hs = {"$\\Gamma$": (0, 0, 0), "X": (L, 0, 0), "W": (L, L // 2, 0),
          "L": (L // 2, L // 2, L // 2)}
    seq = ["$\\Gamma$", "X", "W", "L", "$\\Gamma$"]
    pts, ticks, labels = [], [], []
    for a, b in zip(seq[:-1], seq[1:]):
        A, B = np.array(hs[a]), np.array(hs[b])
        diff = B - A
        g = math.gcd(math.gcd(abs(diff[0]), abs(diff[1])), abs(diff[2])) or 1
        step = diff // g
        ticks.append(len(pts))
        labels.append(a)
        for t in range(g + (1 if b == seq[-1] else 0)):   # include final endpoint once
            pts.append(tuple(A + t * step))
    ticks.append(len(pts) - 1)
    labels.append(seq[-1])
    return np.array(pts), ticks, labels


def welch_psd(sig, nperseg, dt_s):
    """One-sided PSD of a complex signal, Welch-averaged (Hann, 50% overlap)."""
    n = len(sig)
    nperseg = min(nperseg, n)
    step = nperseg // 2
    win = np.hanning(nperseg)
    half = nperseg // 2
    acc = np.zeros(half)
    nseg = 0
    for s in range(0, n - nperseg + 1, step):
        X = np.fft.fft(win * sig[s:s + nperseg])
        P = (X * np.conj(X)).real
        folded = P[:half].copy()
        folded[1:] += P[:half:-1][:half - 1]          # fold negative frequencies
        acc += folded
        nseg += 1
    acc /= max(nseg, 1)
    freqs = np.arange(half) / (nperseg * dt_s)         # Hz
    return freqs, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tilt_npy", nargs="?", default=None)
    ap.add_argument("--poscar", default=None)
    ap.add_argument("--dt-fs", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=7.0, help="max frequency shown (THz)")
    ap.add_argument("--nseg", type=int, default=4, help="Welch segments (variance vs resolution)")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    npy = args.tilt_npy or (sorted(glob.glob("tilt*.npy")) or [None])[0]
    if npy is None:
        raise SystemExit("[error] no tilt*.npy in cwd; give a path")
    T = (re.search(r"(\d+)K", os.path.basename(npy)) or [None, "?"])[1]
    os.makedirs(args.out_dir, exist_ok=True)

    tilt, dt_hint = load_tilt(npy)
    dt_fs = dt_hint if dt_hint is not None else args.dt_fs
    dt_s = dt_fs * 1e-15
    _, nt, N = tilt.shape
    sn = read_sn_frac(find_poscar(args.poscar))
    dtilt = tilt - tilt.mean(axis=1, keepdims=True)
    print(f"loaded {npy}: nt={nt}, N={N}, dt={dt_fs:.1f} fs, T={T} K")

    pts, ticks, labels = build_path(L=6)
    nperseg = max(64, nt // args.nseg)
    print(f"path: {len(pts)} q-points, Welch nperseg={nperseg} "
          f"(df={1/(nperseg*dt_s)/1e12:.3f} THz)")

    Sqw = None
    freqs_THz = None
    for iq, g in enumerate(pts):
        ph = np.exp(-2j * np.pi * (sn @ g))            # (N,)
        col = None
        for a in range(3):                             # sum over x,y,z tilt components
            A = dtilt[a] @ ph                          # (nt,) complex collective amplitude
            fr, psd = welch_psd(A, nperseg, dt_s)
            col = psd if col is None else col + psd
        if Sqw is None:
            freqs_THz = fr / 1e12
            m = freqs_THz <= args.fmax
            freqs_THz = freqs_THz[m]
            Sqw = np.zeros((len(pts), m.sum()))
        Sqw[iq] = col[freqs_THz.size and slice(0, freqs_THz.size)]

    # peak (libration) frequency per q, ignoring the DC/quasi-elastic bin
    lo = np.searchsorted(freqs_THz, 0.15)
    peak = freqs_THz[lo + np.argmax(Sqw[:, lo:], axis=1)]

    # column-normalise for a clean dispersion image
    Snorm = Sqw / (Sqw[:, lo:].max(axis=1, keepdims=True) + 1e-30)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Snorm.T, origin="lower", aspect="auto", cmap="inferno",
                   extent=[0, len(pts) - 1, freqs_THz[0], freqs_THz[-1]],
                   vmin=0, vmax=1)
    ax.plot(np.arange(len(pts)), peak, "o", ms=4, color="cyan",
            label="peak (libration branch)")
    for t in ticks:
        ax.axvline(t, color="w", lw=0.6, alpha=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("frequency (THz)")
    ax.set_ylim(0, 2.0)
    ax.set_title(rf"Cs$_2$SnI$_6$ {T} K — tilt-projected $S(q,\omega)$ "
                 r"(column-normalised); peaks = libration dispersion")
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, label=r"$S_{\rm tilt}(q,\omega)$ (norm.)")
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"tilt_sqw_{T}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")

    np.savez(os.path.join(args.out_dir, f"tilt_sqw_{T}.npz"),
             q_grid=pts, ticks=ticks, labels=labels, freqs_THz=freqs_THz,
             Sqw=Sqw, peak_THz=peak, dt_fs=dt_fs)
    print(f"peak libration freq along path (THz): "
          f"min {peak.min():.2f}, max {peak.max():.2f}")
    print(f"at Gamma: {peak[0]:.2f} THz;  at X: {peak[ticks[1]]:.2f} THz")


if __name__ == "__main__":
    main()
