#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDynA octahedral tilting/distortion analysis for Cs2SnI6 supercell trajectories.

Usage
-----
    python pdyna_analysis.py <temp_K> [--seed S] [--read-every N] [--threads T]
    e.g.  python pdyna_analysis.py 300
          python pdyna_analysis.py 600 --seed 1 --read-every 1

Reads   ./SUPERCELL_666.vasp, ./INCAR, ./XDATCAR_<T>/XDATCAR
Writes  ./tilt_<T>K.npy, distort_<T>K.npy, tilt_corr_<T>K.npy, Benv_<T>K.npy
        ./pdyna_<T>K.npz   (bundle of all arrays + metadata)

Note on speed: the tilt/distortion kernel lives inside PDynA and is not
vectorized, so the only performance knobs are --read-every (fewer frames)
and --threads. Everything below is I/O robustness and reusability.

Created 2023-12-08 by tianshu; parameterized 2026-07.
"""
import argparse
import glob
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")   # headless: save every figure instead of showing;
                        # must be set before pdyna imports pyplot

import numpy as np

# ---- analysis physics (unchanged; tuned for this system) --------------------
TLIB = {
    'type 0':  np.array([-0.03596765, -5.7926216, -6.108742]),
    'type 1':  np.array([0.12824278, -5.9277034, 5.9476123]),
    'type 2':  np.array([0.09822315, 6.0677495, -6.1263957]),
    'type 3':  np.array([-0.07746969, 5.775201, 6.041878]),
    'type 4':  np.array([-5.8992033e+00, -6.1558331e-03, -6.2621160e+00]),
    'type 5':  np.array([-5.920039, 0.12105582, 5.852914]),
    'type 6':  np.array([5.889005, 0.13272977, -6.1773925]),
    'type 7':  np.array([6.1792936, -0.13027453, 5.8999553]),
    'type 8':  np.array([-5.8697762, -5.927197, -0.19155903]),
    'type 9':  np.array([-6.0770674, 5.9103384, -0.09176036]),
    'type 10': np.array([6.0253925, -5.9481206, -0.07620952]),
    'type 11': np.array([5.777682, 5.9908667, -0.0833529]),
}
NEW_SYS = {'fpg_val_BB': [[5, 13], [8, 12]],
           'fpg_val_BX': [[0.1, 8], [3, 6.6]]}

ALLOW_EQUIL = 0.05   # first fraction dropped as equilibration; set 0.0 for NVE
                     # trajectories that were already equilibrated separately.
POSCAR = "./SUPERCELL_666.vasp"
INCAR = "./INCAR"


def run(temp, read_every, threads):
    from pdyna.core import Trajectory   # lazy: keeps --help usable without pdyna

    xdatcar = f"./XDATCAR_{temp}/XDATCAR"
    for path in (POSCAR, INCAR, xdatcar):
        if not os.path.isfile(path):
            sys.exit(f"[error] missing input: {path}")

    traj = Trajectory("vasp", (POSCAR, xdatcar, INCAR))
    traj.dynamics(
        read_mode=1,                    # 1: equilibration mode
        uniname=f"{temp}K",
        allow_equil=ALLOW_EQUIL,
        read_every=read_every,
        preset=0,
        toggle_tilt_distort=True,
        structure_type=3,               # non-perovskite, initial config as reference
        multi_thread=threads,
        tilt_corr_NN1=True,             # first-NN tilt correlation (Glazer)
        structure_ref_NN1=TLIB,
        symm_n_fold=4,
        system_overwrite=NEW_SYS,
    )
    return traj


def extract_benv(traj):
    """Pull the B-site environment array; guarded because it depends on the
    BB/BX env calculation succeeding and touches a private attribute."""
    try:
        raw = traj._Benv_type
        return np.array([raw[k] for k in raw])[:, :, 0].T
    except (AttributeError, IndexError, KeyError) as e:
        print(f"[warn] Benv extraction skipped ({e.__class__.__name__})")
        return None

def save_all_figures(figdir, pre_png):
    """Collect every figure this run produced into figdir: (1) relocate PNGs
    PDynA auto-saved to cwd, (2) save any figures left open but not written."""
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    new = sorted(set(glob.glob("*.png")) - pre_png)
    for f in new:
        shutil.move(f, os.path.join(figdir, os.path.basename(f)))
    nums = plt.get_fignums()
    for n in nums:
        plt.figure(n).savefig(os.path.join(figdir, f"figure_{n:02d}.png"),
                              dpi=200, bbox_inches="tight")
        plt.close(n)
    print(f"[figures] {len(new)} auto-saved + {len(nums)} open -> {figdir}/")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("temp", type=int, help="temperature in K (selects XDATCAR_<T>)")
    p.add_argument("--seed", type=int, default=None, help="seed label for output names")
    p.add_argument("--read-every", type=int, default=2, help="read every N frames")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                   help="PDynA worker threads")
    args = p.parse_args()

    tag = f"{args.temp}K" + (f"_s{args.seed}" if args.seed is not None else "")
    pre_png = set(glob.glob("*.png"))          # snapshot before PDynA writes figures

    traj = run(args.temp, args.read_every, args.threads)
    save_all_figures(f"figs_{tag}", pre_png)

    tilt = np.asarray(traj.Tilting, dtype=np.float32)
    distort = np.asarray(traj.Distortion, dtype=np.float32)
    tilt_corr = np.asarray(traj.Tilting_Corr, dtype=np.float32)
    benv = extract_benv(traj)

    print(f"[shapes] tilt {tilt.shape}  distort {distort.shape}  "
          f"tilt_corr {tilt_corr.shape}"
          + (f"  Benv {benv.shape}" if benv is not None else ""))

    # backward-compatible individual arrays
    np.save(f"tilt_{tag}.npy", tilt)
    np.save(f"distort_{tag}.npy", distort)
    np.save(f"tilt_corr_{tag}.npy", tilt_corr)
    if benv is not None:
        np.save(f"Benv_{tag}.npy", benv.astype(np.float32))

    # self-describing bundle (recommended input for downstream analysis)
    bundle = dict(tilt=tilt, distort=distort, tilt_corr=tilt_corr,
                  temp=args.temp, read_every=args.read_every,
                  allow_equil=ALLOW_EQUIL)
    if benv is not None:
        bundle["Benv"] = benv.astype(np.float32)
    np.savez_compressed(f"pdyna_{tag}.npz", **bundle)
    print(f"[done] wrote tilt/distort/tilt_corr .npy + pdyna_{tag}.npz")


if __name__ == "__main__":
    main()
