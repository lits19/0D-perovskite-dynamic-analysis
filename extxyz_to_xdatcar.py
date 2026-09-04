#!/usr/bin/env python3
"""
Convert a multi-frame (possibly multi-trajectory) extxyz file into VASP
XDATCAR file(s).

Frames are grouped by (chemical formula, number of atoms). Frames belonging
to the same group are written into a single multi-configuration XDATCAR
(the standard VASP format for MD trajectories / NEB-like frame sequences).
If the extxyz file actually contains several *different* systems (e.g. you
concatenated several independent trajectories with different compositions),
each distinct system is written to its own XDATCAR_<idx>_<formula> file.

Usage:
    python extxyz_to_xdatcar.py input.extxyz [--outdir OUTDIR] [--prefix PREFIX]

Requires: ase  (pip install ase)
"""
import argparse
import os
from collections import defaultdict

from ase.io import read, write


def group_key(atoms):
    """Group frames by composition + atom count (order-sensitive formula)."""
    return (atoms.get_chemical_formula(mode="hill"), len(atoms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="input extxyz file")
    parser.add_argument("--outdir", default=".", help="output directory (default: current dir)")
    parser.add_argument("--prefix", default="XDATCAR", help="output filename prefix (default: XDATCAR)")
    args = parser.parse_args()

    frames = read(args.input, index=":")
    print(f"Read {len(frames)} frames from {args.input}")

    groups = defaultdict(list)
    order = []
    for atoms in frames:
        key = group_key(atoms)
        if key not in groups:
            order.append(key)
        groups[key].append(atoms)

    os.makedirs(args.outdir, exist_ok=True)

    if len(groups) == 1:
        key = order[0]
        outpath = os.path.join(args.outdir, args.prefix)
        write(outpath, groups[key], format="vasp-xdatcar")
        print(f"Single trajectory detected ({len(groups[key])} frames, {key[0]}) -> {outpath}")
    else:
        print(f"Detected {len(groups)} distinct systems (different composition/atom count).")
        print("Writing one XDATCAR per system:")
        for i, key in enumerate(order):
            formula, natoms = key
            outpath = os.path.join(args.outdir, f"{args.prefix}_{i:03d}_{formula}")
            write(outpath, groups[key], format="vasp-xdatcar")
            print(f"  [{i}] {formula} ({natoms} atoms), {len(groups[key])} frames -> {outpath}")


if __name__ == "__main__":
    main()