"""Full collective-coordinate decomposition of FC3 for Cs2SnI6.

Transforms the cubic anharmonicity (FC3) from atomic Cartesian displacements into
each octahedron's collective coordinates {rotation(3), translation(3),
distortion(12)} (Sn/Cs: 3 translation each, no rotation), then measures how much
of the total cubic anharmonicity has octahedral ROTATION (libration) on >=1 leg,
split into intra- vs inter-octahedral.

Because a rotation coordinate is a coherent sum over the 6 I atoms, the collective
FC3 element Phi3(Q_A,Q_B,Q_C) is accumulated over ALL triplets sharing the same
octahedron/site triple BEFORE squaring. The per-octahedron transform is
orthonormal, so sum|Phi3_collective|^2 must equal sum|Phi3_cartesian|^2 -- printed
as a self-check.

Reports the fraction of cubic anharmonicity with rotation on >=1 leg (the
libration-mediated anharmonic scattering), and how much of it is inter-octahedral
(= anharmonic tilting interplay).

Run: python project_fc3_channels.py [FORCE_CONSTANTS_3RD] [--poscar POSCAR_prim_relaxed]
"""
import argparse
import numpy as np
from collections import defaultdict


def read_cell(poscar):
    with open(poscar) as f:
        L = f.readlines()
    scale = float(L[1])
    latt = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)]) * scale
    cnts = [int(x) for x in L[6].split()]
    syms = L[5].split()
    mode = L[7].strip().lower()
    pos = np.array([[float(x) for x in L[8 + i].split()[:3]] for i in range(sum(cnts))])
    frac = pos if mode.startswith("d") else pos @ np.linalg.inv(latt)
    species = [s for s, n in zip(syms, cnts) for _ in range(n)]
    return latt, frac, species


def octahedron_basis(latt, frac, species):
    """18x18 orthonormal collective basis for the 6 I of the reference octahedron.
    Rows 0-2 = rotation, 3-5 = translation, 6-17 = distortion. Returns basis and a
    boolean mask of which rows are rotation, plus the list of I unit-cell indices."""
    sn = species.index("Sn")
    i_atoms = [k for k, s in enumerate(species) if s == "I"]
    nI = len(i_atoms)
    d = {}
    for k in i_atoms:
        df = frac[k] - frac[sn]
        df -= np.round(df)
        d[k] = df @ latt
    axes = np.eye(3)
    vecs = []
    for a in range(3):                       # rotation
        v = np.zeros(3 * nI)
        for m, k in enumerate(i_atoms):
            v[3 * m:3 * m + 3] = np.cross(axes[a], d[k])
        vecs.append(v)
    for b in range(3):                       # translation
        v = np.zeros(3 * nI)
        v[b::3] = 1.0
        vecs.append(v)
    # orthonormalise the 6, then complete with the 12-dim null space
    B = np.array(vecs)
    Q, _ = np.linalg.qr(B.T)                  # Q: 18x6 orthonormal columns
    U6 = Q.T                                   # 6x18
    # null space (distortion) via SVD of U6
    u, s, vt = np.linalg.svd(U6)
    dist = vt[6:]                              # 12x18, orthogonal to the 6
    U = np.vstack([U6, dist])                  # 18x18
    is_rot = np.array([True] * 3 + [False] * 15)
    return U, is_rot, i_atoms


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fc3", nargs="?", default="FORCE_CONSTANTS_3RD")
    ap.add_argument("--poscar", default="POSCAR_prim")
    args = ap.parse_args()

    latt, frac, species = read_cell(args.poscar)
    U, is_rot, i_atoms = octahedron_basis(latt, frac, species)
    loc = {k: m for m, k in enumerate(i_atoms)}   # I unit-cell index -> local 0..5
    inv = np.linalg.inv(latt)

    def cell_key(R):
        return tuple(np.round(R @ inv).astype(int))

    def site_and_M(idx, cell):
        """Return (site_id, projection matrix ncoord x 3, rot-mask ncoord) for an atom."""
        sp = species[idx]
        if sp == "I":
            m = loc[idx]
            M = U[:, 3 * m:3 * m + 3]          # 18x3 : collective coord x cartesian
            return ("oct", cell), M, is_rot
        else:                                  # Sn or Cs: 3 translation coords = identity
            return (sp, idx, cell), np.eye(3), np.array([False, False, False])

    # accumulate collective FC3 coherently, keyed by (siteA,siteB,siteC)
    coll = {}
    masks = {}
    total_cart = 0.0
    with open(args.fc3) as f:
        raw = f.readlines()
    p = 0
    while raw[p].strip() == "":
        p += 1
    nb = int(raw[p].split()[0]); p += 1
    for _ in range(nb):
        while raw[p].strip() == "":
            p += 1
        p += 1
        R2 = np.array([float(x) for x in raw[p].split()]); p += 1
        R3 = np.array([float(x) for x in raw[p].split()]); p += 1
        ijk = [int(x) - 1 for x in raw[p].split()]; p += 1
        phi = np.zeros((3, 3, 3))
        for _ in range(27):
            t = raw[p].split(); p += 1
            phi[int(t[0]) - 1, int(t[1]) - 1, int(t[2]) - 1] = float(t[3])
        total_cart += np.sum(phi ** 2)
        sA, MA, rA = site_and_M(ijk[0], (0, 0, 0))
        sB, MB, rB = site_and_M(ijk[1], cell_key(R2))
        sC, MC, rC = site_and_M(ijk[2], cell_key(R3))
        contrib = np.einsum("abc,ia,jb,kc->ijk", phi, MA, MB, MC)
        key = (sA, sB, sC)
        if key not in coll:
            coll[key] = np.zeros_like(contrib)
            masks[key] = (rA, rB, rC)
        coll[key] += contrib

    # bin |collective|^2 by number of rotational legs and intra/inter
    tot_coll = 0.0
    nrot_bins = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    rot_involved_intra = rot_involved_inter = 0.0
    for key, arr in coll.items():
        rA, rB, rC = masks[key]
        sA, sB, sC = key
        sq = arr ** 2
        tot_coll += sq.sum()
        # nrot per element = rA[i]+rB[j]+rC[k]
        nrot = (rA[:, None, None].astype(int) + rB[None, :, None].astype(int)
                + rC[None, None, :].astype(int))
        for n in range(4):
            nrot_bins[n] += sq[nrot == n].sum()
        # rotation-involved (>=1 leg), split intra/inter
        involved = sq[nrot >= 1].sum()
        intra = (sA == sB == sC) and sA[0] == "oct"
        if intra:
            rot_involved_intra += involved
        else:
            rot_involved_inter += involved

    rot_involved = rot_involved_intra + rot_involved_inter
    print("=== self-check (orthonormal transform preserves total) ===")
    print(f"  sum|Phi3|^2 Cartesian  : {total_cart:.6e}")
    print(f"  sum|Phi3|^2 collective : {tot_coll:.6e}   ratio = {tot_coll/total_cart:.4f}")
    print("\n=== cubic anharmonicity by number of rotational (libration) legs ===")
    for n in range(4):
        print(f"  {n} rot leg(s): {100*nrot_bins[n]/tot_coll:5.2f}%")
    print("\n=== anharmonicity involving octahedral rotation (>=1 leg) ===")
    print(f"  total rotation-involved : {100*rot_involved/tot_coll:.2f}% of all cubic anharmonicity")
    print(f"    intra-octahedral      : {100*rot_involved_intra/rot_involved:.1f}% of the rotation-involved")
    print(f"    INTER-octahedral      : {100*rot_involved_inter/rot_involved:.1f}%   <- anharmonic tilting interplay")


if __name__ == "__main__":
    main()
