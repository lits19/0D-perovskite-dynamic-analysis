# Cs₂SnI₆ — Octahedral Tilt & Distortion Dynamics from MLFF Molecular Dynamics

Analysis pipeline for the collective octahedral dynamics of the vacancy-ordered
double perovskite **Cs₂SnI₆** (0D perovskite), extracted from long **machine-learning
force-field (MLFF)** molecular-dynamics trajectories run in VASP (R²SCAN-trained
on-the-fly MLFF, NVE).

The trajectories are analysed on a **6×6×6 supercell** (`864` octahedra,
`SUPERCELL_666.vasp`) over a temperature series **300 → 650 K** in 50 K steps.
For each temperature we quantify the rigid-body octahedral **tilt** field and the
intra-octahedral **distortion** field, and characterise their spatial (S(**q**))
and temporal correlations — including a tilt-projected dynamical structure factor
S_tilt(**q**, ω) that traces the libration branch.

---

## Pipeline overview

```
 VASP MLFF (NVE) MD                     INCAR  (R2SCAN MLFF, POTIM=2 fs, NBLOCK=5)
        │
        ▼  666_nve_<T>K_s1.xyz  (multi-frame extxyz)
 ┌──────────────────────┐
 │ extxyz_to_xdatcar.py │  ── extxyz → VASP XDATCAR
 └──────────┬───────────┘
            ▼  XDATCAR_<T>/XDATCAR
 ┌──────────────────────┐   uses PDynA (WMD-group)
 │  pdyna_analysis.py   │  ── per-octahedron tilt / distortion / correlation
 └──────────┬───────────┘
            ▼  tilt_<T>K.npy · distort_<T>K.npy · tilt_corr_<T>K.npy · Benv_<T>K.npy
            ▼  pdyna_<T>K.npz   (bundle of all arrays + metadata)
            │
   ┌────────┴──────────────────────────────────┬─────────────────────────────┐
   ▼                                            ▼                             ▼
 ┌────────────────┐   ┌────────────────────┐  ┌──────────────┐   ┌────────────────────────┐
 │ tilt_analysis  │   │ distort_analysis   │  │ tilt_sqw.py  │   │ plot_tilt_distort_vs_T │
 │  S(q), C(R),   │   │  (negative control)│  │ S_tilt(q,ω)  │   │  tilt/distortion vs T  │
 │  time-corr     │   │  distortion S(q)   │  │  libration   │   │  (Glazer-shaded)       │
 └───────┬────────┘   └─────────┬──────────┘  └──────┬───────┘   └───────────┬────────────┘
         ▼                      ▼                     ▼                       ▼
                            figs_<T>K/*.png   +   top-level *.png
```

---

## Repository layout

```
666_mlff/
├── README.md                     ← this file
├── LICENSE
├── requirements.txt / environment.yml
├── exec.sh                       ← end-to-end driver for one temperature
│
├── INCAR                         ← VASP MLFF-MD settings (R2SCAN, NVE) — provenance
├── SUPERCELL_666.vasp            ← 6×6×6 reference structure (Sn/octahedron order)
├── Cs2SnI6_tavg.cif              ← time-averaged structure
│
├── extxyz_to_xdatcar.py          ← extxyz → XDATCAR
├── pdyna_analysis.py             ← PDynA tilt/distortion extraction (main driver)
├── tilt_analysis.py              ← tilt: S(q) maps, shell correlations, time-corr
├── distort_analysis.py           ← distortion: same analysis (negative control)
├── tilt_sqw.py                   ← tilt dynamical structure factor S_tilt(q,ω)
├── plot_tilt_dist.py             ← 300 K tilt-angle distribution (3 axes)
├── plot_tilt_distort_vs_T.py     ← temperature trend of tilt & distortion
├── project_fc3_channels.py       ← FC3 → octahedral collective-coordinate channels
├── run_pdyna_tilt.py             ← PDynA-native tilt density (filled)
├── run_pdyna_glazer.py           ← PDynA Glazer plot (tilt + sign-of-correlation)
├── _glazer_src.py                ← vendored PDynA Glazer helper
│
├── XDATCAR_<T>/                  ← trajectory + PDynA arrays  (300…650 K)   [not tracked]
│     XDATCAR, tilt_<T>K.npy, distort_<T>K.npy, tilt_corr_<T>K.npy,
│     Benv_<T>K.npy, pdyna_<T>K.npz
└── figs_<T>K/                    ← figures for each temperature  (300…650 K) [tracked]
      fig1_CR_shells_<T>.png, fig2_Sq_maps_<T>.png, fig2b_Sq_path_<T>.png,
      fig3_time_corr_<T>.png, fig4_static_check_<T>.png,
      distort_fig1…fig4_<T>.png, figure_01…14.png
```

> **Note on large data.** A single `XDATCAR` is ≈1 GB and the per-temperature
> `.npy`/`.npz` arrays are tens of MB, so they are **git-ignored** (see
> `.gitignore`) and must be regenerated locally with the steps below. The small
> inputs (`SUPERCELL_666.vasp`, `INCAR`, `*.cif`) and all result figures
> (`figs_<T>K/*.png`, top-level `*.png`) are tracked.

---

## Installation

```bash
# option A — conda (recommended; PDynA installed via pip inside the env)
conda env create -f environment.yml
conda activate cs2sni6-mlff

# option B — pip
pip install -r requirements.txt
```

Requires **numpy, scipy, matplotlib, ase**, and
[**PDynA**](https://github.com/WMD-group/PDynA) (Perovskite Dynamics Analysis).

---

## Usage

### One-shot driver

`exec.sh` runs the whole chain for a single temperature — set `T` at the top:

```bash
T=400
python extxyz_to_xdatcar.py 666_nve_${T}K_s1.xyz --outdir XDATCAR_${T}
python pdyna_analysis.py ${T} --read-every 2
python tilt_analysis.py     tilt_${T}K.npy    --dt-fs 20.0 --out-dir figs_${T}K
python distort_analysis.py  distort_${T}K.npy --dt-fs 20.0 --out-dir figs_${T}K
```

### Step by step

```bash
# 1) MD trajectory (extxyz) → VASP XDATCAR
python extxyz_to_xdatcar.py 666_nve_300K_s1.xyz --outdir XDATCAR_300

# 2) PDynA: per-octahedron tilt / distortion / correlation
#    reads SUPERCELL_666.vasp, INCAR, XDATCAR_300/XDATCAR
#    writes tilt_300K.npy, distort_300K.npy, tilt_corr_300K.npy, Benv_300K.npy, pdyna_300K.npz
python pdyna_analysis.py 300 --read-every 2

# 3) tilt correlations — S(q) maps, fcc-shell correlations C(R), time correlation
python tilt_analysis.py tilt_300K.npy --dt-fs 20.0 --out-dir figs_300K

# 4) distortion (negative control) — same analysis on the distortion magnitude field
python distort_analysis.py distort_300K.npy --dt-fs 20.0 --out-dir figs_300K

# 5) dynamical structure factor of the tilt field, S_tilt(q, ω)  → libration branch
python tilt_sqw.py tilt_300K.npy --dt-fs 20.0

# 6) temperature trend across the whole series (reads pdyna_<T>K.npz)
python plot_tilt_distort_vs_T.py
```

`--dt-fs 20.0` matches the effective XDATCAR stride (`POTIM 2 fs × NBLOCK 5`
gives a 10 fs write stride; `--read-every 2` in step 2 doubles it to 20 fs).

---

## Scripts

| Script | Role |
| --- | --- |
| `extxyz_to_xdatcar.py` | Convert multi-frame extxyz MD output to VASP `XDATCAR` (groups by formula/atom count). |
| `pdyna_analysis.py` | **Main extraction.** Runs PDynA on `XDATCAR_<T>` to get per-octahedron tilt (signed, 3 axes), distortion (4 non-negative modes), tilt correlation, and B-site environment; bundles to `pdyna_<T>K.npz`. |
| `tilt_analysis.py` | Spatial/temporal analysis of the **tilt** field: cubic-star-averaged S(**q**) maps and high-symmetry path, fcc-shell correlations C(R), and time-correlation decay. |
| `distort_analysis.py` | Same analysis on the **distortion magnitude** field — the *negative control* (intra-octahedral, expected weakly correlated; Γ annotated not colour-scaled). |
| `tilt_sqw.py` | Tilt **dynamical structure factor** S_tilt(**q**, ω) — the MD analogue of a phonon dispersion; intensity peaks trace ω_lib(**q**), overlayable on SCPH dispersion. |
| `plot_tilt_dist.py` | 300 K tilt-angle distribution on the three pseudo-cubic axes (histogram + Gaussian fit), Glazer-style. |
| `plot_tilt_distort_vs_T.py` | Temperature trend of mean tilt amplitude ⟨|θ|⟩ and the four distortion modes, with Glazer-shaded stacked panels. |
| `project_fc3_channels.py` | Decompose the cubic anharmonicity (FC3) into octahedral collective coordinates (rotation/translation/distortion); quantify libration-channel anharmonicity, intra- vs inter-octahedral. |
| `run_pdyna_tilt.py` | PDynA-native tilt density plot (filled variant). |
| `run_pdyna_glazer.py` | PDynA Glazer plot: tilt histograms shaded by the sign of the tilt correlation (red = in-phase, blue = anti-phase). |
| `_glazer_src.py` | Vendored PDynA `draw_tilt_and_corr_density_shade` helper used by `run_pdyna_glazer.py`. |

---

## Data & provenance

- **MD settings** (`INCAR`): VASP MLFF (`ML_LMLFF=.TRUE.`, `ML_MODE=run`),
  metaGGA **R²SCAN**, **NVE** (`MDALGO=0`, `SMASS=-3`), `POTIM = 2 fs`,
  `NBLOCK = 5` → 10 fs XDATCAR stride, `NSW = 17500` (≈35 ps per run).
- **System**: 6×6×6 Cs₂SnI₆ supercell, 864 octahedra.
- Representative statistics (`_diag.json`, 300 K): 1664 frames, 33.3 ps,
  correlation time τ ≈ 0.46 ps.

---

## License

Released under the [MIT License](LICENSE). If you use this pipeline, please also
cite [PDynA](https://github.com/WMD-group/PDynA).
