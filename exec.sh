#!/bin/bash

T=400

python extxyz_to_xdatcar.py 666_nve_${T}K_s1.xyz --outdir XDATCAR_${T}
python pdyna_analysis.py ${T} --read-every 2
python tilt_analysis.py tilt_${T}K.npy --dt-fs 20.0 --out-dir figs_${T}K
python distort_analysis.py distort_${T}K.npy --dt-fs 20.0 --out-dir figs_${T}K
