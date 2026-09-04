"""Call PDynA's draw_tilt_and_corr_density_shade (the Glazer plot) on Cs2SnI6
300 K data: tilt histogram per axis, filled/shaded by the sign of the tilt
correlation (red = in-phase/+, blue = anti-phase/-).

Run (pmg env):
    MPLBACKEND=Agg /opt/anaconda3/envs/pmg/bin/python 666_mlff/run_pdyna_glazer.py
"""
import os
import json
import numpy as np

status = {}
try:
    from pdyna.analysis import draw_tilt_and_corr_density_shade
    os.chdir("/Users/tianshu/Documents/MLFF/Cs2SnI6/666_mlff")
    d = np.load("pdyna_300K.npz", allow_pickle=True)
    T = np.asarray(d["tilt"], float)             # (nf, n_oct, 3)
    Corr = np.asarray(d["tilt_corr"], float)     # (nf, n_oct, 1, 3)  1 = shell index
    if Corr.ndim == 4 and Corr.shape[2] == 1:
        Corr = Corr[:, :, 0, :]                  # -> (nf, n_oct, 3): drop the single shell axis
    status["T_shape"] = list(T.shape)
    status["Corr_shape"] = list(Corr.shape)
    draw_tilt_and_corr_density_shade(
        T, Corr, "Cs2SnI6_300K", saveFigures=True,
        title=r"Cs$_2$SnI$_6$  300 K")
    status["status"] = "OK"
    status["fig"] = "666_mlff/traj_tilt_and_corr_density_Cs2SnI6_300K.png"
except Exception as e:
    import traceback
    status["status"] = "ERROR"
    status["err"] = repr(e)
    status["trace"] = traceback.format_exc()[-2000:]

with open("/Users/tianshu/Documents/MLFF/Cs2SnI6/666_mlff/_glazer_status.json", "w") as f:
    json.dump(status, f, indent=1)
