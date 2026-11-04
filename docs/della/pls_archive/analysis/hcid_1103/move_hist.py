"""Where do the off-diagonal entries of a = I + res of depth_hc_res="move" sit? From mix_<tag>_<step>.npz (move_mix.py). Bins by layer group: a < -0.2 (gradient factor sigmoid(50 (a - 0.1)) below 3e-7), -0.2 to -0.1 (below 5e-5), -0.1 to 0 (below 7e-3), 0 to 0.06 (below 0.12), above 0.06 (open). Diagonal: mean a. The HC column of the same bins reads a raw."""
import glob, sys
from pathlib import Path
import numpy as np
H = Path(__file__).parent
GROUPS = [(1, 1), (2, 6), (7, 11), (12, 12), (13, 24), (25, 36), (37, 48)]
EDGES = [-np.inf, -0.2, -0.1, 0.0, 0.06, np.inf]
for f in sys.argv[1:]:
    res = np.load(H / f)["res"]
    m = res.shape[-1]
    off = ~np.eye(m, dtype=bool)
    print("==", f, "share of off-diagonal entries by a: <-0.2, -0.2..-0.1, -0.1..0, 0..0.06, >0.06; mean, min, max off-diagonal a; mean diagonal a")
    for lo, hi in GROUPS:
        a = np.eye(m) + res[lo - 1: hi]
        v = a[:, :, off].ravel()
        d = a[:, :, ~off].ravel()
        h = np.histogram(v, EDGES)[0] / v.size
        print(f"  L{lo}-{hi}: " + " ".join(f"{x:.2f}" for x in h) + f"  |  {v.mean():+.3f} {v.min():+.3f} {v.max():+.3f}  |  {d.mean():.3f}")
