"""Norm of the routing parameters by step and layer group, from the W&B caches (sep_layers_fetch2.py). Usage: norm_traj.py TAG ... """
import sys, numpy as np
from pathlib import Path
H = Path(__file__).parent / "cache"
ST = [10, 20, 50, 100, 200, 500, 1000, 2000, 4950]
GR = [(1, 1), (2, 6), (7, 11), (12, 12), (13, 24), (37, 48)]
for t in sys.argv[1:]:
    d = np.load(H / f"{t}.npz"); s, K, v = d["step"], list(d["keys"]), d["v"]
    print("==", t)
    for h in ("res", "read", "write", "w_res", "scale"):
        print(f" {h}: step " + " ".join(f"{x:7d}" for x in ST))
        for lo, hi in GR:
            row = []
            for x in ST:
                i = np.argmin(np.abs(s - x)); vals = [v[i, K.index(f"params/norm/transformer.layers.{l - 1}.hyper.{j}.{h}")] for l in range(lo, hi + 1) for j in (0, 1)]
                row.append(np.nanmean(vals))
            print(f"   L{lo}-{hi:<3d}".ljust(13) + " ".join(f"{x:7.3f}" for x in row))
