"""6-layer version of eff_depth.py: ordinary models of 1 to 6 layers (depth.json, baselines.json) against the 6-layer model with separate heads (layers.npz, -pls1-sep), train, mean of the last 50 logged steps; and the runs with the layer losses switched off early."""
import json, numpy as np
from pathlib import Path
B = Path("/scratch/gpfs/GROUP/USER/tmp/pls_blog")
d = json.load(open(B / "depth.json")); b = json.load(open(B / "baselines.json"))
cur = {int(k): float(v["loss"]) for k, v in d.items()}; cur[6] = float(b["130m"]["loss"])
print("ordinary by depth, train:", {k: round(v, 3) for k, v in sorted(cur.items())})
z = np.load(B / "layers.npz", allow_pickle=True)
print([k for k in z.files if k.endswith("/ce")][:30])
for t in ("-pls1-sep", "-pls1", "-pls1-dh", "-pls1-off80", "-pls1-off140", "-pls1-off0-150"):
    if t + "/ce" in z.files:
        c = z[t + "/ce"]; f = c[-50:].mean(0)
        y = float(f[-1]); ds = sorted(cur)
        w = next((a * (e / a) ** ((cur[a] - y) / (cur[a] - cur[e])) for a, e in zip(ds, ds[1:]) if cur[e] <= y <= cur[a]), None)
        print(t, "per layer", np.round(f, 3).tolist(), "last minus ordinary", round(y - cur[6], 4), "worth", None if w is None else round(w, 2))
