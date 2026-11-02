"""Ordinary-model (probes) L48 train final and eval of all HC / mHC design runs in d48g4.npz, after fetching w811; per-layer
differences w811 - HC and w811 - mHC identity + linear, train final and eval."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
ks = sorted(x[:-6] for x in z.files if x.endswith("/final") and x.startswith("-sep-bbfrozen"))
rows = []
for k in ks:
    if not ("ahc" in k or "amhc" in k or "apreln" in k): continue
    f, e = z[k + "/final"], z[k + "/eval"]
    if f.shape[0] != 48: continue
    rows.append((float(f[47]), float(e[47]), k))
for f, e, k in sorted(rows): print(f"{k:58s} train {f:.4f} eval {e:.4f}")
I, H, M = "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"
L = [0, 3, 7, 11, 23, 35, 46, 47]
print("layer", [l + 1 for l in L])
for name, b in (("w811-HC", H), ("w811-mHCidlin", M)):
    print(name, "train", np.round((z[I + "/final"] - z[b + "/final"])[L], 3).tolist(), "eval", np.round((z[I + "/eval"] - z[b + "/eval"])[L], 3).tolist())
