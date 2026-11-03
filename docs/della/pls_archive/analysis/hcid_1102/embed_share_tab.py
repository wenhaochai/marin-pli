"""Across ordinary-model designs: probe train loss at a layer against the state statistics of embed_share.py, Spearman.
keep_k lists start at layer k+1 (index l-k)."""
import json, numpy as np
from scipy.stats import spearmanr
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
R = json.load(open("/scratch/gpfs/GROUP/USER/tmp/pls_hcid/embed_share.json"))
tags = sorted(R, key=lambda t: float(z[t + "/final"][23]))
L = [7, 11, 15, 23, 31, 35, 39]
print(f"{'design':40s}" + " ".join(f"{'CE'+str(l+1):>6s} {'e'+str(l+1):>6s} {'k4_'+str(l+1):>6s}" for l in (11, 23, 35)))
for t in tags:
    r = R[t]; f = z[t + "/final"]
    print(f"{t[len('-sep-bbfrozen'):]:40s}" + " ".join(f"{f[l]:6.2f} {r['e_share'][l]:6.3f} {r['keep_4'][l-4]:6.3f}" for l in (11, 23, 35)))
print("Spearman of probe CE at layer l with each statistic at layer l, n =", len(tags))
print("layer " + " ".join(f"{s:>10s}" for s in ("e_share", "norm", "keep_1", "keep_4", "keep_12")))
hc = [t for t in tags if "-ahc" in t]; rest = [t for t in tags if "-ahc" not in t]
for name, T in (("all", tags), ("HC only", hc), ("non-HC", rest)):
    print(name, len(T))
    for l in L:
        row = []
        for s in ("e_share", "norm_ratio", "keep_1", "keep_4", "keep_12"):
            k = int(s.split("_")[1]) if s.startswith("keep") else 0
            x = [R[t][s][l - k] if l >= k else np.nan for t in T]; y = [z[t + "/final"][l] for t in T]
            row.append(spearmanr(x, y).correlation)
        print(f"  L{l+1:<3d} " + " ".join(f"{v:+10.2f}" for v in row))
