"""Print the mean pairwise cosine between token states (aniso, depthmed.py: over a fixed sample of 2,048 of the 16 x 1,024
Paloma c4_en tokens; uncentred) by layer, for the 10 designs x {ordinary (ord), separate heads (sep)} at 48 layers.
Source: tmp/pls_blog/depthmed/<design>-<ord|sep>.json ([embed_stats].aniso, [layers][k].aniso). 2026-11-01, for the
owner's question about Mufan Li's same-layer feature correlation."""
import json
from pathlib import Path
D = Path("/scratch/gpfs/GROUP/USER/tmp/pls_blog/depthmed")
ks = [1, 2, 4, 8, 12, 16, 24, 32, 40, 44, 46, 47, 48]
print(f"{'design':16s} {'emb':>5s} " + " ".join(f"{k:>5d}" for k in ks) + "   max(layer)")
for d in ["preln", "sandwich", "lns", "deepnorm", "keel", "hc", "mhc", "moda", "attnres_block", "attnres"]:
    for arm in ("ord", "sep"):
        j = json.load(open(D / f"{d}-{arm}.json"))
        a = [L.get("aniso") for L in j["layers"]]
        e = j.get("embed_stats", {}).get("aniso")
        mx = max(range(len(a)), key=lambda i: a[i] if a[i] is not None else -9)
        print(f"{d + '-' + arm:16s} {e if e is None else round(e, 2):>5} " + " ".join(f"{a[k-1]:5.2f}" for k in ks) + f"   {a[mx]:.2f}({mx+1})")
