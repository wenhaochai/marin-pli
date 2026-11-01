"""Which term of the separate-heads cost does the stream-difference use track? Spearman of the use (mean collapse and
shuffle, mean over l <= 16, on the separate-heads checkpoint and on the probes checkpoint; tmp/pls_paths/ha[s]-sep[-bbfrozen]-
<tag>.json) against the separate-heads L48 final, the probes L48 final and their difference (d48g4.npz), over the
multi-stream designs that have all of them (one-stream designs have no use to measure and are left out here). Read-only.
2026-11-01."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
O = Path("/scratch/gpfs/GROUP/USER/tmp/pls_paths")
d = dict(np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True))
T = ["ahc-vs-d48-g4", "ahc-vs-d48-s1", "ahc-hcrwsig-vs-d48", "ahc-hcs8-vs-d48", "ahc-hcs2-vs-d48", "amhc-vs-d48-g4",
     "amhc-mhcid-vs-d48", "amhc-mhcrwlin-vs-d48", "amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "ahc-hcwsh-fa4thd-vs-d48"]


def use(prefix, arm, tag):
    f = O / f"{prefix}-sep{arm}-{tag}.json"
    if not f.exists():
        return None
    v = [x for k, x in json.loads(f.read_text())["dce"].items() if int(k) <= 16]
    return sum(v) / len(v)


def fin(arm, tag):
    k = f"-sep{arm}-{tag}/final"
    return float(d[k][47]) if k in d else None


rows = []
for t in T:
    r = dict(tag=t, sep=fin("", t), prb=fin("-bbfrozen", t))
    for prefix in ("ha", "has"):
        for arm, a in (("", "S"), ("-bbfrozen", "P")):
            r[f"{prefix}{a}"] = use(prefix, arm, t)
    rows.append(r)
for xk in ("haS", "hasS", "haP", "hasP"):
    for yk in ("sep", "prb", "cost"):
        pts = [(r[xk], r["sep"] - r["prb"] if yk == "cost" else r[yk]) for r in rows
               if r[xk] is not None and r["sep"] is not None and r["prb"] is not None]
        x, y = zip(*pts)
        s = spearmanr(x, y)
        print(f"use {xk:5s} vs {yk:5s} n={len(pts):2d} spearman {s.statistic:+.2f} (p {s.pvalue:.3f})")
