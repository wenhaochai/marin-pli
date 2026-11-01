"""Across all 48-layer pairs in d48g4.npz: how much of the spread in separate-heads cost comes from each term.
cost = sep - probes, so var(cost) = var(sep) + var(probes) - 2 cov. Prints Pearson / Spearman of cost with each term,
for all pairs and without the two outliers (AttnRes seed 0: spiky ordinary model; DeepNorm: 0.3 nats worse overall).
Read-only."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
d = dict(np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True))
tags = sorted({k[5:-6] for k in d if k.startswith("-sep-") and not k.startswith("-sep-bbfrozen-") and k.endswith("/final")})
R = []
for t in tags:
    a, b = d.get(f"-sep-{t}/final"), d.get(f"-sep-bbfrozen-{t}/final")
    if a is not None and b is not None and len(a) >= 48 and len(b) >= 48:
        R.append((t, float(a[47]), float(b[47])))
for name, excl in (("all", set()), ("no AttnRes s0, no DeepNorm", {"aattnres-vs-d48-g4", "adeepnorm-vs-d48-g4"})):
    rows = [r for r in R if r[0] not in excl]
    s = np.array([r[1] for r in rows]); p = np.array([r[2] for r in rows]); c = s - p
    print(f"{name}: n={len(rows)}  sd sep {s.std():.4f}  sd probes {p.std():.4f}  sd cost {c.std():.4f}  corr(sep,probes) {pearsonr(s,p)[0]:+.2f}")
    for lab, x in (("sep", s), ("probes", p)):
        print(f"   cost vs {lab:6s}: pearson {pearsonr(c,x)[0]:+.2f} (p {pearsonr(c,x)[1]:.3f})  spearman {spearmanr(c,x)[0]:+.2f} (p {spearmanr(c,x)[1]:.3f})")
