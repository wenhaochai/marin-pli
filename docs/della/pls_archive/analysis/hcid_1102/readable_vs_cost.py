"""Across every 48-layer design pair in d48g4.npz: does the ordinary model's mid-layer readability go with its layer-48
loss and with the cost of separate heads? cost = separate-heads L48 final - probes L48 final. mid = mean probe train loss
over layers 12-36 (final = last 50 steps). gap = mean over layers 12-36 of (probe loss - separate-head loss).
Spearman rank correlations, all pairs and without the two outliers named in docs pls.md (AttnRes seed 0, DeepNorm)."""
import numpy as np
from scipy.stats import spearmanr, pearsonr
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
rows = []
for k in z.files:
    if not (k.startswith("-sep-bbfrozen") and k.endswith("/final")): continue
    t = k[len("-sep-bbfrozen"):-6]
    s = "-sep" + t + "/final"
    if s not in z.files or z[k].shape[0] != 48: continue
    p, q = z[k], z[s]
    pe, qe = z[k[:-6] + "/eval"], z[s[:-6] + "/eval"]
    m = slice(11, 36)
    rows.append(dict(t=t, cost=float(q[47] - p[47]), costev=float(qe[47] - pe[47]), L48o=float(p[47]), L48s=float(q[47]), L48oe=float(pe[47]), mid=float(p[m].mean()), gap=float((p - q)[m].mean()), L24=float(p[23]), L12=float(p[11]), mide=float(pe[m].mean())))
rows.sort(key=lambda r: r["mid"])
print(f"{'design':38s} mid(12-36)  gap   L24   L48ord L48sep  cost  evalL48ord costeval")
for r in rows:
    print(f"{r['t']:38s} {r['mid']:6.3f} {r['gap']:6.3f} {r['L24']:6.3f} {r['L48o']:6.3f} {r['L48s']:6.3f} {r['cost']:6.3f} {r['L48oe']:7.3f} {r['costev']:7.3f}")
def rep(rs, name):
    print(name, "n", len(rs))
    for a in ("mid", "gap", "L24", "L12", "mide"):
        for b in ("cost", "costev", "L48o", "L48oe", "L48s"):
            x, y = [r[a] for r in rs], [r[b] for r in rs]
            sp = spearmanr(x, y); pe = pearsonr(x, y)
            print(f"  {a:5s} vs {b:7s} spearman {sp.correlation:+.2f} (p {sp.pvalue:.3g})  pearson {pe.statistic:+.2f}")
rep(rows, "all pairs")
rep([r for r in rows if r["t"] not in ("-adeepnorm-vs-d48-g4", "-aattnres-vs-d48-g4")], "without DeepNorm, AttnRes seed 0")
