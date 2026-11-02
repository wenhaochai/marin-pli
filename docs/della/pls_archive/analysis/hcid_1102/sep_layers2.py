"""Do the gains of w810 (HC mixing = I, separate heads) run away, and only there? Routing and pre-norm parameter norms over
training for w810, w811 (same design, ordinary model), HC separate heads / ordinary, and mHC identity + linear separate
heads. mHC's read/write offsets are its bias; its gains its gain. Also: w810's gap to HC around its largest grad norm, and
grad norms of the branch weights per layer. Source: cache/*.npz (sep_layers_fetch.py), rows every 10 steps."""
import numpy as np, re
from pathlib import Path
C = Path(__file__).parent / "cache"
R = {"w810 HC I sep": "sep-ahc-hcid-fa4thd-vs-d48", "w811 HC I ord": "sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", "HC sep": "sep-ahc-fa4thd-vs-d48",
     "HC ord": "sep-bbfrozen-ahc-fa4thd-vs-d48", "mHC I lin sep": "sep-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"}
D = {}
for n, t in R.items():
    z = np.load(C / f"{t}.npz"); D[n] = (z["step"], {str(k): i for i, k in enumerate(z["keys"])}, z["v"])
def series(n, pat, pref="params/norm/"):
    s, k, v = D[n]; ids = [i for name, i in k.items() if name.startswith(pref) and re.search(pat, name)]
    return s, np.sqrt(np.nansum(v[:, ids] ** 2, 1)) if ids else None
T = [500, 1000, 1500, 2000, 2500, 2750, 3000, 3250, 3500, 3650, 4000, 4500, 4950]
def show(title, pats):
    print(f"\n{title}\n{'run':16s}" + " ".join(f"{t:>6d}" for t in T))
    for n in D:
        pat = pats.get("mhc" if n.startswith("mHC") else "hc")
        if pat is None: continue
        s, y = series(n, pat)
        if y is None: continue
        print(f"{n:16s}" + " ".join(f"{y[np.argmin(abs(s-t))]:6.2f}" if s[-1] >= t - 5 else "     -" for t in T))
show("write offsets, root sum of squares (mHC: bias, read + write columns only cannot be split in the log: see bias)", {"hc": r"hyper\.\d\.write$", "mhc": r"hyper\.\d\.bias$"})
show("read offsets", {"hc": r"hyper\.\d\.read$"})
show("dynamic scales (hc: s; mhc: gain, includes the unused residual gain at 0.01)", {"hc": r"hyper\.\d\.scale$", "mhc": r"hyper\.\d\.gain$"})
show("pre-norm scales ln_2 (MLP)", {"hc": r"ln_2\.weight$", "mhc": r"ln_2\.weight$"})
show("pre-norm scales ln_1 (attention)", {"hc": r"ln_1\.weight$", "mhc": r"ln_1\.weight$"})
print("\ngrowth rate of w810's and HC's write norm per 100 steps, with the learning rate")
for n in ("w810 HC I sep", "HC sep", "w811 HC I ord", "HC ord"):
    s, y = series(n, r"hyper\.\d\.write$"); out = []
    for a, b in ((1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, 3250), (3250, 3500), (3500, 3650)):
        ia, ib = np.argmin(abs(s - a)), np.argmin(abs(s - b)); out.append(f"{(y[ib]-y[ia])/(b-a)*100:+.3f}")
    print(f" {n:16s}", " ".join(out))
print("  windows 1000-1500, 1500-2000, 2000-2500, 2500-3000, 3000-3250, 3250-3500, 3500-3650; LR 0.0187, 0.0162, 0.0137, 0.0112, 0.0093, 0.0080, 0.0070")
# gap around the spike
sa, ka, va = D["w810 HC I sep"]; sb, kb, vb = D["HC sep"]
st = np.intersect1d(sa, sb); ia, ib = np.searchsorted(sa, st), np.searchsorted(sb, st)
gap = va[ia, ka["train/pls/L47"]] - vb[ib, kb["train/pls/L47"]]; g = va[ia, ka["grad/norm/total"]]
print("\n50-step mean CE48 gap and max total grad, w810 minus HC, steps 3200-3650")
for lo in range(3200, 3650, 50):
    m = (st >= lo) & (st < lo + 50)
    if m.any(): print(f"  {lo}: gap {gap[m].mean():+.3f}, grad max {g[m].max():.1f}")
print("\nshare of rows with total grad > 1.5 x window median, w810 / HC")
gb = vb[ib, kb["grad/norm/total"]]
for lo, hi in ((1000, 2000), (2000, 2750), (2750, 3250), (3250, 3660)):
    m = (st >= lo) & (st < hi); print(f"  {lo}-{hi}: {(g[m] > 1.5*np.median(g[m])).sum()}/{m.sum()} , {(gb[m] > 1.5*np.median(gb[m])).sum()}/{m.sum()}")
print("\nbranch weight grad (down_proj, o_proj) per layer, mean by window, w810 / HC")
LS = [0, 1, 2, 3, 7, 15, 23, 35, 47]
print("window     " + " ".join(f"L{l+1:>10d}" for l in LS))
for lo, hi in ((1750, 2750), (2750, 3250), (3250, 3660)):
    m = (st >= lo) & (st < hi); row = []
    for l in LS:
        f = lambda k, v, i: np.sqrt(sum(v[i][:, k[f"grad/norm/transformer.layers.{l}.{w}.weight"]] ** 2 for w in ("mlp.down_proj", "self_attn.o_proj")))
        row.append(f"{np.median(f(ka, va, ia)[m]):5.2f}/{np.median(f(kb, vb, ib)[m]):5.2f}")
    print(f"{lo}-{hi} " + " ".join(row))
