"""w810 (HC mixing = I, separate heads) to its end against HC and mHC identity + linear separate heads, and w811: routing and
pre-norm parameter norms, write-offset growth per 100 steps, the rows with the largest grad norms and the layers they sit
in, the layer-48 gap around them. Source: cache/*.npz (sep_layers_fetch.py, rows every 10 steps)."""
import numpy as np, re
from pathlib import Path
C = Path(__file__).parent / "cache"
R = {"w810 HC I sep": "sep-ahc-hcid-fa4thd-vs-d48", "w811 HC I ord": "sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", "HC sep": "sep-ahc-fa4thd-vs-d48",
     "HC ord": "sep-bbfrozen-ahc-fa4thd-vs-d48", "mHC I lin sep": "sep-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"}
D = {}
for n, t in R.items():
    z = np.load(C / f"{t}.npz"); D[n] = (z["step"], {str(k): i for i, k in enumerate(z["keys"])}, z["v"])
    print(n, "rows", len(z["step"]), "last row", z["step"][-1])
def series(n, pat, pref="params/norm/"):
    s, k, v = D[n]; ids = [i for name, i in k.items() if name.startswith(pref) and re.search(pat, name)]
    return s, np.sqrt(np.nansum(v[:, ids] ** 2, 1)) if ids else None
T = [500, 1000, 2000, 2750, 3250, 3650, 4000, 4250, 4500, 4750, 4950]
def show(title, pats):
    print(f"\n{title}\n{'run':16s}" + " ".join(f"{t:>6d}" for t in T))
    for n in D:
        pat = pats.get("mhc" if n.startswith("mHC") else "hc")
        if pat is None: continue
        s, y = series(n, pat)
        print(f"{n:16s}" + " ".join(f"{y[np.argmin(abs(s-t))]:6.2f}" if s[-1] >= t - 5 else "     -" for t in T))
show("write offsets", {"hc": r"hyper\.\d\.write$"})
show("read offsets", {"hc": r"hyper\.\d\.read$"})
show("dynamic scales", {"hc": r"hyper\.\d\.scale$", "mhc": r"hyper\.\d\.gain$"})
show("dynamic write weights w_write", {"hc": r"hyper\.\d\.w_write$"})
show("dynamic read weights w_read", {"hc": r"hyper\.\d\.w_read$"})
show("ln_2", {"hc": r"ln_2\.weight$", "mhc": r"ln_2\.weight$"})
show("ln_1", {"hc": r"ln_1\.weight$", "mhc": r"ln_1\.weight$"})
print("\nwrite-offset growth per 100 steps")
W = ((1000, 2000), (2000, 3000), (3000, 3500), (3500, 4000), (4000, 4500), (4500, 4950))
print(" windows", W)
for n in ("w810 HC I sep", "HC sep", "w811 HC I ord", "HC ord"):
    s, y = series(n, r"hyper\.\d\.write$"); out = []
    for a, b in W:
        ia, ib = np.argmin(abs(s - a)), np.argmin(abs(s - b)); out.append(f"{(y[ib]-y[ia])/(b-a)*100:+.3f}")
    print(f" {n:16s}", " ".join(out))
print("\nlargest total grad rows of w810, with the three parameters carrying most of it")
sa, ka, va = D["w810 HC I sep"]
g = va[:, ka["grad/norm/total"]]
names = [k for k in ka if k.startswith("grad/norm/") and k != "grad/norm/total"]
idx = np.array([ka[k] for k in names])
for i in np.argsort(-g)[:10]:
    row = va[i, idx]; top = np.argsort(-np.nan_to_num(row))[:3]
    print(f"  step {int(sa[i])}: total {g[i]:.1f}; " + "; ".join(f"{names[t].replace('grad/norm/','').replace('transformer.layers.','L')} {row[t]:.1f}" for t in top))
print("\nmedian total grad per 250 steps, rows above 1.5 x window median: w810, HC sep, w811")
for lo in range(2500, 5000, 250):
    out = []
    for n in ("w810 HC I sep", "HC sep", "w811 HC I ord"):
        s, k, v = D[n]; gg = v[:, k["grad/norm/total"]]; m = (s >= lo) & (s < lo + 250)
        out.append(f"{np.median(gg[m]):7.2f} {int((gg[m] > 1.5 * np.median(gg[m])).sum()):2d}/{int(m.sum())}" if m.any() else "     -")
    print(f"  {lo}", " | ".join(out))
print("\nbranch weight grad (down_proj, o_proj) per layer, medians, w810 / HC")
LS = [0, 1, 2, 3, 7, 15, 23, 35, 47]
sb, kb, vb = D["HC sep"]
st = np.intersect1d(sa, sb); ia, ib = np.searchsorted(sa, st), np.searchsorted(sb, st)
print("window     " + " ".join(f"L{l+1:>10d}" for l in LS))
for lo, hi in ((1750, 2750), (2750, 3250), (3250, 3660), (3660, 4250), (4250, 4950)):
    m = (st >= lo) & (st < hi); row = []
    for l in LS:
        f = lambda k, v, i: np.sqrt(sum(v[i][:, k[f"grad/norm/transformer.layers.{l}.{w}.weight"]] ** 2 for w in ("mlp.down_proj", "self_attn.o_proj")))
        row.append(f"{np.median(f(ka, va, ia)[m]):5.2f}/{np.median(f(kb, vb, ib)[m]):5.2f}")
    print(f"{lo}-{hi} " + " ".join(row))
print("\nper-layer CE gap w810 minus HC, 250-step means")
LL = [0, 3, 7, 11, 23, 35, 47]
print("window     " + " ".join(f"L{l+1:>7d}" for l in LL))
for lo in (1750, 2500, 3000, 3500, 4000, 4500, 4700):
    m = (st >= lo) & (st < lo + 250)
    print(f"{lo:5d}      " + " ".join(f"{np.mean(va[ia][m][:, ka[f'train/pls/L{l}']] - vb[ib][m][:, kb[f'train/pls/L{l}']]):+8.3f}" for l in LL))
