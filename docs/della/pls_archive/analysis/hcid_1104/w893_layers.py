"""w893 (HC mixing = I, seed 1, separate heads) against w810 (seed 0) and HC separate heads, from cache/*.npz (sep_layers_fetch.py, sep_layers_fetch2.py; rows every 10 steps).
Is the pattern of w810 repeated? Tables: branch-weight grad (down_proj, o_proj) per layer, medians, in windows; the largest grad rows and the parameters carrying them; write-offset norm growth; ln_2 and scale norms; the per-layer CE gap to HC."""
import numpy as np, re
from pathlib import Path
C = Path(__file__).parent / "cache"
R = {"w893": "sep-ahc-hcid-fa4thd-vs-d48-s1", "w810": "sep-ahc-hcid-fa4thd-vs-d48", "HC": "sep-ahc-fa4thd-vs-d48"}
D = {}
for n, t in R.items():
    z = np.load(C / f"{t}.npz")
    D[n] = (z["step"], {str(k): i for i, k in enumerate(z["keys"])}, z["v"])
    print(n, "rows", len(z["step"]), "last row", z["step"][-1])
def br(n, l):
    s, k, v = D[n]
    return s, np.sqrt(sum(v[:, k[f"grad/norm/transformer.layers.{l}.{w}.weight"]] ** 2 for w in ("mlp.down_proj", "self_attn.o_proj")))
LS = [0, 1, 2, 3, 4, 7, 11, 15, 23, 35, 47]
W = ((1000, 1500), (1500, 1800), (1800, 2400), (2400, 3000), (3000, 3500), (3500, 4000), (4000, 4550))
print("\nbranch weight grad per layer, median; w893 / w810 / HC")
print("window      " + " ".join(f"L{l + 1:<16d}" for l in LS))
for lo, hi in W:
    o = f"{lo}-{hi - 1:<5} "
    for l in LS:
        x = []
        for n in D:
            s, y = br(n, l)
            m = (s >= lo) & (s < hi)
            x.append(f"{np.nanmedian(y[m]):5.2f}")
        o += "/".join(x) + " "
    print(o)
print("\nlargest total grad rows of w893, with the 4 parameters carrying most of it")
s, k, v = D["w893"]
g = v[:, k["grad/norm/total"]]
names = [a for a in k if a.startswith("grad/norm/") and a != "grad/norm/total"]
idx = np.array([k[a] for a in names])
for i in np.argsort(-g)[:12]:
    row = v[i, idx]
    top = np.argsort(-np.nan_to_num(row))[:4]
    print(f"  step {int(s[i])}: total {g[i]:.1f}; " + "; ".join(f"{names[t].replace('grad/norm/', '').replace('transformer.layers.', 'L')} {row[t]:.1f}" for t in top))
def series(n, pat, pref="params/norm/"):
    s, k, v = D[n]
    ids = [i for name, i in k.items() if name.startswith(pref) and re.search(pat, name)]
    return s, np.sqrt(np.nansum(v[:, ids] ** 2, 1))
T = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
for title, pat in (("write offsets", r"hyper\.\d\.write$"), ("read offsets", r"hyper\.\d\.read$"), ("dynamic scales s", r"hyper\.\d\.scale$"), ("w_write", r"hyper\.\d\.w_write$"), ("w_read", r"hyper\.\d\.w_read$"), ("ln_1", r"ln_1\.weight$"), ("ln_2", r"ln_2\.weight$")):
    print(f"\n{title}\nrun   " + " ".join(f"{t:>7d}" for t in T))
    for n in D:
        s, y = series(n, pat)
        print(f"{n:5s} " + " ".join(f"{y[np.argmin(abs(s - t))]:7.2f}" for t in T))
print("\nCE gap to HC per layer, 500-step means of the rows with grad norms; w893, w810")
LL = [0, 1, 2, 3, 4, 7, 11, 23, 35, 47]
print("window     " + " ".join(f"L{l + 1:<12d}" for l in LL))
sb, kb, vb = D["HC"]
for lo in range(1000, 4600, 500):
    o = f"{lo}-{lo + 499} "
    for l in LL:
        x = []
        for n in ("w893", "w810"):
            sa, ka, va = D[n]
            st = np.intersect1d(sa, sb)
            st = st[(st >= lo) & (st < lo + 500)]
            ia, ib = np.searchsorted(sa, st), np.searchsorted(sb, st)
            x.append(f"{np.nanmean(va[ia, ka[f'train/pls/L{l}']] - vb[ib, kb[f'train/pls/L{l}']]):+.2f}")
        o += " ".join(x) + "  "
    print(o)
