"""Where and in which parameters does w810 (HC mixing = I, separate heads) fall behind HC after step 2750?
Source: cache/*.npz from sep_layers_fetch.py (rows every 10 steps). Prints: the CE gap to HC per layer by window; the top
grad-norm rows of w810 and which parameters carry them; per-window sum of grad norm by parameter kind; the routing
parameter norms against the separate-heads run of mHC identity + linear; and the same for the ordinary-model pair."""
import numpy as np, re
from pathlib import Path
C = Path(__file__).parent / "cache"
def load(t):
    z = np.load(C / f"{t}.npz"); k = [str(x) for x in z["keys"]]; return z["step"], {n: i for i, n in enumerate(k)}, z["v"]
ID, HC, MH = "sep-ahc-hcid-fa4thd-vs-d48", "sep-ahc-fa4thd-vs-d48", "sep-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"
sa, ka, va = load(ID); sb, kb, vb = load(HC)
st = np.intersect1d(sa, sb); ia = np.searchsorted(sa, st); ib = np.searchsorted(sb, st)
ce = lambda k, v, idx: np.stack([v[idx, k[f"train/pls/L{l}"]] for l in range(48)], 1)
A, B = ce(ka, va, ia), ce(kb, vb, ib)
W = [(1000, 1500), (1750, 2750), (2750, 3000), (3000, 3250), (3250, 3500), (3500, 3660)]
L = [0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45, 46, 47]
print("CE gap to HC per layer (1-based), by window")
print("window      " + " ".join(f"L{l+1:>5d}" for l in L))
for lo, hi in W:
    m = (st >= lo) & (st < hi); print(f"{lo:5d}-{hi:5d} " + " ".join(f"{(A[m, l]-B[m, l]).mean():+6.3f}" for l in L))
print("mean over all 48 layers:", [round(float((A[(st>=lo)&(st<hi)]-B[(st>=lo)&(st<hi)]).mean()),3) for lo, hi in W])
def group(k):
    m = re.match(r"(grad|params)/norm/transformer\.layers\.(\d+)\.(hyper\.\d\.\w+|.*)", k)
    return m
g = va[ia, ka["grad/norm/total"]]; gh = vb[ib, kb["grad/norm/total"]]
top = np.argsort(-g)[:6]
print("\nlargest grad norms of w810 (HC at the same step), and the 5 largest parameter entries")
gk = [k for k in ka if k.startswith("grad/norm/") and k != "grad/norm/total"]
for t in top:
    row = va[ia[t]]; o = sorted(((row[ka[k]], k) for k in gk if np.isfinite(row[ka[k]])), reverse=True)[:5]
    print(f" step {st[t]} total {g[t]:.1f} (HC {gh[t]:.1f}) CE48 {A[t,47]:.3f}:", "; ".join(f"{k.replace('grad/norm/transformer.layers.','')} {v:.1f}" for v, k in o))
def kind(k):
    k = k.replace("grad/norm/", "").replace("params/norm/", "")
    m = re.search(r"hyper\.\d\.(\w+)$", k)
    if m: return "hyper." + m.group(1)
    return k.split(".")[0] if k.startswith("aux") else re.sub(r"transformer\.layers\.\d+\.", "", k)
kinds = sorted({kind(k) for k in gk})
def rss(k_, v, idx, pref, kd, m):
    ks = [i for n, i in k_.items() if n.startswith(pref) and n != pref + "total" and kind(n) == kd]
    return float(np.sqrt(np.nansum(v[idx][:, ks] ** 2, 1))[m].mean()) if ks else float("nan")
print("\nroot sum of squared grad norms by parameter kind, w810 / HC")
print(f"{'kind':26s}" + " ".join(f"{lo:>11d}" for lo, _ in W))
for kd in kinds:
    print(f"{kd:26s}" + " ".join(f"{rss(ka,va,ia,'grad/norm/',kd,(st>=lo)&(st<hi)):5.2f}/{rss(kb,vb,ib,'grad/norm/',kd,(st>=lo)&(st<hi)):5.2f}" for lo, hi in W))
print("\nroot sum of squared parameter norms by kind, w810 / HC")
pk = sorted({kind(k) for k in ka if k.startswith("params/norm/") and k != "params/norm/total"})
for kd in pk:
    print(f"{kd:26s}" + " ".join(f"{rss(ka,va,ia,'params/norm/',kd,(st>=lo)&(st<hi)):6.2f}/{rss(kb,vb,ib,'params/norm/',kd,(st>=lo)&(st<hi)):6.2f}" for lo, hi in W))
print("\nper-layer write / scale parameter norm of w810 at the end, largest 8")
last = va[-1]
items = sorted(((last[i], n) for n, i in ka.items() if n.startswith("params/norm/") and re.search(r"hyper\.\d\.(write|read|scale)$", n) and np.isfinite(last[i])), reverse=True)[:8]
for v, n in items: print(f"  {n.replace('params/norm/transformer.layers.','')} {v:.3f}")
