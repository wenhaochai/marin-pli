"""Where does w893 (HC mixing = I, seed 1, separate heads) leave w810 (seed 0) and HC, separate heads? 100-step means of the train CE at L2, L12, L24, L48 minus HC at the same steps, from live_1104.npz and d48g4.npz.
Also the 3 largest grad rows per 500 steps (live_1104.npz keeps grad/norm/total of w893, w810, HC, one row per 10 steps), and the cost so far: separate-heads L48 minus ordinary L48 at the same steps, for HC, mixing = I and hcrwj."""
import numpy as np
L = np.load("live_1104.npz", allow_pickle=True)
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
lay = [int(x) for x in L["lay"]]
ix = {l: lay.index(l) for l in lay}
def win(st, y, lo, hi):
    m = (st >= lo) & (st < hi)
    return y[m].mean() if m.any() else np.nan
HC = "-sep-ahc-fa4thd-vs-d48"; ID = "-sep-ahc-hcid-fa4thd-vs-d48"
rs, rc = z[HC + "/step"], z[HC + "/ce"]; ds, dc = z[ID + "/step"], z[ID + "/ce"]
ws, wc = L["w893 hcid s1 sep/step"], L["w893 hcid s1 sep/ce"]
print("steps      | w893 minus HC: L2 L12 L24 L48 | w810 minus HC: L2 L12 L24 L48")
for lo in range(1000, 4600, 200):
    hi = lo + 200
    a = " ".join(f"{win(ws, wc[:, ix[l]], lo, hi) - win(rs, rc[:, l - 1], lo, hi):+6.3f}" for l in (2, 12, 24, 48))
    b = " ".join(f"{win(ds, dc[:, l - 1], lo, hi) - win(rs, rc[:, l - 1], lo, hi):+6.3f}" for l in (2, 12, 24, 48))
    print(f"{lo:>5}-{hi - 1:<5}| {a} | {b}")
print("\nlargest grad rows")
for n in ("w893", "w810"):
    a = L[n + "/grad"]
    for lo in range(1500, 5000, 500):
        q = a[(a[:, 0] >= lo) & (a[:, 0] < lo + 500)]
        if not len(q):
            continue
        t = q[np.argsort(-q[:, 1])[:3]]
        print(f"  {n} {lo}-{lo + 499}: " + ", ".join(f"step {int(s)} {v:.1f}" for s, v in t) + f"; rows above 1.5 x median {int((q[:, 1] > 1.5 * np.median(q[:, 1])).sum())}/{len(q)}")
print("\ncost so far, L48 mean over the window: separate heads minus ordinary at the same steps")
PAIRS = {"HC": (HC, "-sep-bbfrozen-ahc-fa4thd-vs-d48"), "mixing = I, seed 0": (ID, "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48")}
def ser(t):
    return z[t + "/step"], z[t + "/ce"][:, 47]
G = {"HC": lambda: (ser(HC), ser("-sep-bbfrozen-ahc-fa4thd-vs-d48")),
     "mixing = I, s0 (w810, w811)": lambda: (ser(ID), ser("-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48")),
     "mixing = I, s1 (w893, w891)": lambda: ((ws, wc[:, ix[48]]), (L["w891 hcid s1/step"], L["w891 hcid s1/ce"][:, ix[48]])),
     "hcrwj (w888, w889)": lambda: (ser("-sep-ahc-hcrwj-fa4thd-vs-d48"), (L["w889 hcrwj/step"], L["w889 hcrwj/ce"][:, ix[48]])),
     "HC seed 1 s0 sep, s1 ord (HC, w890)": lambda: (ser(HC), (L["w890 HC s1/step"], L["w890 HC s1/ce"][:, ix[48]]))}
W = [(1000, 1500), (2000, 2500), (2800, 3300), (3200, 3392), (3392, 3680), (4000, 4500)]
print("  " + " " * 37 + " ".join(f"{a}-{b - 1:<5}" for a, b in W))
for n, f in G.items():
    (a, ay), (b, by) = f()
    o = f"  {n:37s}"
    for lo, hi in W:
        ok = lambda s: ((s >= lo) & (s < hi)).sum() >= 0.8 * (hi - lo)
        o += f" {win(a, ay, lo, hi) - win(b, by, lo, hi):+9.3f}" if ok(a) and ok(b) else "         -"
    print(o)
