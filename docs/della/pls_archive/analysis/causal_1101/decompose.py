"""The separate-heads cost is a difference: L48 final of the separate-heads model (sep) minus that of the probes model
(probes = the ordinary model, backbone trained on the last layer's loss only). For the HC / mHC family of the causal test,
print both terms relative to Pre-LN on the same stack (d48g4.npz <tag>/final[47], mean of the last 50 logged steps), so it
is visible which term moves the cost. fa4thd designs are compared with Pre-LN fa4thd, the others with Pre-LN g4.
Read-only. 2026-11-01."""
import numpy as np
d = dict(np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True))
D = [("Pre-LN", "apreln-vs-d48-g4"), ("Pre-LN seed 1", "apreln-vs-d48-s1"), ("Pre-LN fa4thd", "apreln-fa4thd-vs-d48"),
     ("HC", "ahc-vs-d48-g4"), ("HC seed 1", "ahc-vs-d48-s1"), ("HC fa4thd", "ahc-fa4thd-vs-d48"),
     ("HC sigmoid r/w", "ahc-hcrwsig-vs-d48"), ("HC 8 streams", "ahc-hcs8-vs-d48"), ("HC 2 streams", "ahc-hcs2-vs-d48"),
     ("HC 1 stream", "ahc-hcs1-vs-d48"), ("HC static", "ahc-rstatic-vs-d48"), ("HC shared write", "ahc-hcwsh-fa4thd-vs-d48"),
     ("mHC", "amhc-vs-d48-g4"), ("mHC mixing=I", "amhc-mhcid-vs-d48"), ("mHC linear r/w", "amhc-mhcrwlin-vs-d48"),
     ("mHC I + linear", "amhc-mhcid-mhcrwlin-fa4thd-vs-d48")]


def fin(arm, tag):
    k = f"-sep{arm}-{tag}/final"
    return float(d[k][47]) if k in d else None


ref = {"g4": (fin("", "apreln-vs-d48-g4"), fin("-bbfrozen", "apreln-vs-d48-g4")),
       "fa4thd": (fin("", "apreln-fa4thd-vs-d48"), fin("-bbfrozen", "apreln-fa4thd-vs-d48"))}
print(f"{'design':17s} {'sep':>6s} {'probes':>6s} {'cost':>6s}   {'d sep':>6s} {'d prb':>6s}  (d = minus Pre-LN on the same stack)")
for name, tag in D:
    s, p = fin("", tag), fin("-bbfrozen", tag)
    rs, rp = ref["fa4thd" if "fa4thd" in tag else "g4"]
    f = lambda v: f"{v:6.3f}" if v is not None else "     -"
    g = lambda v, r: f"{v - r:+6.3f}" if v is not None else "     -"
    print(f"{name:17s} {f(s)} {f(p)} {f(s - p) if s is not None and p is not None else '     -'}   {g(s, rs)} {g(p, rp)}")
