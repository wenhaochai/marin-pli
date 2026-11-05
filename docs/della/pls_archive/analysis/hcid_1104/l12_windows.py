"""Layer 12 and 24 train CE of finished ordinary runs at step windows (d48g4.npz)."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
WIN = [(100, 200), (200, 300), (300, 400), (500, 600), (1000, 1100), (2100, 2197), (3000, 3100), (4000, 4100), (4900, 4958)]
TAGS = [("HC 4 streams", "-sep-bbfrozen-ahc-fa4thd-vs-d48"), ("HC 4 old code", "-sep-bbfrozen-ahc-vs-d48-g4"), ("HC 8", "-sep-bbfrozen-ahc-hcs8-vs-d48"), ("HC 2", "-sep-bbfrozen-ahc-hcs2-vs-d48"), ("HC 1", "-sep-bbfrozen-ahc-hcs1-vs-d48"), ("HC static", "-sep-bbfrozen-ahc-rstatic-vs-d48"), ("learned scalar", "-sep-bbfrozen-ahc-hcscalar-fa4thd-vs-d48"), ("mixing = I", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48"), ("mHC", "-sep-bbfrozen-amhc-vs-d48-g4"), ("Pre-LN", "-sep-bbfrozen-apreln-fa4thd-vs-d48"), ("Sandwich-LN", "-sep-bbfrozen-vs-d48-g4"), ("LNS", "-sep-bbfrozen-alns-vs-d48-g4")]
for l in (12, 24):
    print(f"-- layer {l}:" + " " * 12 + " ".join(f"{a}-{b - 1}".rjust(10) for a, b in WIN))
    for n, t in TAGS:
        if t + "/step" not in z.files:
            print(n, "missing"); continue
        s, c = z[t + "/step"], z[t + "/ce"][:, l - 1]
        print(f"{n:22s}" + " ".join(f"{c[(s >= a) & (s < b)].mean():10.3f}" for a, b in WIN))
