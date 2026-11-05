"""Final train CE by layer, ordinary models, the HC family and references (d48g4.npz)."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
TAGS = [("HC4", "-sep-bbfrozen-ahc-fa4thd-vs-d48"), ("HC4old", "-sep-bbfrozen-ahc-vs-d48-g4"), ("HC4s1", "-sep-bbfrozen-ahc-vs-d48-s1"), ("HC8", "-sep-bbfrozen-ahc-hcs8-vs-d48"), ("HC2", "-sep-bbfrozen-ahc-hcs2-vs-d48"), ("HC1", "-sep-bbfrozen-ahc-hcs1-vs-d48"), ("HCstat", "-sep-bbfrozen-ahc-rstatic-vs-d48"), ("rwj", "-sep-bbfrozen-ahc-hcrwj-fa4thd-vs-d48"), ("scalar", "-sep-bbfrozen-ahc-hcscalar-fa4thd-vs-d48"), ("I", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48"), ("Is1", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48-s1"), ("moves", "-sep-bbfrozen-ahc-hcmove-fa4thd-vs-d48"), ("mHCid", "-sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"), ("PreLN", "-sep-bbfrozen-apreln-fa4thd-vs-d48"), ("Sandw", "-sep-bbfrozen-vs-d48-g4"), ("LNS", "-sep-bbfrozen-alns-vs-d48-g4"), ("KEEL", "-sep-bbfrozen-akeel-vs-d48-g4"), ("MoDA", "-sep-bbfrozen-amoda-vs-d48-g4")]
print("layer " + " ".join(f"{n:>6s}" for n, _ in TAGS))
for l in list(range(1, 17)) + [18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 47, 48]:
    print(f"L{l:<4d} " + " ".join(f"{z[t + '/final'][l - 1]:6.2f}" for _, t in TAGS))
