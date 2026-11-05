"""The last layers of the main designs, train and eval, ordinary (probes) and separate heads: is the sign change at the last layer real, or an artifact of the probes on held-out text?"""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
T = [("Sandwich-LN", "vs-d48-g4"), ("Pre-LN", "apreln-vs-d48-g4"), ("LNS", "alns-vs-d48-g4"), ("DeepNorm", "adeepnorm-vs-d48-g4"), ("KEEL", "akeel-vs-d48-g4"), ("HC", "ahc-vs-d48-g4"), ("mHC", "amhc-vs-d48-g4"), ("MoDA", "amoda-vs-d48-g4"), ("AttnRes", "aattnres-vs-d48-s1-g4"), ("AttnRes Block", "aattnres_block-vs-d48-g4")]
LL = (2, 6, 12, 24, 36, 42, 44, 45, 46, 47, 48)
print("gap = separate heads minus ordinary, by layer " + " ".join(f"L{l:<5d}" for l in LL))
for n, t in T:
    o, a = z[f"-sep-bbfrozen-{t}/final"], z[f"-sep-{t}/final"]
    eo, ea = z[f"-sep-bbfrozen-{t}/eval"], z[f"-sep-{t}/eval"]
    print(f"{n:14s} train " + " ".join(f"{a[l - 1] - o[l - 1]:+.3f}" for l in LL))
    print(f"{'':14s} eval  " + " ".join(f"{ea[l - 1] - eo[l - 1]:+.3f}" for l in LL))
    print(f"{'':14s} ord e " + " ".join(f"{eo[l - 1]:6.3f}" for l in LL))
    print(f"{'':14s} sep e " + " ".join(f"{ea[l - 1]:6.3f}" for l in LL))
