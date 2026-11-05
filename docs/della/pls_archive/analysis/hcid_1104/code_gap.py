"""How much does the code version move a run, at the last layer? Pairs on old and current code with the same design and seed. Also the stream count of the old references at 12 and 24 layers."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
for n, a, b in (("HC 48, seed 0", "ahc-vs-d48-g4", "ahc-fa4thd-vs-d48"), ("Pre-LN 48, seed 0", "apreln-vs-d48-g4", "apreln-fa4thd-vs-d48"), ("HC 48, old code, seed 0 vs 1", "ahc-vs-d48-g4", "ahc-vs-d48-s1"), ("Pre-LN 48, old code, seed 0 vs 1", "apreln-vs-d48-g4", "apreln-vs-d48-s1"), ("HC mixing = I 48, seed 0 vs 1", "ahc-hcid-fa4thd-vs-d48", "ahc-hcid-fa4thd-vs-d48-s1")):
    A, B = z[f"-sep-bbfrozen-{a}/eval"][47], z[f"-sep-bbfrozen-{b}/eval"][47]; TA, TB = z[f"-sep-bbfrozen-{a}/final"][47], z[f"-sep-bbfrozen-{b}/final"][47]
    print(f"{n:34s} held-out {A:.3f} {B:.3f} ({B - A:+.3f}); train {TA:.3f} {TB:.3f} ({TB - TA:+.3f})")
for t in ("ahc-vs-d12", "ahc-vs-d24", "apreln-vs-d12", "apreln-vs-d24"):
    k = f"-sep-bbfrozen-{t}"; print(t, "held-out", round(float(z[k + "/eval"][-1]), 3), "train", round(float(z[k + "/final"][-1]), 3), "steps", int(z[k + "/step"][-1]))
