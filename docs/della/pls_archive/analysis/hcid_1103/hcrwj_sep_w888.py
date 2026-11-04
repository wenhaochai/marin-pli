"""w888 (HC with mHC's joint read/write dynamic terms, separate heads) against HC, HC mixing = I (w810) and mHC mixing = I + linear read/write, from d48g4.npz: train and eval CE by layer."""
import numpy as np
d = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
R = {"w888 hcrwj": "-sep-ahc-hcrwj-fa4thd-vs-d48", "HC": "-sep-ahc-fa4thd-vs-d48", "w810 hcid": "-sep-ahc-hcid-fa4thd-vs-d48", "mHC id-lin": "-sep-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"}
L = [2, 6, 12, 24, 36, 47, 48]
print("layer", L)
for n, t in R.items():
    if t + "/final" not in d:
        print(n, "missing"); continue
    print(f"{n:11s} train", np.round(d[t + "/final"][[k - 1 for k in L]], 3).tolist(), "eval", np.round(d[t + "/eval"][[k - 1 for k in L]], 3).tolist(), "params", int(d[t + "/params"]))
st = d[R["w888 hcrwj"] + "/step"]
for n in ("w888 hcrwj", "HC", "w810 hcid", "mHC id-lin"):
    t = R[n]
    if t + "/final" not in d: continue
    s, c = d[t + "/step"], d[t + "/ce"][:, 47]
    print(n, "L48 mean by 500 steps", [round(float(c[(s >= a) & (s < a + 500)].mean()), 3) for a in range(1000, 5000, 500)])
