"""w973 (only shrinks, separate heads) layer 48 and layer 12 by 100 steps from step 1000, next to HC and mixing = I with separate heads; live_1104d.npz, d48g4.npz."""
import numpy as np
L = np.load("live_1104d.npz", allow_pickle=True); Z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True); lay = [int(v) for v in L["lay"]]
s, c = L["w973 only shrinks/step"], L["w973 only shrinks/ce"]
hs, hc = Z["-sep-ahc-fa4thd-vs-d48/step"], Z["-sep-ahc-fa4thd-vs-d48/ce"]
print("window  w973 L2 L12 L24 L36 L48 | HC L48 | max L48 in window")
for lo in range(900, int(s[-1]) - 40, 100):
    m = (s >= lo) & (s < lo + 100); m2 = (hs >= lo) & (hs < lo + 100)
    print(lo, " ".join(f"{c[m, i].mean():.3f}" for i in range(len(lay))), f"| {hc[m2, 47].mean():.3f} | {c[m, 4].max():.3f}")
print(s[-1], np.round(c[-1], 3).tolist())
