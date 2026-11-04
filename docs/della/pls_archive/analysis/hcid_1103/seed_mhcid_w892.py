"""Seed 1 (w892) against seed 0 of mHC mixing = I + linear read/write, ordinary model, from d48g4.npz: train CE by layer, eval by layer, and the mean over steps 2,100-2,197 used in the 2 x 2."""
import numpy as np
d = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
a, b = "-sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "-sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48-s1"
L = [2, 6, 12, 18, 24, 30, 36, 42, 47, 48]
print("layer            ", L)
print("final seed 0     ", np.round(d[a + "/final"][[k - 1 for k in L]], 3).tolist())
print("final seed 1     ", np.round(d[b + "/final"][[k - 1 for k in L]], 3).tolist())
print("eval  seed 0     ", np.round(d[a + "/eval"][[k - 1 for k in L]], 3).tolist())
print("eval  seed 1     ", np.round(d[b + "/eval"][[k - 1 for k in L]], 3).tolist())
dif = d[b + "/final"] - d[a + "/final"]
print("final s1 - s0: mean |dif| L2-48 %.3f, max |dif| %.3f at layer %d" % (np.abs(dif[1:]).mean(), np.abs(dif[1:]).max(), 2 + int(np.abs(dif[1:]).argmax())))
de = d[b + "/eval"] - d[a + "/eval"]
print("eval  s1 - s0: mean |dif| L2-48 %.3f, max |dif| %.3f at layer %d" % (np.abs(de[1:]).mean(), np.abs(de[1:]).max(), 2 + int(np.abs(de[1:]).argmax())))
for t in (a, b):
    st, ce = d[t + "/step"], d[t + "/ce"]
    m = (st >= 2100) & (st <= 2197)
    print(t[-6:], "steps 2100-2197 L2 L12 L24 L48", np.round(ce[m].mean(0)[[1, 11, 23, 47]], 2).tolist(), "n", int(m.sum()))
hc = "-sep-bbfrozen-ahc-fa4thd-vs-d48"
if hc + "/final" in d:
    print("HC seed 0 final L12 L24 L48", np.round(d[hc + "/final"][[11, 23, 47]], 3).tolist(), "eval", np.round(d[hc + "/eval"][[11, 23, 47]], 3).tolist())
