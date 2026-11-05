"""The collapse on relative depth, more checks. (1) Held-out loss of the same probes (eval/L{k}/loss). (2) Pre-LN seed 1 and rerun at 48 layers against the 12- and 24-layer curves, train. (3) The step at which a layer's curve is read: the same collapse at steps 1000-1099 and 3000-3099. Mean absolute difference against relative depth (40 points, f from 0.25 to 1) and against the layer index on the shared range."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
F = np.linspace(0.25, 1.0, 40)
def cmp(ca, cb):
    a, b = len(ca), len(cb)
    return np.abs(np.interp(F, (np.arange(a) + 1) / a, ca) - np.interp(F, (np.arange(b) + 1) / b, cb)).mean(), np.abs(ca[:a] - cb[:a]).mean()
T = {12: "-sep-bbfrozen-apreln-vs-d12", 24: "-sep-bbfrozen-apreln-vs-d24", 48: "-sep-bbfrozen-apreln-vs-d48-g4"}
print("held-out, Pre-LN")
E = {d: z[t + "/eval"] for d, t in T.items()}
for f in (0.25, 0.5, 0.75, 0.9, 1.0):
    print(f"  f {f:4.2f} " + " ".join(f"{np.interp(f, (np.arange(d) + 1) / d, E[d]):7.3f}" for d in E))
print("  12 vs 24: relative %.3f, index %.3f; 24 vs 48: relative %.3f, index %.3f" % (*cmp(E[12], E[24]), *cmp(E[24], E[48])))
print("train by step, Pre-LN")
for lo, hi in ((500, 600), (1000, 1100), (2000, 2100), (3000, 3100), (4000, 4100)):
    W = {}
    for d, t in T.items():
        s, c = z[t + "/step"], z[t + "/ce"]
        W[d] = c[(s >= lo) & (s < hi)].mean(0)[:d]
    print(f"  steps {lo}-{hi - 1}: f=1/4 " + " ".join(f"{np.interp(0.25, (np.arange(d) + 1) / d, W[d]):.3f}" for d in W) + "; f=1/2 " + " ".join(f"{np.interp(0.5, (np.arange(d) + 1) / d, W[d]):.3f}" for d in W) + "; 12 vs 24 relative %.3f index %.3f; 24 vs 48 relative %.3f index %.3f" % (*cmp(W[12], W[24]), *cmp(W[24], W[48])))
print("other 48-layer Pre-LN runs against the 24-layer curve, train")
for t in ("-sep-bbfrozen-apreln-vs-d48-s1", "-sep-bbfrozen-apreln-vs-d48-rerun", "-sep-bbfrozen-apreln-fa4thd-vs-d48"):
    print("  ", t, "relative %.3f index %.3f" % cmp(z[T[24] + "/final"], z[t + "/final"]))
print("every-layer curves, Pre-LN: by layer index and by relative depth")
S = {12: z["-sep-apreln-vs-d12/final"], 24: z["-sep-apreln-vs-d24/final"], 48: z["-sep-apreln-vs-d48-g4/final"]}
print("  12 vs 24: relative %.3f, index %.3f; 24 vs 48: relative %.3f, index %.3f" % (*cmp(S[12], S[24]), *cmp(S[24], S[48])))
S = {12: z["-sep-ahc-vs-d12/final"], 24: z["-sep-ahc-vs-d24/final"], 48: z["-sep-ahc-vs-d48-g4/final"]}
print("every-layer curves, HC: 12 vs 24: relative %.3f, index %.3f; 24 vs 48: relative %.3f, index %.3f" % (*cmp(S[12], S[24]), *cmp(S[24], S[48])))
