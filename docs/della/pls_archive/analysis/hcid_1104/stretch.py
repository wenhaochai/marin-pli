"""Do models of different depth trace the same curve when each layer is placed at its relative depth l / d? Ordinary models, probes, final train CE (mean of the last 50 logged steps). Same width 512, recipe and tokens for each depth. For every pair of depths, the mean absolute difference of the curves against l / d (linear interpolation on 40 points, f from 0.25 to 1), and against the absolute layer index on the shared range."""
import numpy as np
from pathlib import Path
B = Path("/scratch/gpfs/GROUP/USER/tmp/pls_blog")
z = np.load(B / "d48g4.npz", allow_pickle=True); bb = np.load(B / "bbfrozen.npz", allow_pickle=True)
C = {"Pre-LN": {12: z["-sep-bbfrozen-apreln-vs-d12/final"], 24: z["-sep-bbfrozen-apreln-vs-d24/final"], 48: z["-sep-bbfrozen-apreln-vs-d48-g4/final"]},
     "HC": {12: z["-sep-bbfrozen-ahc-vs-d12/final"], 24: z["-sep-bbfrozen-ahc-vs-d24/final"], 48: z["-sep-bbfrozen-ahc-vs-d48-g4/final"]},
     "Sandwich-LN": {6: bb["-pls1-sep-bbfrozen/final"], 48: z["-sep-bbfrozen-vs-d48-g4/final"]}}
F = np.linspace(0.25, 1.0, 40)
for name, cur in C.items():
    print(f"\n{name}")
    print("   f     " + " ".join(f"d={d:<5d}" for d in cur))
    for f in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        print(f"   {f:5.3f} " + " ".join(f"{np.interp(f, (np.arange(len(c)) + 1) / len(c), c):7.3f}" for c in cur.values()))
    ds = sorted(cur)
    for a, b in zip(ds, ds[1:]):
        ca, cb = cur[a], cur[b]
        rel = np.abs(np.interp(F, (np.arange(a) + 1) / a, ca) - np.interp(F, (np.arange(b) + 1) / b, cb)).mean()
        ab = np.abs(ca[:a] - cb[:a]).mean()
        print(f"   d={a} vs d={b}: mean |difference| against relative depth {rel:.3f}, against the layer index {ab:.3f}")
