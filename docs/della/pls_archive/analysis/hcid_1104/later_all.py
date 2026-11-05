"""How general is it? Layer 12 of every 48-layer ordinary model against the ordinary 12-layer model of the same design where it exists; and the same check in training over time for Pre-LN: is layer 12 of the 48-layer model still catching up at the end?"""
import json, numpy as np
from pathlib import Path
B = Path("/scratch/gpfs/GROUP/USER/tmp/pls_blog")
z = np.load(B / "d48g4.npz", allow_pickle=True); ref = json.load(open(B / "d48ref.json"))
print("ordinary 12-layer: Sandwich-LN", round(ref["12"]["train"], 3), "Pre-LN", round(float(z["-sep-bbfrozen-apreln-vs-d12/final"][-1]), 3), "HC", round(float(z["-sep-bbfrozen-ahc-vs-d12/final"][-1]), 3))
for n, t in (("Sandwich-LN", "vs-d48-g4"), ("Pre-LN", "apreln-vs-d48-g4"), ("LNS", "alns-vs-d48-g4"), ("DeepNorm", "adeepnorm-vs-d48-g4"), ("KEEL", "akeel-vs-d48-g4"), ("MoDA", "amoda-vs-d48-g4"), ("HC", "ahc-vs-d48-g4"), ("mHC", "amhc-vs-d48-g4"), ("AttnRes", "aattnres-vs-d48-s1-g4"), ("AttnRes Block", "aattnres_block-vs-d48-g4")):
    f = z[f"-sep-bbfrozen-{t}/final"]; s = z[f"-sep-{t}/final"]
    print(f"{n:14s} 48-layer model, ordinary: L12 {f[11]:.3f} L24 {f[23]:.3f} L36 {f[35]:.3f} L48 {f[47]:.3f} | every layer: L12 {s[11]:.3f} L24 {s[23]:.3f} L36 {s[35]:.3f} L48 {s[47]:.3f}")
s, c = z["-sep-bbfrozen-apreln-vs-d48-g4/step"], z["-sep-bbfrozen-apreln-vs-d48-g4/ce"]
print("Pre-LN 48 layers, L12 by step:", [(a, round(float(c[(s >= a) & (s < a + 100), 11].mean()), 3)) for a in (500, 1000, 2000, 3000, 4000, 4850)])
s, c = z["-sep-bbfrozen-apreln-vs-d12/step"], z["-sep-bbfrozen-apreln-vs-d12/ce"]
print("Pre-LN 12 layers, L12 by step:", [(a, round(float(c[(s >= a) & (s < a + 100), 11].mean()), 3)) for a in (500, 1000, 2000, 3000, 4000, 4850)])
