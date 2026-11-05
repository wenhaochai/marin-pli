"""Does a layer know less when more layers sit above it? Ordinary models (backbone trained on the last layer only, probes read every layer), same width, recipe and tokens, depth 6 to 48. Train CE of the probe at layer l in a model of depth d. Also by distance from the end. Train only: probes' held-out loss has an unexplained extra gap."""
import json, numpy as np
from pathlib import Path
B = Path("/scratch/gpfs/GROUP/USER/tmp/pls_blog")
z = np.load(B / "d48g4.npz", allow_pickle=True)
ref = json.load(open(B / "d48ref.json")); depth = json.load(open(B / "depth.json")); base = json.load(open(B / "baselines.json"))
bb = np.load(B / "bbfrozen.npz", allow_pickle=True); print("bbfrozen keys", [k for k in bb.files][:12])
keys = [k for k in z.files if k.startswith("-sep-bbfrozen") and k.endswith("/final") and ("d12" in k or "d24" in k or "d36" in k)]
print("other depths with probes", keys)
for fam, tags in (("Pre-LN", {12: "apreln-vs-d12", 24: "apreln-vs-d24", 48: "apreln-vs-d48-g4"}), ("HC", {12: "ahc-vs-d12", 24: "ahc-vs-d24", 48: "ahc-vs-d48-g4"})):
    F = {d: z[f"-sep-bbfrozen-{t}/final"] for d, t in tags.items()}
    print(f"\n{fam}: train CE of the probe at layer l, by model depth")
    print("   layer " + " ".join(f"d={d:<5d}" for d in F))
    for l in (1, 2, 3, 4, 6, 8, 10, 12, 16, 18, 20, 24, 30, 36, 42, 48):
        print(f"   L{l:<4d} " + " ".join(f"{F[d][l - 1]:7.3f}" if l <= d else "      -" for d in F))
    print("   by layers remaining above (0 = last layer)")
    for r in (0, 1, 2, 3, 4, 6, 8, 11):
        print(f"   r={r:<3d} " + " ".join(f"{F[d][d - 1 - r]:7.3f}" for d in F))
print("\nSandwich-LN: ordinary d: last layer train", {k: round(v["train"], 3) for k, v in ref.items()}, "48-layer model's probes at L12, L24, L36:", [round(float(z["-sep-bbfrozen-vs-d48-g4/final"][l - 1]), 3) for l in (12, 24, 36, 48)])
print("6 layers: ordinary d1-d5 train", {k: round(v["loss"], 3) for k, v in depth.items()}, "6-layer", round(base["130m"]["loss"], 3))
L = np.load(B / "layers.npz", allow_pickle=True)
print("bbfrozen npz arrays", {k: bb[k].shape for k in bb.files if k.endswith("ce")})
