"""What depth is a model with a loss at every layer worth? Ordinary models of several depths, same width, recipe and tokens; for a model with separate heads, its last layer's loss is read on the ordinary loss-by-depth curve (log2 depth, linear between measured depths). Train: final = mean of the last 50 logged steps. Eval: eval/L{k}/loss at the last step."""
import json, numpy as np
from pathlib import Path
B = Path("/scratch/gpfs/GROUP/USER/tmp/pls_blog")
z = np.load(B / "d48g4.npz", allow_pickle=True)
ref = json.load(open(B / "d48ref.json"))
depth = json.load(open(B / "depth.json")); base = json.load(open(B / "baselines.json"))
print("depth.json keys", list(depth)[:12]); k0 = list(depth)[0]; print(k0, str(depth[k0])[:200])
print("layers.npz keys", list(np.load(B / "layers.npz", allow_pickle=True).keys())[:14])
def pair(t):
    return (z[f"-sep-bbfrozen-{t}/final"], z[f"-sep-{t}/final"], z[f"-sep-bbfrozen-{t}/eval"], z[f"-sep-{t}/eval"])
def eff(curve, y):
    ds = sorted(curve); v = [curve[d] for d in ds]
    if y >= v[0]:
        return f"<{ds[0]}" if y > v[0] else str(ds[0])
    for a, b, va, vb in zip(ds, ds[1:], v, v[1:]):
        if vb <= y <= va:
            return f"{a * 2 ** ((np.log2(b) - np.log2(a)) * (va - y) / (va - vb) / 1):.1f}" if False else f"{a * (b / a) ** ((va - y) / (va - vb)):.1f}"
    return f">{ds[-1]}"
fam = {"Sandwich-LN": {12: None, 24: None, 36: None, 48: "vs-d48-g4"}, "Pre-LN": {12: "apreln-vs-d12", 24: "apreln-vs-d24", 48: "apreln-vs-d48-g4"}, "HC": {12: "ahc-vs-d12", 24: "ahc-vs-d24", 48: "ahc-vs-d48-g4"}}
for name, ds in fam.items():
    tr, ev = {}, {}
    for d, t in ds.items():
        if t is None:
            tr[d], ev[d] = ref[str(d)]["train"], ref[str(d)]["eval"]
        else:
            o, a, eo, ea = pair(t)
            tr[d], ev[d] = float(o[-1]), float(eo[-1])
    print(f"\n{name} ordinary by depth: train " + ", ".join(f"{d}: {tr[d]:.3f}" for d in sorted(tr)) + " | eval " + ", ".join(f"{d}: {ev[d]:.3f}" for d in sorted(ev)))
    for d, t in ds.items():
        if t is None:
            continue
        o, a, eo, ea = pair(t)
        print(f"  {d} layers, separate heads: last layer train {a[-1]:.3f} (cost {a[-1] - o[-1]:+.3f}), worth {eff(tr, float(a[-1]))} ordinary layers; eval {ea[-1]:.3f} (cost {ea[-1] - eo[-1]:+.3f}), worth {eff(ev, float(ea[-1]))} ordinary layers")
        idx = [k for k in (3, 6, 9, 12, 18, 24, 36) if k < d]
        print("    separate heads' own layers, eval: " + ", ".join(f"L{k} {ea[k - 1]:.3f}" for k in idx) + f", L{d} {ea[-1]:.3f}; train: " + ", ".join(f"L{k} {a[k - 1]:.3f}" for k in idx) + f", L{d} {a[-1]:.3f}")
