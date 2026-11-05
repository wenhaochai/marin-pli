"""Learned mixing and extra residual paths at 12, 24 and 48 layers, final, 2026-11-05. Status figure for the owner; not a page figure.

Data: ../../pls_hcid/final_depth_1105.npz (W&B eval/L{L-1}/loss of the ordinary models with probes, final_depth_1105.py) and the Pre-LN values printed by that script: -apreln-vs-d12 3.8446, -apreln-vs-d24 3.7447, -apreln-vs-d48-g4 3.6678 and -apreln-fa4thd-vs-d48 3.6736 (the first three on the older code).
Left: final held-out loss of the last layer by number of layers. Lines join the means; dots are the runs. Right: D = no mixing minus learned mixing at the evaluation steps, on the pair of runs on the same code.
Output: depth_final_1105.{pdf,png}."""
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
Z = np.load(HERE.parents[1] / "pls_hcid/final_depth_1105.npz")
fin = lambda n: float(Z[n + "/ey"][-1])
X = {12: 0, 24: 1, 48: 2}
V = {"Pre-LN, one path": ({12: [3.8446], 24: [3.7447], 48: [3.6678, 3.6736]}, GREY),
     "HC, no mixing": ({12: [fin("id12")], 24: [fin("id24")], 48: [fin("id48"), fin("id48s1")]}, BLUE),
     "HC, learned mixing": ({12: [fin("hc12")], 24: [fin("hc24")], 48: [fin("hc48"), fin("hc48old"), fin("hc48s1")]}, tone("red", 600))}
legend = [(k, c) for k, (_, c) in V.items()]
fig, axes = canvas(1, 2, width=WIDTH_TEXT, panel_height=2.2, title="Held-out loss of the last layer at 12, 24 and 48 layers", legend=legend)
a, b = np.ravel(axes)
for name, (d, col) in V.items():
    a.plot([X[L] for L in d], [np.mean(d[L]) for L in d], color=col, lw=1.4)
    for L in d:
        dots(a, [X[L]] * len(d[L]), d[L], col)
    print(name, {L: [round(v, 3) for v in d[L]] for L in d}, "means", {L: round(float(np.mean(d[L])), 3) for L in d})
nice_y(a, 3.5, 3.9, fmt="{:.2f}")
a.set_xlim(-0.25, 2.25)
a.set_xticks([0, 1, 2])
a.set_xticklabels(["12 layers", "24 layers", "48 layers"])
a.grid(False, axis="x")
panel_label(a, "Final (nats)")
COL = {12: tone("purple", 300), 24: tone("purple", 600), 48: tone("purple", 900)}
b.axhline(0, color=GREY, lw=1.0)
ends = []
for L in (12, 24, 48):
    ea, va, eb, vb = Z[f"id{L}/es"], Z[f"id{L}/ey"], Z[f"hc{L}/es"], Z[f"hc{L}/ey"]
    common = sorted(set(ea.tolist()) & set(eb.tolist()))
    dy = [va[ea == s][0] - vb[eb == s][0] for s in common]
    b.plot(common, dy, color=COL[L], lw=1.4, marker="o", ms=3.2)
    ends.append((float(dy[-1]), f"{L} layers", COL[L]))
    print(L, [(s, round(float(d), 3)) for s, d in zip(common, dy)])
nice_y(b, -0.08, 0.08, fmt="{:+.2f}")
b.set_xlim(0, 5000)
b.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
room(b)
panel_label(b, "No mixing minus learned mixing (nats)")
end_labels(b, ends)
save(fig, HERE / "depth_final_1105")
