"""Learned mixing against no mixing at 12, 24 and 48 layers, interim, 2026-11-05 12:05 EST. Status figure for the owner; not a page figure. The two 24-layer runs are still training.

Data: ../../pls_hcid/live_depth_1105.npz (W&B train/pls/L{L-1} and eval/L{L-1}/loss, live_depth_1105.py).
Left: D = no mixing minus learned mixing, last layer, train cross-entropy, 100-step bins, steps both runs have. Right: the same difference on held-out loss at the evaluation steps.
Output: depth_live_1105.{pdf,png}."""
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
Z = np.load(HERE.parents[1] / "pls_hcid/live_depth_1105.npz")
COL = {12: tone("blue", 300), 24: tone("blue", 600), 48: tone("blue", 900)}
legend = [("12 layers", COL[12]), ("24 layers, still training", COL[24]), ("48 layers", COL[48])]
fig, axes = canvas(1, 2, width=WIDTH_TEXT, panel_height=2.0, title="No mixing minus learned mixing, by number of layers", legend=legend)
a, b = np.ravel(axes)
for L in (12, 24, 48):
    sa, ya, sb, yb = Z[f"id{L}/step"], Z[f"id{L}/ce"], Z[f"hc{L}/step"], Z[f"hc{L}/ce"]
    xs, ys = [], []
    for lo in range(500, 5000, 100):
        ma, mb = (sa >= lo) & (sa < lo + 100), (sb >= lo) & (sb < lo + 100)
        if ma.sum() >= 50 and mb.sum() >= 50:
            xs.append(lo + 50); ys.append(ya[ma].mean() - yb[mb].mean())
    a.plot(xs, ys, color=COL[L], lw=1.2)
    ea, va, eb, vb = Z[f"id{L}/es"], Z[f"id{L}/ey"], Z[f"hc{L}/es"], Z[f"hc{L}/ey"]
    common = sorted(set(ea.tolist()) & set(eb.tolist()))
    dy = [va[ea == s][0] - vb[eb == s][0] for s in common]
    b.plot(common, dy, color=COL[L], lw=1.4, marker="o", ms=3.2)
    print(L, "held-out", [(s, round(float(d), 3)) for s, d in zip(common, dy)], "train last bin", xs[-1], round(float(ys[-1]), 3))
for ax, lo, hi, q in ((a, -0.04, 0.06, "Train, last layer (nats)"), (b, -0.08, 0.08, "Held-out, last layer (nats)")):
    ax.axhline(0, color=GREY, lw=1.0)
    nice_y(ax, lo, hi, fmt="{:+.2f}")
    ax.set_xlim(0, 5000)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    room(ax)
    panel_label(ax, q)
save(fig, HERE / "depth_live_1105")
