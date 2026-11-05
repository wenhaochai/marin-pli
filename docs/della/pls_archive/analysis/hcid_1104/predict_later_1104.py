"""Candidate core figure of a blog: deeper models predict later. Measured results; one run per curve; Pre-LN, width 512, same recipe and 2.6B tokens at every depth.

Data: ../../pls_blog/d48g4.npz, final train CE (mean of the last 50 logged steps) of each layer's readout. Ordinary: runs -sep-bbfrozen-apreln-vs-d{12,24}, -sep-bbfrozen-apreln-vs-d48-g4: the backbone trains on the last layer's loss only, probes read every layer. Every layer: runs -sep-apreln-vs-d{12,24}, -sep-apreln-vs-d48-g4: a next-token loss at every layer, each with its own head.
Left: ordinary models by layer. Middle: the same curves with each layer at its relative depth, layer / depth. Right: a loss at every layer, by layer.
Output: predict_later_1104.{pdf,png}."""
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
Z = np.load(HERE.parents[1] / "pls_blog/d48g4.npz", allow_pickle=True)
T = {12: "apreln-vs-d12", 24: "apreln-vs-d24", 48: "apreln-vs-d48-g4"}
C = {12: tone("blue", 300), 24: tone("blue", 500), 48: tone("blue", 900)}
fig, axes = canvas(1, 3, width=WIDTH_TEXT, panel_height=2.2, title="Next-token loss of each layer, Pre-LN models of 12, 24 and 48 layers", legend=[(f"{d} layers", C[d]) for d in T], quantity="Training loss (nats)")
a, b, c = np.ravel(axes)
for d, t in T.items():
    o = Z[f"-sep-bbfrozen-{t}/final"]; s = Z[f"-sep-{t}/final"]; x = np.arange(1, d + 1)
    a.plot(x, o, color=C[d], lw=1.6); b.plot(x / d, o, color=C[d], lw=1.6); c.plot(x, s, color=C[d], lw=1.6)
for ax in (a, c):
    ax.set_xlim(1, 48); ax.set_xticks([1, 12, 24, 48])
b.set_xlim(0, 1); b.set_xticks([0.25, 0.5, 0.75, 1]); b.set_xticklabels(["1/4", "1/2", "3/4", "1"])
for ax, name in ((a, "Ordinary, by layer"), (b, "Ordinary, by layer / depth"), (c, "Loss at every layer, by layer")):
    nice_y(ax, 2.9, 5.4, headroom=0.12, fmt="{:.1f}")
    room(ax)
    panel_label(ax, name)
save(fig, HERE / "predict_later_1104")
for l in (3, 6, 12, 24):
    print("L", l, "ordinary", [round(float(Z[f"-sep-bbfrozen-{t}/final"][l - 1]), 3) if l <= d else None for d, t in T.items()], "every layer", [round(float(Z[f"-sep-{t}/final"][l - 1]), 3) if l <= d else None for d, t in T.items()])
