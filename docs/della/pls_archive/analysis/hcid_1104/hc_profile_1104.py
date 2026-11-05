"""Candidate blog figure: same last layer, different middle. Status figure for the owner; not a page figure.

Data: ../../pls_blog/d48g4.npz (W&B train/pls/L{k}, final = mean of the last 50 logged steps; ordinary models, finished runs: the backbone trains on the last layer's loss only, the probes read every layer).
Plotted: final train cross-entropy of each layer's probe by layer, 48 layers, width 512.
Output: hc_profile_1104.{pdf,png}."""
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
Z = np.load(HERE.parents[1] / "pls_blog/d48g4.npz", allow_pickle=True)
PUR, GRN, RED = tone("purple", 600), tone("green", 600), tone("red", 600)
S = [("HC", "-sep-bbfrozen-ahc-fa4thd-vs-d48", GREY), ("HC, 1 stream", "-sep-bbfrozen-ahc-hcs1-vs-d48", BLUE), ("Pre-LN", "-sep-bbfrozen-apreln-fa4thd-vs-d48", GRN), ("HC, mixing = I", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", PUR)]
fig, axes = canvas(1, 1, width=WIDTH_POST, panel_height=2.3, title="Probe loss by layer, ordinary models", legend=[(n, c) for n, _, c in S], quantity="Train cross-entropy (nats)", title_pt=9.2, tick_pt=7.35, side=0.10)
a = np.ravel(axes)[0]
x = np.arange(1, 49)
for n, t, c in S:
    a.plot(x, Z[t + "/final"], color=c, lw=1.4)
    print(n, "L12", round(float(Z[t + "/final"][11]), 3), "L24", round(float(Z[t + "/final"][23]), 3), "L36", round(float(Z[t + "/final"][35]), 3), "L48", round(float(Z[t + "/final"][47]), 3))
nice_y(a, 2.9, 5.5, fmt="{:.1f}")
a.set_xlim(1, 48)
a.set_xticks([1, 12, 24, 36, 48])
room(a)
save(fig, HERE / "hc_profile_1104")
