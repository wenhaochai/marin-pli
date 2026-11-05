"""Candidate core figure of a blog: last-layer held-out loss by depth, ordinary models against models with a next-token loss at every layer. Contains measured results only; one run per point. Not a page figure yet.

Data: ../../pls_blog/d48g4.npz (eval/L{k}/loss of the last layer: Pre-LN and HC at 12, 24, 48 layers, runs -sep-bbfrozen-<design> ordinary, -sep-<design> every layer; Sandwich-LN at 48), ../../pls_blog/d48ref.json (ordinary Sandwich-LN at 12, 24, 36). Same width 512, recipe and 2.6B tokens at every depth. x axis log2 depth.
Output: depth_gain_1104.{pdf,png}."""
import json
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
B = HERE.parents[1] / "pls_blog"
Z = np.load(B / "d48g4.npz", allow_pickle=True)
ref = json.load(open(B / "d48ref.json"))
def ev(t):
    return float(Z[t + "/eval"][-1])
P = [("Sandwich-LN", {12: ref["12"]["eval"], 24: ref["24"]["eval"], 36: ref["36"]["eval"], 48: ev("-sep-bbfrozen-vs-d48-g4")}, {48: ev("-sep-vs-d48-g4")}),
     ("Pre-LN", {d: ev(f"-sep-bbfrozen-apreln-vs-d{d}" + ("-g4" if d == 48 else "")) for d in (12, 24, 48)}, {d: ev(f"-sep-apreln-vs-d{d}" + ("-g4" if d == 48 else "")) for d in (12, 24, 48)}),
     ("Hyper-Connections", {d: ev(f"-sep-bbfrozen-ahc-vs-d{d}" + ("-g4" if d == 48 else "")) for d in (12, 24, 48)}, {d: ev(f"-sep-ahc-vs-d{d}" + ("-g4" if d == 48 else "")) for d in (12, 24, 48)})]
fig, axes = canvas(1, 3, width=WIDTH_TEXT, panel_height=2.1, title="Last-layer held-out loss by depth", legend=[("Ordinary model", LIGHT), ("A next-token loss at every layer", BLUE)], quantity="Held-out cross-entropy (nats)")
for a, (name, o, s) in zip(np.ravel(axes), P):
    x = sorted(o); a.plot(np.log2(x), [o[d] for d in x], color=LIGHT, lw=1.6, marker="o", ms=3.5)
    x = sorted(s); a.plot(np.log2(x), [s[d] for d in x], color=BLUE, lw=1.6, marker="o", ms=3.5)
    nice_y(a, 3.55, 4.05, fmt="{:.1f}")
    a.set_xlim(np.log2(10), np.log2(56))
    a.set_xticks(np.log2([12, 24, 48])); a.set_xticklabels(["12", "24", "48"])
    room(a)
    panel_label(a, name)
    print(name, {k: round(v, 3) for k, v in o.items()}, {k: round(v, 3) for k, v in s.items()}, "gain 12 to 48", round(o[12] - o[48], 3), "cost at 48", round(s[48] - o[48], 3), "share", round((s[48] - o[48]) / (o[12] - o[48]), 2))
save(fig, HERE / "depth_gain_1104")
