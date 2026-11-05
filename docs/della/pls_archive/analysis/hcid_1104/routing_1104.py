"""Candidate core figure of a blog: extra residual paths help, learned routing does not. Measured results; each dot one run; 48 layers, width 512, same recipe and 2.6B tokens.

Data: ../../pls_blog/d48g4.npz, eval/L47/loss, the last layer's held-out loss of ordinary models (trained on the last layer only). With routing: HC runs -ahc-vs-d48-g4, -s1, -fa4thd; AttnRes (Full) -aattnres-vs-d48-g4, -s1-g4; mHC -amhc-vs-d48-g4. Without: HC mixing = I, two runs; AttnRes static routing, one run; mHC mixing = I, one run. The grey band spans the four Pre-LN and Sandwich-LN runs with one path. The HC run that overflowed at step 4,410 (9.885) is left out and stated in the caption.
Output: routing_1104.{pdf,png}."""
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
Z = np.load(HERE.parents[1] / "pls_blog/d48g4.npz", allow_pickle=True)
ev = lambda t: float(Z[f"-sep-bbfrozen-{t}/eval"][47])
R = [("Hyper-Connections", ["ahc-vs-d48-g4", "ahc-vs-d48-s1", "ahc-fa4thd-vs-d48"], ["ahc-hcid-fa4thd-vs-d48", "ahc-hcid-fa4thd-vs-d48-s1"]),
     ("AttnRes (Full)", ["aattnres-vs-d48-g4", "aattnres-vs-d48-s1-g4"], ["aattnres-rstatic-vs-d48"]),
     ("mHC", ["amhc-vs-d48-g4"], ["amhc-mhcid-vs-d48"])]
band = [ev(t) for t in ("apreln-vs-d48-g4", "apreln-vs-d48-s1", "apreln-fa4thd-vs-d48", "vs-d48-g4")]
fig, axes = canvas(1, 1, width=WIDTH_POST, panel_height=1.5, title="Last-layer held-out loss, 48 layers", legend=[("With learned routing", BLUE), ("Routing removed or fixed", LIGHT), ("One path: Pre-LN, Sandwich-LN", GREY)], quantity="Held-out cross-entropy (nats)", title_pt=9.2, tick_pt=7.35, side=0.10)
a = np.ravel(axes)[0]
a.axvspan(min(band), max(band), color=tone("grey", 300), alpha=0.6, lw=0)
for i, (n, w, wo) in enumerate(R):
    y = -i
    dots(a, [ev(t) for t in w], [y] * len(w), BLUE)
    dots(a, [ev(t) for t in wo], [y] * len(wo), LIGHT)
    a.text(3.465, y, n, ha="right", va="center", fontsize=7.35)
    print(n, "with", [round(ev(t), 3) for t in w], "without", [round(ev(t), 3) for t in wo])
a.set_xlim(3.30, 3.76); a.set_ylim(-2.6, 0.9); a.set_yticks([])
a.set_xticks([3.5, 3.6, 3.7])
save(fig, HERE / "routing_1104")
