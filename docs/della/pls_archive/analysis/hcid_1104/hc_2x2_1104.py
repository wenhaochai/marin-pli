"""The 2 x 2 on HC's mixing, interim, 2026-11-04 13:30 EST. Status figure for the owner; not a page figure. Runs still training.

Data: ../../pls_hcid/live_1104c.npz (W&B train/pls/L{k} of the running runs, live_1104c.py), ../../pls_blog/d48g4.npz (finished: HC, mixing = I).
Left: layer 12 train cross-entropy of the ordinary models by step, rolling mean over 50 steps. Right: layer 48 train cross-entropy of the separate-heads models minus HC with separate heads at the same steps, 100-step bins.
Output: hc_2x2_1104.{pdf,png}."""
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
D = HERE.parents[1]
L = np.load(D / "pls_hcid/live_1104c.npz", allow_pickle=True)
Z = np.load(D / "pls_blog/d48g4.npz", allow_pickle=True)
lay = [int(v) for v in L["lay"]]
PUR, GRN, RED, YEL, RED1 = tone("purple", 600), tone("green", 600), tone("red", 600), tone("yellow", 900), tone("red", 300)
legend = [("HC", GREY), ("Mixing = I", PUR), ("Only shrinks", RED), ("Half dose", RED1), ("Only moves", GRN), ("Keeps the sum, signed", BLUE), ("Mixing = I, |F| bound", YEL)]
fig, axes = canvas(1, 2, width=WIDTH_TEXT, panel_height=2.0, title="The 2 x 2 on the mixing of HC, runs still training", legend=legend)
a, b = np.ravel(axes)
def roll(st, y, w=50):
    return st[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid")
def live(n, l):
    return L[n + "/step"], L[n + "/ce"][:, lay.index(l)]
def done(t, l):
    return Z[t + "/step"], Z[t + "/ce"][:, l - 1]
for s, col, lw in ((done("-sep-bbfrozen-ahc-fa4thd-vs-d48", 12), GREY, 1.2), (done("-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", 12), PUR, 1.2), (live("w972 only shrinks", 12), RED, 1.4), (live("w1002 half dose", 12), RED1, 1.2), (live("w1000 only moves", 12), GRN, 1.2), (live("w1006 keeps the sum", 12), BLUE, 1.4)):
    a.plot(*roll(*s), color=col, lw=lw)
nice_y(a, 3.5, 7, fmt="{:.1f}")
a.set_xlim(0, 4000)
a.set_xticks([0, 1000, 2000, 3000, 4000])
room(a)
panel_label(a, "Layer 12 ordinary model (nats)")
def gap(x, w=100):
    sa, ya = x
    sb, yb = done("-sep-ahc-fa4thd-vs-d48", 48)
    xs, ys = [], []
    for s in range(100, int(sa[-1]) - w + 2, w):
        ma, mb = (sa >= s) & (sa < s + w), (sb >= s) & (sb < s + w)
        if ma.sum() > 0.8 * w:
            xs.append(s + w / 2); ys.append(ya[ma].mean() - yb[mb].mean())
    return xs, ys
b.axhline(0, color=GREY, lw=1.0)
for s, col, lw in ((done("-sep-ahc-hcid-fa4thd-vs-d48", 48), PUR, 1.2), (live("w973 only shrinks", 48), RED, 1.4), (live("w1003 half dose", 48), RED1, 1.2), (live("w1001 only moves", 48), GRN, 1.2), (live("w1007 keeps the sum", 48), BLUE, 1.4), (live("w1091 mixing = I, bound", 48), YEL, 1.2)):
    b.plot(*gap(s), color=col, lw=lw)
nice_y(b, -0.05, 0.25, fmt="{:.2f}")
b.set_xlim(0, 4000)
b.set_xticks([0, 1000, 2000, 3000, 4000])
room(b)
panel_label(b, "Layer 48 separate heads minus HC (nats)")
save(fig, HERE / "hc_2x2_1104")
