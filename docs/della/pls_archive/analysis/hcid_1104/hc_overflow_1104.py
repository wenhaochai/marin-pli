"""w890 (HC, ordinary model, seed 1): the late failure. Status figure for the owner; not a page figure.

Data: ../../pls_blog/d48g4.npz (W&B train/pls/L{k}); ../../pls_hcid/fw_ce_by_target<tag>.npz (ce_by_target.py: 32 random windows of 4,096 tokens of FineWeb-Edu 10B, compute bfloat16, state norm computed without overflow, at the final checkpoint).
Left: train cross-entropy by step at layers 46, 47 and 48 of w890, and layer 48 of HC seed 0, rolling mean over 20 steps. Right: median state norm by layer, log10, for HC ordinary (seed 0, seed 1 on the earlier code, w890), HC with separate heads, and HC joint read/write ordinary (w889). The dashed line is 1.8e19, where a float32 sum of squares overflows.
Output: hc_overflow_1104.{pdf,png}."""
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
D = HERE.parents[1]
Z = np.load(D / "pls_blog/d48g4.npz", allow_pickle=True)
RED, PUR, GRN, YEL = tone("red", 600), tone("purple", 600), tone("green", 600), tone("yellow", 900)
W890, HC0, HCS1 = "-sep-bbfrozen-ahc-fa4thd-vs-d48-s1", "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-ahc-vs-d48-s1"
legend = [("w890, layer 48 and median state", RED), ("w890, layer 47", PUR), ("w890, layer 46", YEL), ("HC seed 0", GREY), ("HC seed 1, earlier code", LIGHT), ("HC, separate heads", GRN), ("Joint read/write", BLUE)]
fig, axes = canvas(1, 2, width=WIDTH_TEXT, panel_height=2.0, title="HC ordinary model, seed 1: last-layer loss and state norm", legend=legend)
a, b = np.ravel(axes)
def roll(st, y, w=20):
    return st[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid")
s, c = Z[W890 + "/step"], Z[W890 + "/ce"]
m = s >= 3900
for l, col in ((46, YEL), (47, PUR), (48, RED)):
    a.plot(*roll(s[m], c[m, l - 1]), color=col, lw=1.2)
s0, c0 = Z[HC0 + "/step"], Z[HC0 + "/ce"]
m0 = s0 >= 3900
a.plot(*roll(s0[m0], c0[m0, 47]), color=GREY, lw=1.2)
nice_y(a, 2.5, 11, fmt="{:.0f}")
a.set_xlim(3900, 5000)
a.set_xticks([4000, 4400, 4800])
room(a)
panel_label(a, "Train cross-entropy (nats)")
x = np.arange(1, 49)
for t, col, lw in ((HC0, GREY, 1.2), (HCS1, LIGHT, 1.2), ("-sep-ahc-fa4thd-vs-d48", GRN, 1.2), ("-sep-bbfrozen-ahc-hcrwj-fa4thd-vs-d48", BLUE, 1.0), (W890, RED, 1.4)):
    nr = np.load(D / f"pls_hcid/fw_ce_by_target{t}.npz")["nr"]
    b.plot(x, np.log10(np.median(nr, axis=(1, 2))), color=col, lw=lw)
b.axhline(np.log10(1.8e19), color=GREY, lw=0.9, ls=(0, (3, 2)))
nice_y(b, 0, 21, fmt="{:.0f}")
b.set_xlim(1, 48)
b.set_xticks([1, 12, 24, 36, 48])
room(b)
panel_label(b, "log10 median state norm, by layer")
save(fig, HERE / "hc_overflow_1104")
