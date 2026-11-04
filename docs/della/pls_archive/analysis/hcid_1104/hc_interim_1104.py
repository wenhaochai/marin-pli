"""Interim read of the unfinished HC runs, 2026-11-04. Status figure for the owner; not a page figure.

Data: ../../pls_hcid/live_1104.npz (W&B, live_1104.py: train/pls/L{k} of the unfinished runs), ../../pls_blog/d48g4.npz (finished references), ../../pls_hcid/blown_state_1104.json (blown_fig_data2.py:
4 x 1024 Paloma c4_en validation tokens at the newest checkpoint; w893 at step 4,550, w810 and HC at step 4,958).
Top left: layer 12 train cross-entropy by step, ordinary models (no intermediate loss reaches the backbone), rolling mean over 100 steps.
Top right: layer 48 train cross-entropy of HC with mixing = I and separate heads minus HC with separate heads at the same steps, 200-step bins, seeds 0 (w810) and 1 (w893).
Bottom left: layer 48 train cross-entropy, separate heads minus ordinary model of the same design and seed, at the same steps, 200-step bins. HC seed 1 has no separate-heads partner in this set, so it is absent.
Bottom right: share of tokens with state norm above 100 times that layer's median, by layer.
Output: hc_interim_1104.{pdf,png}."""
import json
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
D = HERE.parents[1]
L = np.load(D / "pls_hcid/live_1104.npz", allow_pickle=True)
Z = np.load(D / "pls_blog/d48g4.npz", allow_pickle=True)
lay = [int(x) for x in L["lay"]]
GREY1, PUR, PUR1, GRN, YEL = tone("grey", 400), tone("purple", 700), tone("purple", 300), tone("green", 600), tone("yellow", 900)


def live(name, l):
    return L[name + "/step"], L[name + "/ce"][:, lay.index(l)]


def done(tag, l):
    return Z[tag + "/step"], Z[tag + "/ce"][:, l - 1]


def roll(st, y, w=100):
    c = np.convolve(y, np.ones(w) / w, mode="valid")
    return st[w - 1:], c


def bins(a, b, w=200, lo=200):
    """mean of a minus mean of b per w-step bin over the steps both have."""
    sa, ya = a
    sb, yb = b
    hi = int(min(sa[-1], sb[-1]))
    x, y = [], []
    for s in range(lo, hi - w + 2, w):
        ma = (sa >= s) & (sa < s + w)
        mb = (sb >= s) & (sb < s + w)
        if ma.sum() > 0.8 * w and mb.sum() > 0.8 * w:
            x.append(s + w / 2)
            y.append(ya[ma].mean() - yb[mb].mean())
    return np.array(x), np.array(y)


HCO, IDO, HCS, IDS, RWS = "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", "-sep-ahc-fa4thd-vs-d48", "-sep-ahc-hcid-fa4thd-vs-d48", "-sep-ahc-hcrwj-fa4thd-vs-d48"
legend = [("HC", GREY), ("HC, seed 1", GREY1), ("Mixing = I", PUR), ("Mixing = I, seed 1", PUR1), ("Joint read/write", BLUE), ("Only moves", GRN), ("Learned scalar", YEL)]
fig, axes = canvas(2, 2, width=WIDTH_TEXT, panel_height=1.5, title="Unfinished HC runs on 2026-11-04", legend=legend)
(a, b), (c, e) = axes
for s, col, lw in ((done(HCO, 12), GREY, 1.2), (live("w890 HC s1", 12), GREY1, 1.0), (done(IDO, 12), PUR, 1.2), (live("w891 hcid s1", 12), PUR1, 1.0),
                   (live("w889 hcrwj", 12), BLUE, 1.0), (live("w1000 move", 12), GRN, 1.2), (live("w960 hcscalar", 12), YEL, 1.0)):
    a.plot(*roll(*s), color=col, lw=lw)
nice_y(a, 3.5, 6, fmt="{:.1f}")
a.set_xlim(0, 5000)
a.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
room(a)
panel_label(a, "Layer 12 ordinary model (nats)")
b.axhline(0, color=GREY, lw=1.0)
b.plot(*bins(done(IDS, 48), done(HCS, 48)), color=PUR, lw=1.2)
b.plot(*bins(live("w893 hcid s1 sep", 48), done(HCS, 48)), color=PUR1, lw=1.2)
nice_y(b, -0.05, 0.25, fmt="{:.2f}")
b.set_xlim(0, 5000)
b.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
room(b)
panel_label(b, "Layer 48 separate heads minus HC (nats)")
c.plot(*bins(done(HCS, 48), done(HCO, 48)), color=GREY, lw=1.2)
c.plot(*bins(done(IDS, 48), done(IDO, 48)), color=PUR, lw=1.2)
c.plot(*bins(live("w893 hcid s1 sep", 48), live("w891 hcid s1", 48)), color=PUR1, lw=1.2)
c.plot(*bins(done(RWS, 48), live("w889 hcrwj", 48)), color=BLUE, lw=1.2)
nice_y(c, -0.05, 0.3, fmt="{:.2f}")
c.set_xlim(0, 5000)
c.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
room(c)
panel_label(c, "Layer 48 separate heads minus ordinary (nats)")
J = json.load(open(D / "pls_hcid/blown_state_1104.json"))
x = np.arange(1, 49)
e.plot(x, 100 * np.array(J["-sep-ahc-fa4thd-vs-d48"]["share"]), color=GREY, lw=1.2)
e.plot(x, 100 * np.array(J["-sep-ahc-hcid-fa4thd-vs-d48"]["share"]), color=PUR, lw=1.2)
e.plot(x, 100 * np.array(J["-sep-ahc-hcid-fa4thd-vs-d48-s1"]["share"]), color=PUR1, lw=1.2)
nice_y(e, 0, 12, zero=True, fmt="{:.0f}")
e.set_xlim(1, 48)
e.set_xticks([1, 12, 24, 36, 48])
room(e)
panel_label(e, "Tokens above 100 x median state norm (%), by layer")
save(fig, HERE / "hc_interim_1104")
