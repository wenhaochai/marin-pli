"""HC with mixing = I and separate heads: the high-gain MLP layers, when they grow, and the first-layer gradient. Status figure for the owner; not a page figure.

Data: ../../pls_hcid/mlp_spike_all.json (mlp_spike_all.py: gain = top singular value of gate x diag(ln_2) times that of up x diag(ln_2) times that of down, per layer, from the checkpoint weights; w810, HC, w811, mHC at step 4,958, w893 at step 4,655);
../../pls_hcid/cache/*.npz (W&B, params/norm/transformer.layers.K.ln_2.weight, every 10 steps); ../../pls_hcid/live_1104.npz and ../../pls_blog/d48g4.npz (W&B train/pls/L47); ../../pls_hcid/fw_grad*-cut0.npz (grad_tail2.py: 128 random windows of 4,096 tokens of FineWeb-Edu 10B, the training data, compute bfloat16, all heads; gradient norm of layer 1's down_proj and o_proj per window).
Top left: gain by layer. Top right: norm of ln_2 at the layers that first blow up, by step, against HC at layer 5. Bottom left: layer 48 train CE minus HC, separate heads, 200-step bins, same step axis. Bottom right: the 128 per-window gradient norms at layer 1, largest first.
Output: hc_spike_1104.{pdf,png}."""
import json
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
D = HERE.parents[1]
P = D / "pls_hcid"
PUR, PUR1, GRN = tone("purple", 700), tone("purple", 300), tone("green", 600)
J = json.load(open(P / "mlp_spike_all.json"))
TW810, TW893, THC = "sep-ahc-hcid-fa4thd-vs-d48", "sep-ahc-hcid-fa4thd-vs-d48-s1", "sep-ahc-fa4thd-vs-d48"
legend = [("HC", GREY), ("Mixing = I", PUR), ("Mixing = I, seed 1", PUR1), ("Mixing = I, ordinary", BLUE), ("mHC, mixing = I, linear", GRN)]
fig, axes = canvas(2, 2, width=WIDTH_TEXT, panel_height=1.5, title="HC with mixing = I: MLP gain and first-layer gradient", legend=legend)
(a, b), (c, e) = axes
x = np.arange(1, 49)
for t, col, lw in (("-" + THC, GREY, 1.0), ("-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", BLUE, 1.0), ("-sep-amhc-mhcid-mhcrwlin-fa4thd-vs-d48", GRN, 1.0), ("-" + TW810, PUR, 1.2), ("-" + TW893, PUR1, 1.2)):
    a.plot(x, np.log10(np.maximum(J[t]["gains"], 1)), color=col, lw=lw)
nice_y(a, 0, 4.7, fmt="{:.0f}")
a.set_xlim(1, 48)
a.set_xticks([1, 12, 24, 36, 48])
room(a)
panel_label(a, "log10 MLP gain, by layer")
def ln2(tag, l):
    z = np.load(P / f"cache/{tag}.npz")
    k = {str(n): i for i, n in enumerate(z["keys"])}
    return z["step"], z["v"][:, k[f"params/norm/transformer.layers.{l}.ln_2.weight"]]
for tag, l, col, ls in ((THC, 4, GREY, "-"), (TW810, 4, PUR, "-"), (TW810, 9, PUR, (0, (3, 2))), (TW893, 11, PUR1, (0, (3, 2))), (TW893, 12, PUR1, "-")):
    b.plot(*ln2(tag, l), color=col, lw=1.2, ls=ls)
nice_y(b, 20, 50, fmt="{:.0f}")
b.set_xlim(0, 5000)
b.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
room(b)
panel_label(b, "Norm of ln_2 at the spike layers")
L = np.load(P / "live_1104.npz", allow_pickle=True)
Z = np.load(D / "pls_blog/d48g4.npz", allow_pickle=True)
lay = [int(v) for v in L["lay"]]
def gap(st, y, w=200):
    sb, yb = Z["-" + THC + "/step"], Z["-" + THC + "/ce"][:, 47]
    xs, ys = [], []
    for s in range(200, int(min(st[-1], sb[-1])) - w + 2, w):
        ma, mb = (st >= s) & (st < s + w), (sb >= s) & (sb < s + w)
        if ma.sum() > 0.8 * w and mb.sum() > 0.8 * w:
            xs.append(s + w / 2); ys.append(y[ma].mean() - yb[mb].mean())
    return xs, ys
c.axhline(0, color=GREY, lw=1.0)
c.plot(*gap(Z["-" + TW810 + "/step"], Z["-" + TW810 + "/ce"][:, 47]), color=PUR, lw=1.2)
c.plot(*gap(L["w893 hcid s1 sep/step"], L["w893 hcid s1 sep/ce"][:, lay.index(48)]), color=PUR1, lw=1.2)
nice_y(c, -0.05, 0.25, fmt="{:.2f}")
c.set_xlim(0, 5000)
c.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
room(c)
panel_label(c, "Layer 48 separate heads minus HC (nats)")
for t, col in ((THC, GREY), (TW810, PUR), (TW893, PUR1)):
    g = np.sqrt(np.load(P / f"fw_grad-{t}-cut0.npz")["norm2"][:, 0])
    e.plot(np.arange(1, g.size + 1), np.log10(np.sort(g)[::-1]), color=col, lw=1.2)
nice_y(e, 1, 4, fmt="{:.0f}")
e.set_xlim(1, 128)
e.set_xticks([1, 32, 64, 96, 128])
room(e)
panel_label(e, "log10 layer 1 gradient per window, largest first")
save(fig, HERE / "hc_spike_1104")
