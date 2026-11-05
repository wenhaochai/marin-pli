"""What the probe at layer 12 reads, in ordinary models. Status figure for the owner; not a page figure. Newest checkpoints; w972, w1006 still training.

Data: ../../pls_hcid/kernel<tag>.npz (kernel.py, 2 windows of 4,096 FineWeb-Edu tokens: the sum is linear in the embedding and the branch outputs, S_12 = T_e e + Sigma_i T_i F_i; median over tokens of |T_i F_i| / |S_12|); ../../pls_hcid/state_share<tag>.npz (state_share.py: type share = share of the variance of the RMS-normalised sum explained by the current token's type, 16,384 tokens); ../../pls_blog/d48g4.npz final train CE, ../../pls_hcid/live_1104d.npz newest 50 steps for running runs.
Left: share of the layer 12 sum by source; the x axis is the embedding (0), then the 24 sublayers of layers 1 to 12 in order. Right: layer 12 train CE against the type share at layer 12, one dot per design.
Output: hc_sources_1104.{pdf,png}."""
import glob
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
D = HERE.parents[1]
H = D / "pls_hcid"
Z = np.load(D / "pls_blog/d48g4.npz", allow_pickle=True)
L = np.load(H / "live_1104d.npz", allow_pickle=True); lay = [int(v) for v in L["lay"]]
PUR, GRN, RED, YEL = tone("purple", 600), tone("green", 600), tone("red", 600), tone("yellow", 900)
S = [("HC", "-sep-bbfrozen-ahc-fa4thd-vs-d48", GREY), ("HC, 1 stream", "-sep-bbfrozen-ahc-hcs1-vs-d48", BLUE), ("Keeps the sum", "-sep-bbfrozen-ahc-hcsum-hcf32-fa4thd-vs-d48", LIGHT), ("Mixing = I", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", PUR), ("Only moves", "-sep-bbfrozen-ahc-hcmove-fa4thd-vs-d48", GRN), ("Only shrinks", "-sep-bbfrozen-ahc-hcshrink-fa4thd-vs-d48", RED)]
fig, axes = canvas(1, 2, width=WIDTH_TEXT, panel_height=2.2, title="Layer 12 of ordinary models: sources of the sum, and its token-type share", legend=[(n, c) for n, _, c in S])
a, b = np.ravel(axes)
for n, t, c in S:
    s = np.load(H / f"kernel{t}.npz")["12/share"]
    a.plot(np.arange(len(s)), s[0:1].tolist() + s[1:].tolist(), color=c, lw=1.3)
nice_y(a, 0, 0.95, zero=True, headroom=0.1, fmt="{:.2f}")
a.set_xlim(0, 24)
a.set_xticks([0, 2, 6, 12, 18, 24])
room(a)
panel_label(a, "Share of the sum by source")
LIVE = {"-sep-bbfrozen-ahc-hcshrink-fa4thd-vs-d48": "w972 only shrinks", "-sep-bbfrozen-ahc-hcshrink-sp0.5-fa4thd-vs-d48": "w1002 half dose", "-sep-bbfrozen-ahc-hcsum-hcf32-fa4thd-vs-d48": "w1006 keeps the sum"}
col = {t: c for _, t, c in S}
for f in sorted(glob.glob(str(H / "state_share-sep-bbfrozen*.npz"))):
    t = Path(f).name[len("state_share"):-4]
    if t + "/final" in Z.files:
        y = float(Z[t + "/final"][11])
    elif t in LIVE:
        y = float(L[LIVE[t] + "/ce"][-50:, lay.index(12)].mean())
    else:
        continue
    x = float(np.load(f)["type_share"][11])
    dots(b, [x], [y], col.get(t, tone("grey", 400)))
nice_y(b, 3.6, 6.3, headroom=0.12, fmt="{:.1f}")
b.set_xlim(0, 1)
b.set_xticks([0, 0.25, 0.5, 0.75, 1])
room(b)
panel_label(b, "Layer 12 CE (nats) by type share")
save(fig, HERE / "hc_sources_1104")
