"""Final state of the ordinary model (probes run, PLS_DETACH_BB=1) after w811 finished. No training state is read live.
Left: train cross-entropy of the probe at each of the 48 layers, mean of the last 50 logged steps (<tag>/final).
Right: each finished 48-layer run on two axes: layer-48 train cross-entropy (x) and eval/L47/loss (y).
Source: ../../pls_blog/d48g4.npz, keys -sep-bbfrozen-<tag>/{final,eval}. W&B reself/marin-della.
Left tags: ahc-fa4thd-vs-d48 (HC), ahc-hcid-fa4thd-vs-d48 (w811), amhc-mhcid-mhcrwlin-fa4thd-vs-d48, apreln-fa4thd-vs-d48,
all on the fa4thd attention stack. Right: every ahc*, amhc* and apreln* run in the file, both attention stacks.
Output: hcid_probes_final.{pdf,png}.
"""
from pathlib import Path
import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
RED = tone("red", 600)
z = np.load(HERE.parents[1] / "pls_blog" / "d48g4.npz", allow_pickle=True)
P = "-sep-bbfrozen-"
ID = "ahc-hcid-fa4thd-vs-d48"
fig, axes = canvas(1, 2, width=WIDTH_POST, panel_height=2.0, title=None,
                   legend=[("HC", GREY), ("HC, mixing = I", BLUE), ("mHC, mixing = I, linear read/write", RED), ("Pre-LN", LIGHT)],
                   quantity="Loss (nats)", title_pt=9.2, tick_pt=TEXT_PT)
ax, bx = np.ravel(axes)
x = np.arange(1, 49)
for tag, c, lw in (("apreln-fa4thd-vs-d48", LIGHT, 1.2), ("ahc-fa4thd-vs-d48", GREY, 1.2),
                   ("amhc-mhcid-mhcrwlin-fa4thd-vs-d48", RED, 1.2), (ID, BLUE, 1.6)):
    ax.plot(x, z[f"{P}{tag}/final"], color=c, lw=lw)
ax.set_xlim(1, 48); ax.set_xticks([1, 12, 24, 36, 48])
nice_y(ax, 2.9, 5.6, fmt="{:.1f}")
room(ax); panel_label(ax, "Train loss by layer")
rows = []
for k in z.files:
    if not (k.startswith(P) and k.endswith("/final")): continue
    tag = k[len(P):-6]
    if z[k].shape[0] != 48: continue
    fam = "id" if tag == ID else "mhc" if tag.startswith("amhc") else "hc" if tag.startswith("ahc") else "pre" if tag.startswith("apreln") else None
    if fam: rows.append((fam, float(z[k][47]), float(z[f"{P}{tag}/eval"][47])))
for fam, c in (("pre", LIGHT), ("hc", GREY), ("mhc", RED), ("id", BLUE)):
    a = np.array([(f, e) for g, f, e in rows if g == fam])
    dots(bx, a[:, 0], a[:, 1], c)
bx.set_xlim(2.90, 3.00); bx.set_xticks([2.90, 2.95, 3.00]); bx.set_xticklabels(["2.90", "2.95", "3.00"])
nice_y(bx, 3.56, 3.69, fmt="{:.2f}")
room(bx); panel_label(bx, "Layer 48: train (x), eval (y)")
save(fig, HERE / "hcid_probes_final")
