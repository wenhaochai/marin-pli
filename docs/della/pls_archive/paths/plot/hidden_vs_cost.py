"""Separate-heads cost against how much the separate-heads model's last layer depends on the stream differences.

x: the layer-48 cross-entropy rise (nats, Paloma c4_en, 4 x 1024 tokens) when every stream is replaced by the stream
   mean after layer l, averaged over l = 1, 4, ..., 16 (scripts/della/pls_hidden_ablate.py on the separate-heads
   checkpoint; ../ha-sep-<tag>.json). One-stream designs have no stream differences: x = 0.
y: separate-heads cost, layer-48 final training loss of the separate-heads model minus the probes model (mean of the last
   50 logged steps; docs/della/pls.md).
Output: hidden_vs_cost.{pdf,png}.
"""
import json
from pathlib import Path
from style import *

HERE = Path(__file__).parent
apply_style()
RED = tone("red", 600)
PTS = [  # tag, label, cost, family colour
    ("ahc-vs-d48-g4", "HC", 0.048, BLUE), ("ahc-vs-d48-s1", "HC seed 1", 0.034, BLUE),
    ("ahc-hcrwsig-vs-d48", "HC, sigmoid read/write", 0.044, BLUE), ("ahc-hcs8-vs-d48", "HC, 8 streams", 0.049, BLUE),
    ("ahc-hcs2-vs-d48", "HC, 2 streams", 0.074, BLUE),
    ("amhc-vs-d48-g4", "mHC", 0.102, RED), ("amhc-mhcid-vs-d48", "mHC, identity mixing", 0.100, RED),
    ("amhc-mhcrwlin-vs-d48", "mHC, linear read/write", 0.080, RED),
    (None, "HC, 1 stream", 0.125, GREY), (None, "Pre-LN", 0.089, GREY)]


OFF = {"HC": (6, 0, "left"), "HC, 8 streams": (0, 9, "center"), "HC, sigmoid read/write": (-6, 0, "right"),
       "HC seed 1": (6, -2, "left"), "HC, 2 streams": (-6, -2, "right"), "mHC, linear read/write": (6, 3, "left"),
       "mHC": (-6, 0, "right"), "mHC, identity mixing": (6, 4, "left"), "HC, 1 stream": (6, 0, "left"), "Pre-LN": (6, 0, "left")}


def early(tag):
    if tag is None:
        return 0.0
    d = json.loads((HERE.parent / f"ha-sep-{tag}.json").read_text())["dce"]
    v = [x for k, x in d.items() if int(k) <= 16]
    return sum(v) / len(v)


fig, axes = canvas(1, 1, width=WIDTH_POST, panel_height=2.6, title=None,
                   legend=[("Hyper-Connections", BLUE, "dot"), ("mHC", RED, "dot"), ("One stream", GREY, "dot")],
                   quantity="Separate-heads cost (nats)", xlabel="Last-layer loss rise when the stream differences are deleted (nats)",
                   title_pt=9.2, tick_pt=TEXT_PT)
ax = axes[0][0] if hasattr(axes[0], "__len__") else axes[0]
for tag, lab, c, col in PTS:
    x = early(tag)
    dots(ax, [x], [c], col, size=26)
    dx, dy, ha = OFF.get(lab, (5, 3, "left"))
    ax.annotate(lab, (x, c), xytext=(dx, dy), textcoords="offset points", fontsize=TEXT_PT - 0.6, color=col, ha=ha, va="center")
ax.set_xlim(-0.08, 2.0)
nice_y(ax, 0.02, 0.13, headroom=0.05, fmt="{:.2f}")
room(ax)
save(fig, HERE / "hidden_vs_cost")
