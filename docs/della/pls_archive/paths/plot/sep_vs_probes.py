"""The two terms of the separate-heads cost, design by design, 48 layers.

Each row is one design. Light dot: layer-48 final training loss of the probes model (the ordinary model: its backbone is
trained on the last layer's loss only). Dark dot: the same for the separate-heads model. The segment between them is the
separate-heads cost. Final = mean of the last 50 logged steps (tmp/pls_blog/d48g4.npz, <tag>/final[47]; the
separate-heads run of mHC mixing = I + linear read/write lost its step-4958 train row, so its mean is over steps
4908-4957). fa4thd = the run on the newer attention / cross-entropy stack (wq w387/w388, w334, w711-w714); the cost is a
difference of two models on the same stack. Rows whose separate-heads run is not fetched yet show only the light dot.
Output: sep_vs_probes.{pdf,png}. Scratch figure for the owner, 2026-11-01 (tmp/pls_audit2/decompose.py prints the table).
"""
from pathlib import Path
import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
d = dict(np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True))
GROUPS = [
    [("apreln-vs-d48-g4", "Pre-LN"), ("apreln-vs-d48-s1", "Pre-LN, seed 1"), ("apreln-fa4thd-vs-d48", "Pre-LN (fa4thd)"),
     ("ahc-hcs1-vs-d48", "HC, 1 stream")],
    [("ahc-vs-d48-g4", "HC"), ("ahc-vs-d48-s1", "HC, seed 1"), ("ahc-fa4thd-vs-d48", "HC (fa4thd)"),
     ("ahc-hcrwsig-vs-d48", "HC, sigmoid read/write"), ("ahc-hcs8-vs-d48", "HC, 8 streams"),
     ("ahc-hcs2-vs-d48", "HC, 2 streams"), ("ahc-rstatic-vs-d48", "HC, static routing"),
     ("ahc-hcwsh-fa4thd-vs-d48", "HC, one shared write (fa4thd)")],
    [("amhc-vs-d48-g4", "mHC"), ("amhc-mhcid-vs-d48", "mHC, mixing = I"), ("amhc-mhcrwlin-vs-d48", "mHC, linear read/write"),
     ("amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "mHC, mixing = I + linear (fa4thd)")]]
GAP = 0.6   # extra space between groups, in rows


def fin(arm, tag):
    k = f"-sep{arm}-{tag}/final"
    return float(d[k][47]) if k in d else None


rows, y = [], 0.0
for g in GROUPS:
    for tag, lab in g:
        rows.append((y, lab, fin("-bbfrozen", tag), fin("", tag)))
        y += 1
    y += GAP
n = y - GAP
fig, axes = canvas(1, 1, width=WIDTH_POST, panel_height=0.17 * n, title=None,
                   legend=[("Ordinary model (probes)", LIGHT, "dot"), ("Separate heads", BLUE, "dot")],
                   xlabel="Last-layer final training loss (nats)", title_pt=9.2, tick_pt=TEXT_PT, extra=(-0.22, 0, 0, 0))
ax = axes[0][0] if hasattr(axes[0], "__len__") else axes[0]
for yy, lab, p, s in rows:
    if p is not None and s is not None:
        ax.plot([p, s], [yy, yy], color=tone("grey", 400), lw=1.2, zorder=2,
                solid_capstyle="butt")
    if p is not None:
        dots(ax, [p], [yy], LIGHT, size=22, alpha=0.9)
    if s is not None:
        dots(ax, [s], [yy], BLUE, size=22, alpha=0.9)
    ax.text(-0.03, yy, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=TEXT_PT - 0.4,
            color=TICK)
    print(f"{lab:36s} probes {p if p is None else round(p, 3)}  sep {s if s is None else round(s, 3)}"
          f"  cost {round(s - p, 3) if p is not None and s is not None else '-'}")
ax.set_ylim(n - 0.4, -0.6)
ax.set_yticks([])
ax.grid(False, axis="y")
ax.set_xlim(2.895, 3.115)
ax.set_xticks([2.90, 2.95, 3.00, 3.05, 3.10])
ax.xaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["x"]).FormatStrFormatter("%.2f"))
save(fig, HERE / "sep_vs_probes")
