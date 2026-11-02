"""Interim check of w810 (HC, mixing = I, separate heads; stopped at step 3654, still to resume) against finished runs.
Left: root sum of squares over all sublayers of hc's learned write offsets (W&B params/norm/...hyper.j.write).
Right: global gradient norm (grad/norm/total), 10-row running median (rows every 10 steps, so 100 steps).
Runs: w810, w811 (same design, ordinary model, stopped at step 4574), HC separate heads and HC ordinary (ahc-fa4thd-vs-d48).
The ordinary models train on one loss, so their gradient norm (about 1.3) is left out of the right panel.
Source: ../../pls_hcid/cache/*.npz (sep_layers_fetch.py). Output: hcid_write_runaway.{pdf,png}."""
import re
from pathlib import Path
import numpy as np
from style import *
HERE = Path(__file__).parent
apply_style()
C = HERE.parents[1] / "pls_hcid" / "cache"
def load(t):
    z = np.load(C / f"{t}.npz"); return z["step"], {str(k): i for i, k in enumerate(z["keys"])}, z["v"]
RUNS = [("sep-ahc-hcid-fa4thd-vs-d48", "HC, mixing = I: separate heads", BLUE, "line"), ("sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", "HC, mixing = I: ordinary", LIGHT, "line"),
        ("sep-ahc-fa4thd-vs-d48", "HC: separate heads", GREY, "line"), ("sep-bbfrozen-ahc-fa4thd-vs-d48", "HC: ordinary", GREY, "dash")]
fig, axes = canvas(1, 2, width=WIDTH_POST, panel_height=2.0, title=None, legend=[(l, c, k) for _, l, c, k in RUNS], xlabel="Training step", title_pt=9.2, tick_pt=TEXT_PT)
a, b = np.ravel(axes)
for t, l, c, kind in RUNS:
    s, k, v = load(t); ids = [i for n, i in k.items() if re.search(r"^params/norm/.*hyper\.\d\.write$", n)]
    y = np.sqrt(np.nansum(v[:, ids] ** 2, 1)); ls = "-" if kind == "line" else (0, (3, 2))
    a.plot(s, y, color=c, lw=1.6, ls=ls)
    if "bbfrozen" not in t:
        g = v[:, k["grad/norm/total"]]; m = np.array([np.median(g[max(0, i - 9):i + 1]) for i in range(len(g))])
        b.plot(s[s >= 500], m[s >= 500], color=c, lw=1.6, ls=ls)
for ax, lo, hi, name in ((a, 0, 18, "Write offsets (norm)"), (b, 0, 24, "Gradient norm")):
    ax.set_xlim(0, 5000); nice_y(ax, lo, hi, zero=True, fmt="{:.0f}"); ax.set_xticks([0, 2500, 5000]); ax.set_xticklabels(["0", "2,500", "5,000"]); room(ax); panel_label(ax, name, x=2500 if ax is b else 0)
save(fig, HERE / "hcid_write_runaway")
