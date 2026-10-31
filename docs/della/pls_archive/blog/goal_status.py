"""Goal status 2026-10-16: how far each 130m attempt moves the separate-heads curve (Figure 3, blue) toward the ordinary
model (grey). Data: final training loss by layer from assets/data/plc-figures.js (fig-heads, fig-fix, fig-mix, fig-late,
the page's own numbers), minus the ordinary model of the same depth (grey dots). Left: layer 1; right: layer 6 (the
model's output). Output: /scratch/gpfs/GROUP/USER/tmp/plc_shots/goal_status.{pdf,png}."""
import json
from pathlib import Path
from style import *
HERE = Path(__file__).resolve().parent
s = open("/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/assets/data/plc-figures.js").read()
F = json.loads(s[s.index("{"):s.rindex("}") + 1])
def series(fid):
    f = F[fid]; leg = {c: n["en"] for n, c, *_ in f["legend"]}
    return {leg.get(m["color"]): [y for _, y in m["pts"]] for m in f["panels"][0]["marks"] if m["type"] == "dots" and len(m["pts"]) == 6}
A = {}
for fid in ("fig-heads", "fig-fix", "fig-mix", "fig-late"):
    A.update(series(fid))
base = A.pop("Baseline by depth")
ROWS = [("Separate heads", "Separate heads (blue)"), ("Layer losses summing to 1", "Weight 0.2"), ("Layer-loss weight 0.1", "Weight 0.1"),
        ("Layer-loss weight 0.05", "Weight 0.05"), ("Weights rising with depth", "Rising with depth"), ("Weight 1, gradient surgery", "Gradient surgery"),
        ("Separate heads from step 1,500", "From step 1,500"), ("Probes only", "Probes only")]
apply_style()
fig, axes = canvas(1, 2, width=WIDTH_POST, panel_height=2.3, title="Gap to the ordinary model of the same depth",
                   quantity="Loss gap (nats)", ticks="left")
names = [b for _, b in ROWS]
for ax, k, lab in ((axes[0][0] if hasattr(axes[0], "__len__") else axes[0], 0, "Layer 1"), (axes[0][1] if hasattr(axes[0], "__len__") else axes[1], 5, "Layer 6 (output)")):
    vals = [A[a][k] - base[k] for a, _ in ROWS]
    y = list(range(len(ROWS)))[::-1]
    ax.barh(y, vals, color=[MEDIUM[1] if a == "Probes only" else MEDIUM[0] for a, _ in ROWS], height=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + 0.01 * (1 if v >= 0 else -1), yy, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=6.5, color=TICK)
    ax.set_yticks(y); ax.set_yticklabels(names if k == 0 else [""] * len(names))
    ax.axvline(0, color=AXIS, lw=0.8)
    lo = min(0, min(vals)); hi = max(vals)
    ax.set_xlim(lo - 0.05 * (hi - lo) - (0.08 if lo < 0 else 0), hi * 1.35)
    panel_label(ax, lab)
save(fig, Path("/scratch/gpfs/GROUP/USER/tmp/plc_shots/goal_status"))
for a, b in ROWS:
    print(f"{b:24s} L1 {A[a][0] - base[0]:+.3f}  L6 {A[a][5] - base[5]:+.3f}")
