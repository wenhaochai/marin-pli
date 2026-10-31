"""Per-layer training loss against backbone compute, 300m shared head, every readout L1-L11 (L0 left out, as in the
130m figure), log-log. Readout k: 3 * (k+1)/12 * F_backbone FLOPs per token (input embedding and LM head not counted),
F_backbone = the baseline's forward FLOPs per token (logged throughput/total_gflops / 3 / tokens) minus 2*hidden*vocab.
One hue, light (shallow) to dark (deep): a ramp between GM2 blue 200 and blue 900, the owner's choice for an ordered set
of eleven; each curve faint, its last point a dot; the final points joined into one curve; legend lists L1, L6, L11. Source: flops.npz (fetch_flops.py, W&B
reself/marin-della). 50-step running means from step 100. Output: layer_backbone_300m_all.*"""
from pathlib import Path
import matplotlib.colors as mc
import matplotlib.ticker
import numpy as np
from style import *
HERE = Path(__file__).resolve().parent
D = dict(np.load(HERE / "flops.npz"))
apply_style()
TOK, V, L, HID = 128 * 4096, 128256, 12, 768
ramp = mc.LinearSegmentedColormap.from_list('blue', [tone('blue', 200), tone('blue', 900)])
LAYERS = list(range(1, 12))
COLS = {k: ramp((k - 1) / (len(LAYERS) - 1)) for k in LAYERS}

b = D["300-base/gflops"][-1] * 1e9 / (D["300-base/step"][-1] * TOK) / 3
F = b - 2 * HID * V


def smooth(x, y, w=50):
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=2.3,
                   title='Per-layer training loss against backbone compute, 300m',
                   legend=[('L1', COLS[1], 'dot'), ('L6', COLS[6], 'dot'), ('L11', COLS[11], 'dot'),
                           ('Final loss by layer', tone('blue', 700)), ('Baseline', GREY_600, 'dot')],
                   quantity='Cross-entropy (nats, log scale)', xlabel='Backbone compute up to the layer (FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
ax = axes.flat[0]
s, ce = D["300-shared/step"], D["300-shared/ce"]
m = s >= 100
ends = []
for k in LAYERS:
    xs, y = smooth((s * TOK * 3 * (k + 1) / L * F)[m], ce[m, k])
    ax.plot(xs, y, color=COLS[k], lw=1.0, alpha=0.3, zorder=3)
    ax.scatter([xs[-1]], [y[-1]], s=24, color=COLS[k], zorder=6)
    ends.append((xs[-1], y[-1], f'L{k}', COLS[k]))
sb, cb = D["300-base/step"], D["300-base/ce"][:, 0]
mb = sb >= 100
xs, y = smooth((sb * TOK * 3 * F)[mb], cb[mb])
ax.plot(xs, y, color=GREY_600, lw=1.0, alpha=0.3, zorder=2)
ax.scatter([xs[-1]], [y[-1]], s=24, color=GREY_600, zorder=5)
ends.append((xs[-1], y[-1], 'Baseline', GREY_700))
ex, ey = [e[0] for e in ends[:-1]], [e[1] for e in ends[:-1]]
ax.plot(ex, ey, color=tone('blue', 700), lw=1.2, zorder=5)   # the final points of L1..L11 as one curve
ax.set_xscale('log')
ax.set_yscale('log')
for a in (ax.xaxis, ax.yaxis):
    a.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xticks([1e17, 1e18, 1e19], ['10¹⁷', '10¹⁸', '10¹⁹'])
ax.set_xlim(4e16, 2.2e19)
ax.set_ylim(2.75, 6.6)
y_values(ax, [3, 4, 5, 6], '{:g}')
panel_label(ax, 'Shared head', x=1.2e18)
save(fig, HERE / 'layer_backbone_300m_all')
for x, y, name, _ in ends:
    print(f"{name:9s} end x={x:.3g} loss={y:.3f}")
