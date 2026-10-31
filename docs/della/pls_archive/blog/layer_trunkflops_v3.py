"""Per-layer training loss against backbone compute, log-log, L0 left out.

Readout k of an L-layer model: 3 * (k+1)/L * F_trunk FLOPs per token (input embedding and LM head not counted),
F_trunk = the baseline's forward FLOPs per token (its logged throughput/total_gflops / 3 / tokens) minus 2*hidden*vocab.
The last readout and the baseline share the same x. Source: flops.npz (fetch_flops.py, W&B reself/marin-della).
50-step running means from step 100, drawn faint; the dot is each curve's last point (end of training). Output: layer_trunkflops_v3_130m.*, layer_trunkflops_v3_300m.*"""
from pathlib import Path
import matplotlib.ticker
import numpy as np
from style import *
HERE = Path(__file__).resolve().parent
D = dict(np.load(HERE / "flops.npz"))
apply_style()
COLS5 = [tone('blue', 600), tone('red', 600), tone('yellow', 900), tone('green', 600), GREY_900]
TOK, V = 128 * 4096, 128256
CFG = {"130": dict(L=6, hidden=512), "300": dict(L=12, hidden=768)}


def trunk(size):
    b = D[f"{size}-base/gflops"][-1] * 1e9 / (D[f"{size}-base/step"][-1] * TOK) / 3
    return b - 2 * CFG[size]["hidden"] * V


def smooth(x, y, w=50):
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


def panel(ax, size, run, layers):
    t, L = trunk(size), CFG[size]["L"]
    s, ce = D[f"{size}-{run}/step"], D[f"{size}-{run}/ce"]
    m = s >= 100
    ends = []
    for k, c in zip(layers, COLS5):
        xs, y = smooth((s * TOK * 3 * (k + 1) / L * t)[m], ce[m, k])
        ax.plot(xs, y, color=c, lw=1.1, alpha=0.3, zorder=3)
        ax.scatter([xs[-1]], [y[-1]], s=26, color=c, zorder=6)
        ends.append((xs[-1], y[-1]))
    ax.plot([e[0] for e in ends], [e[1] for e in ends], color=GREY_800, lw=1.2, zorder=5)
    sb, cb = D[f"{size}-base/step"], D[f"{size}-base/ce"][:, 0]
    mb = sb >= 100
    xs, y = smooth((sb * TOK * 3 * t)[mb], cb[mb])
    ax.plot(xs, y, color=GREY_600, lw=1.1, alpha=0.3, zorder=2)
    ax.scatter([xs[-1]], [y[-1]], s=26, color=GREY_600, zorder=5)


def axes_loglog(ax, xticks, xlim, yticks, ylim):
    ax.set_xscale('log')
    ax.set_yscale('log')
    for a in (ax.xaxis, ax.yaxis):
        a.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(xticks, [f'10{str(int(np.log10(v))).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))}' for v in xticks])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    y_values(ax, yticks, '{:g}')


L130 = [1, 2, 3, 4, 5]
fig, axes = canvas(rows=3, cols=1, width=WIDTH_POST, panel_height=1.15, gap_rows=0.42,
                   title='Per-layer training loss against backbone compute, 130m',
                   legend=[(f'L{k}', c, 'dot') for k, c in zip(L130, COLS5)] + [('Final loss by layer', GREY_800), ('Baseline', GREY_600, 'dot')],
                   quantity='Cross-entropy (nats, log scale)', xlabel='Backbone compute up to the layer (FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
for ax, (run, label) in zip(axes.flat, [("shared", "Shared head"), ("ro", "Read-only shared head"), ("own", "Own heads")]):
    panel(ax, "130", run, L130)
    axes_loglog(ax, [1e16, 1e17, 1e18], (5e15, 1.2e18), [3, 3.5, 4, 4.5, 5], (3.0, 5.2))
    panel_label(ax, label, x=1.6e17)
save(fig, HERE / 'layer_trunkflops_v3_130m')

L300 = [2, 4, 6, 8, 11]
fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=1.9,
                   title='Per-layer training loss against backbone compute, 300m',
                   legend=[(f'L{k}', c, 'dot') for k, c in zip(L300, COLS5)] + [('Final loss by layer', GREY_800), ('Baseline', GREY_600, 'dot')],
                   quantity='Cross-entropy (nats, log scale)', xlabel='Backbone compute up to the layer (FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
ax = axes.flat[0]
panel(ax, "300", "shared", L300)
axes_loglog(ax, [1e17, 1e18, 1e19], (4e16, 1e19), [3, 4, 5, 6], (2.8, 6.6))
panel_label(ax, 'Shared head', x=1.2e18)
save(fig, HERE / 'layer_trunkflops_v3_300m')
