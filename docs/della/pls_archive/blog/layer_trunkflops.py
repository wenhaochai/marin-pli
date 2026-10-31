"""Per-layer training loss against the trunk compute a model truncated at that layer would use; the input embedding
and the LM head are not counted. Readout k of an L-layer model: 3 * (k+1)/L * F_trunk FLOPs per token, F_trunk = the
baseline's forward FLOPs per token (from its logged throughput/total_gflops / 3) minus the head's 2 * hidden * vocab.
The last readout and the baseline share the same x. Source: flops.npz (fetch_flops.py, W&B reself/marin-della).
50-step running means from step 200. Output: layer_trunkflops_130m.*, layer_trunkflops_300m.*"""
from pathlib import Path
import numpy as np
from style import *
HERE = Path(__file__).resolve().parent
D = dict(np.load(HERE / "flops.npz"))
apply_style()
COLS6 = [tone('blue', g) for g in (300, 400, 500, 600, 700, 900)]
TOK = 128 * 4096
V = 128256
CFG = {"130": dict(L=6, hidden=512), "300": dict(L=12, hidden=768)}


def per_token(size):
    b = D[f"{size}-base/gflops"][-1] * 1e9 / (D[f"{size}-base/step"][-1] * TOK) / 3   # forward FLOPs / token
    head = 2 * CFG[size]["hidden"] * V
    print(f"{size}m: forward {b:.3g} FLOPs/token, head share {head / b:.2f}")
    return b - head, head


def smooth(x, y, w=50):
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


def panel(ax, size, run, layers, cols):
    trunk, head = per_token(size)
    L = CFG[size]["L"]
    s, ce = D[f"{size}-{run}/step"], D[f"{size}-{run}/ce"]
    m = s >= 200
    hi = 0
    for k, c in zip(layers, cols):
        x = s * TOK * 3 * ((k + 1) / L * trunk) / 1e18
        xs, y = smooth(x[m], ce[m, k])
        ax.plot(xs, y, color=c, lw=1.1, zorder=3)
        hi = max(hi, y.max())
    sb, cb = D[f"{size}-base/step"], D[f"{size}-base/ce"][:, 0]
    mb = sb >= 200
    xs, y = smooth(sb[mb] * TOK * 3 * trunk / 1e18, cb[mb])
    ax.plot(xs, y, color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=4)
    return y.min(), hi


fig, axes = canvas(rows=3, cols=1, width=WIDTH_POST, panel_height=1.15, gap_rows=0.42,
                   title='Per-layer training loss against trunk compute, 130m',
                   legend=[(f'L{k}', c) for k, c in enumerate(COLS6)] + [('Baseline', GREY, 'dash')],
                   quantity='Cross-entropy (nats)', xlabel='Trunk compute up to the layer (10¹⁸ FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
for ax, (run, label) in zip(axes.flat, [("shared", "Shared head"), ("ro", "Read-only shared head"), ("own", "Own heads")]):
    lo, hi = panel(ax, "130", run, range(6), COLS6)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ['0', '0.2', '0.4', '0.6', '0.8'])
    ax.set_xlim(0, 0.8)
    nice_y(ax, lo, min(hi, 5.0), fmt='{:.1f}')
    room(ax)
    panel_label(ax, label)
save(fig, HERE / 'layer_trunkflops_130m')

L300 = [0, 2, 4, 6, 8, 11]
fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=1.9,
                   title='Per-layer training loss against trunk compute, 300m',
                   legend=[(f'L{k}', c) for k, c in zip(L300, COLS6)] + [('Baseline', GREY, 'dash')],
                   quantity='Cross-entropy (nats)', xlabel='Trunk compute up to the layer (10¹⁸ FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
ax = axes.flat[0]
lo, hi = panel(ax, "300", "shared", L300, COLS6)
ax.set_xticks([0, 2, 4, 6], ['0', '2', '4', '6'])
ax.set_xlim(0, 7)
nice_y(ax, lo, min(hi, 5.0), fmt='{:.1f}')
room(ax)
panel_label(ax, 'Shared head')
save(fig, HERE / 'layer_trunkflops_300m')

# each readout at its own final compute vs the baseline at the same compute
for size, run, layers in [("130", "shared", range(6)), ("130", "own", range(6)), ("300", "shared", L300)]:
    trunk, head = per_token(size)
    L = CFG[size]["L"]
    sb, cb = D[f"{size}-base/step"], D[f"{size}-base/ce"][:, 0]
    xb = sb * TOK * 3 * trunk
    s, ce = D[f"{size}-{run}/step"], D[f"{size}-{run}/ce"]
    row = []
    for k in layers:
        xk = s[-1] * TOK * 3 * ((k + 1) / L * trunk)
        i = min(np.searchsorted(xb, xk), len(cb) - 26)
        row.append(f"L{k} {ce[-50:, k].mean():.3f} vs base {cb[i-25:i+25].mean():.3f} @ {xk/1e18:.2f}")
    print(size, run, " | ".join(row))
