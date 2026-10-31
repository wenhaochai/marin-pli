"""Per-layer Paloma macro eval loss minus Marin's anchored asymptote E = 1.5 against backbone compute, log-log, for the
shared-head pls runs (130m: L1-L5, 300m: L1-L11; L0 left out) and the same-size baseline's final point.

E = 1.5 is Marin's anchored asymptote for its Paloma macro loss fit (L = 1.5 + 84.8 C^-0.09; nats, Llama-3 tokenizer,
C without the LM-head term), from marin_ladder.md in Overleaf project 6a80071edaa6410e97a608b4. Readout k of an L-layer
model: backbone compute 3 * (k+1)/L * F * tokens, F = the baseline's forward FLOPs per token minus 2*hidden*vocab (as in
layer_backbone.py). Eval every 1000 steps: curves through those points, faint; dots are the final evals (no fit: with E fixed at 1.5 the
points bend off a power law, the deep layers flatten). Sources: eval_macro.json (fetch_eval_macro.py),
flops.npz (fetch_flops.py), W&B reself/marin-della. Output: layer_paloma_130m.*, layer_paloma_300m.*"""
import json
from pathlib import Path
import matplotlib.colors as mc
import matplotlib.ticker
import numpy as np
from style import *
HERE = Path(__file__).resolve().parent
D = dict(np.load(HERE / "flops.npz"))
M = json.load(open(HERE / "eval_macro.json"))
apply_style()
TOK, V, E = 128 * 4096, 128256, 1.5
RAMP = mc.LinearSegmentedColormap.from_list('blue', [tone('blue', 200), tone('blue', 900)])
FINAL, BASE = tone('blue', 700), GREY_600
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def figure(size, L, hidden, curves=None):
    F = D[f"{size}-base/gflops"][-1] * 1e9 / (D[f"{size}-base/step"][-1] * TOK) / 3 - 2 * hidden * V
    fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=2.3,
                       title=f'Per-layer Paloma loss against backbone compute, {size}m',
                       legend=[('Final loss by layer', FINAL, 'dot'), ('Baseline', BASE, 'dot')],
                       quantity='Paloma macro loss minus 1.5 (nats, log scale)', xlabel='Backbone compute up to the layer (FLOPs)',
                       title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
    ax = axes.flat[0]
    st = np.array(M[f"{size}m-shared"]["step"], float)
    mac = np.array(M[f"{size}m-shared"]["macro"])
    ends = []
    for k in range(1, L):
        c = RAMP((k - 1) / max(L - 2, 1))
        x = st * TOK * 3 * (k + 1) / L * F
        y = mac[:, k] - E
        if curves is None or k in curves:
            ax.plot(x, y, color=c, lw=1.0, alpha=0.15, zorder=3)
        ax.scatter([x[-1]], [y[-1]], s=22, color=FINAL, zorder=6)
        ends.append((x[-1], y[-1]))
    ex, ey = map(np.array, zip(*ends))
    bstep = M[f"{size}m-base"]["step"][-1]
    by = M[f"{size}m-base"]["macro"][-1] - E
    bx = bstep * TOK * 3 * F
    ax.scatter([bx], [by], s=22, color=BASE, zorder=5)
    xmin, xmax = ex.min() / 4, ex.max() * 2
    ylo, yhi = 0.97 * min(ey.min(), by), ey.max() * 1.12
    ax.set_xscale('log')
    ax.set_yscale('log')
    for a in (ax.xaxis, ax.yaxis):
        a.set_minor_locator(matplotlib.ticker.NullLocator())
    xt = [v for v in (1e16, 1e17, 1e18, 1e19) if xmin <= v <= xmax]
    ax.set_xticks(xt, [f'10{str(int(np.log10(v))).translate(SUP)}' for v in xt])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ylo, yhi)
    y_values(ax, [v for v in (2.25, 2.5, 2.75, 3, 3.25, 3.5) if ylo <= v <= yhi / 1.05], '{:g}')
    panel_label(ax, 'Shared head', x=xt[0] if len(xt) == 1 else xt[-2])
    save(fig, HERE / f'layer_paloma_{size}m')


figure("130", 6, 512)
figure("300", 12, 768, curves={1, 3, 5, 7, 9, 11})
