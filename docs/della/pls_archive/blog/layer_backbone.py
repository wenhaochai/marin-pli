"""Per-layer training loss of the shared-head pls run against backbone compute, with the baseline's final points at
four sizes, log-log. One figure per pls size (130m: readouts L1-L5; 300m: final points of L1-L11, curves of L1, L3, ..., L11; L0 left out).

Readout k of an L-layer pls model: 3 * (k+1)/L * F_backbone FLOPs per token, F_backbone = the same-size baseline's
forward FLOPs per token (logged throughput/total_gflops / 3 / tokens) minus the LM head's 2*hidden*vocab; the input
embedding and the head are not counted. Curves: 50-step running means from step 100, faint, one blue ramped light
(shallow) to dark (deep); the dots are their last points (one colour), with a fitted L = E + A (C/1e18)^-alpha over their range. Baseline: final training loss (mean
of the last 50 logged steps) of the same-size muonh-qwen3-{size}-della4xh100 run, one point. 130m only: the
depth-matched baselines muonh-qwen3-130m-della4xh100-d{2..5} (same width, data and schedule, 2..5 layers; backbone
compute d/6 of the 6-layer baseline's, equal to their logged FLOPs minus the head) and the 6-layer baseline, final
training loss (same mean) with their own fit; d1 (matching L0, left out) is not drawn.
Sources: flops.npz (fetch_flops.py), baselines.json (fetch_baselines.py), depth.json (fetch_depth.py), W&B reself/marin-della.
Output: layer_backbone_130m.*, layer_backbone_300m.*"""
import json
from pathlib import Path
import matplotlib.colors as mc
import matplotlib.ticker
import numpy as np
from scipy.optimize import curve_fit
from style import *
HERE = Path(__file__).resolve().parent
D = dict(np.load(HERE / "flops.npz"))
B = json.load(open(HERE / "baselines.json"))
apply_style()
TOK, V = 128 * 4096, 128256
RAMP = mc.LinearSegmentedColormap.from_list('blue', [tone('blue', 200), tone('blue', 900)])
FINAL, BASE = tone('blue', 700), GREY_600
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def smooth(x, y, w=50):
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


def figure(size, L, hidden, curves=None, controls=None):
    F = D[f"{size}-base/gflops"][-1] * 1e9 / (D[f"{size}-base/step"][-1] * TOK) / 3 - 2 * hidden * V
    fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=2.3,
                       title=f'Per-layer training loss against backbone compute, {size}m',
                       legend=[('Final loss by layer, fit', FINAL)] + ([('Baseline by depth, fit', BASE)] if controls else [('Baseline', BASE, 'dot')]),
                       quantity='Cross-entropy (nats, log scale)', xlabel='Backbone compute up to the layer (FLOPs)',
                       title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
    ax = axes.flat[0]
    s, ce = D[f"{size}-shared/step"], D[f"{size}-shared/ce"]
    m = s >= 100
    ends = []
    for k in range(1, L):
        c = RAMP((k - 1) / max(L - 2, 1))
        xs, y = smooth((s * TOK * 3 * (k + 1) / L * F)[m], ce[m, k])
        if curves is None or k in curves:   # trajectories for a subset only, so neighbours stay apart; every final point is kept
            ax.plot(xs, y, color=c, lw=1.0, alpha=0.15, zorder=3)
        ax.scatter([xs[-1]], [y[-1]], s=22, color=FINAL, zorder=6)
        ends.append((xs[-1], y[-1]))
    ex, ey = map(np.array, zip(*ends))
    law = lambda c, E, A, a: E + A * (c / 1e18) ** (-a)   # saturating power law, fitted to the final points
    (E, A, al), _ = curve_fit(law, ex, ey, p0=(ey.min() - 0.3, 0.3, 0.5), bounds=([0, 0, 0], [ey.min(), 10, 5]), maxfev=20000)
    cx = np.geomspace(ex.min(), ex.max(), 200)
    ax.plot(cx, law(cx, E, A, al), color=FINAL, lw=1.2, zorder=5)
    print(f"{size}m fit: L = {E:.3f} + {A:.3f} (C/1e18)^-{al:.3f}; max |resid| {np.abs(law(ex, E, A, al) - ey).max():.4f}")
    own = B[f"{size}m"]   # the same-size baseline's final point only
    ax.scatter([own["backbone_flops"]], [own["loss"]], s=22, color=BASE, zorder=5)
    if controls:   # depth-matched baselines: one plain model per depth, each at the compute of the readout it matches
        cx_, cy_ = map(np.array, zip(*(controls + [(own["backbone_flops"], own["loss"])])))
        ax.scatter(cx_, cy_, s=22, color=BASE, zorder=5)
        (E2, A2, a2), _ = curve_fit(law, cx_, cy_, p0=(cy_.min() - 0.3, 0.3, 0.5), bounds=([0, 0, 0], [cy_.min(), 10, 5]), maxfev=20000)
        g = np.geomspace(cx_.min(), cx_.max(), 200)
        ax.plot(g, law(g, E2, A2, a2), color=BASE, lw=1.2, zorder=4)
        print(f"{size}m controls fit: L = {E2:.3f} + {A2:.3f} (C/1e18)^-{a2:.3f}; resid {np.round(law(cx_, E2, A2, a2) - cy_, 4)}")
        print("readout minus control at matched compute:", np.round(ey - cy_, 3))
        ex, ey = np.r_[ex, cx_], np.r_[ey, cy_]   # the view frames both sets of final points
    ax.set_xscale('log')
    ax.set_yscale('log')
    for a in (ax.xaxis, ax.yaxis):
        a.set_minor_locator(matplotlib.ticker.NullLocator())
    # frame the endpoints: the view serves the final points, the trajectories are context
    xmin, xmax = ex.min() / 4, ex.max() * 2
    ylo = 0.97 * min(ey.min(), own["loss"])
    yhi = ey.max() * 1.15
    xt = [v for v in (1e16, 1e17, 1e18, 1e19) if xmin <= v <= xmax]
    ax.set_xticks(xt, [f'10{str(int(np.log10(v))).translate(SUP)}' for v in xt])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ylo, yhi)
    y_values(ax, [v for v in (2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5) if ylo <= v <= yhi / 1.05], '{:g}')  # no value right under the top edge, where the panel name sits
    panel_label(ax, 'Shared head', x=xt[0] if len(xt) == 1 else xt[-2])
    save(fig, HERE / f'layer_backbone_{size}m')
    print(size, "final by layer", [round(e[1], 3) for e in ends])


DEPTH = json.load(open(HERE / "depth.json"))
figure("130", 6, 512, controls=[(d / 6 * B["130m"]["backbone_flops"], DEPTH[str(d)]["loss"]) for d in range(2, 6)])
figure("300", 12, 768, curves={1, 3, 5, 7, 9, 11})
