"""Final training loss by layer against backbone compute for the three head types, 130m (log-log): shared head
(muonh-qwen3-130m-della4xh100-pls1), read-only shared head (-pls1-dh, the head gets no gradient from the aux losses),
separate heads (-pls1-sep, one head per layer); and the plain models with 2..5 layers (-d2..-d5) plus the 6-layer baseline.
Readout k of the 6-layer model sits at backbone compute 3*(k+1)/6*F_backbone per token (embedding and LM head not counted),
the d-layer plain model at d/6 of the baseline's. Final value = mean of the last 50 logged steps; readouts L1..L5.
Each series has a fitted L = E + A (C/1e18)^-alpha. Sources: bbfrozen.npz (fetch_bbfrozen.py), depth.json (fetch_depth.py),
baselines.json (fetch_baselines.py). Output: heads_compare_130m.*"""
import json
from pathlib import Path
import matplotlib.ticker
import numpy as np
from scipy.optimize import curve_fit
from style import *
HERE = Path(__file__).resolve().parent
apply_style()
B = json.load(open(HERE / "baselines.json"))["130m"]
DEPTH = json.load(open(HERE / "depth.json"))
Z = np.load(HERE / "bbfrozen.npz")
x = np.array([(k + 1) / 6 * B["backbone_flops"] for k in range(1, 6)])
plain = np.array([DEPTH[str(d)]["loss"] for d in range(2, 6)] + [B["loss"]])
series = [('Separate', Z["-pls1-sep/final"][1:], tone('blue', 700)),
          ('Shared', Z["-pls1/final"][1:], tone('green', 600)),
          ('Read-only head', Z["-pls1-dh/final"][1:], tone('purple', 600)),
          ('Baseline', plain, GREY_600)]
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
law = lambda c, E, A, a: E + A * (c / 1e18) ** (-a)
fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=2.3,
                   title='Final training loss by layer and head type, 130m',
                   legend=[(n, c) for n, _, c in series],
                   quantity='Cross-entropy (nats, log scale)', xlabel='Backbone compute up to the layer (FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
ax = axes.flat[0]
g = np.geomspace(x.min(), x.max(), 200)
for n, y, c in series:
    p, _ = curve_fit(law, x, y, p0=(y.min() - 0.3, 0.3, 0.5), bounds=([0, 0, 0], [y.min(), 10, 5]), maxfev=20000)
    ax.plot(g, law(g, *p), color=c, lw=1.2, zorder=4)
    ax.scatter(x, y, s=22, color=c, zorder=6)
    print(f"{n:22s} {np.round(y, 3)} minus plain {np.round(y - plain, 3)} fit max|resid| {np.abs(law(x, *p) - y).max():.4f}")
ax.set_xscale('log'); ax.set_yscale('log')
for a in (ax.xaxis, ax.yaxis):
    a.set_minor_locator(matplotlib.ticker.NullLocator())
ally = np.concatenate([y for _, y, _ in series])
xmin, xmax = x.min() / 4, x.max() * 2
ylo, yhi = 0.97 * ally.min(), ally.max() * 1.15
xt = [v for v in (1e16, 1e17, 1e18, 1e19) if xmin <= v <= xmax]
ax.set_xticks(xt, [f'10{str(int(np.log10(v))).translate(SUP)}' for v in xt])
ax.set_xlim(xmin, xmax); ax.set_ylim(ylo, yhi)
y_values(ax, [v for v in (3.25, 3.5, 3.75, 4) if ylo <= v <= yhi / 1.05], '{:g}')
save(fig, HERE / 'heads_compare_130m')
