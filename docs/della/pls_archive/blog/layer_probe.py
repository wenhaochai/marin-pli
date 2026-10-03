"""Final training loss by layer against backbone compute, 130m, three ways (log-log):
  separate heads - readouts L1..L5 of the 独立头 pls run (muonh-qwen3-130m-della4xh100-pls1-sep): one head per layer,
                 every layer trained to predict, the backbone gets each head's gradient;
  probes       - readouts L1..L5 of the 只训独立头 run (-pls1-sep-bbfrozen): the same separate heads on a backbone detached from
                 their loss, so the backbone trains exactly as the baseline (its L5 3.257 vs baseline 3.260) and the heads
                 only read how well each layer of the plain model predicts the next token;
  baseline by depth - plain models with 2..5 layers (-d2..-d5, same width, data and schedule) and the 6-layer baseline.
Readout k of the 6-layer model and the plain model with k+1 layers share an x: backbone compute 3*(k+1)/6*F_backbone per
token (embedding and LM head not counted; the d-layer controls' logged FLOPs minus the head equal d/6 of the baseline's).
Final value = mean of the last 50 logged steps. Curves: separate heads and baseline by depth, fitted L = E + A (C/1e18)^-alpha;
probes, which no such law fits (residuals to 0.14), a monotone PCHIP interpolation in log-log, not a fit.
Sources: bbfrozen.npz (fetch_bbfrozen.py), depth.json (fetch_depth.py), baselines.json (fetch_baselines.py).
Output: layer_probe_130m.*"""
import json
from pathlib import Path
import matplotlib.ticker
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from style import *
HERE = Path(__file__).resolve().parent
apply_style()
B = json.load(open(HERE / "baselines.json"))["130m"]
DEPTH = json.load(open(HERE / "depth.json"))
Z = np.load(HERE / "bbfrozen.npz")
Fb = B["backbone_flops"]
x = np.array([(k + 1) / 6 * Fb for k in range(1, 6)])
shared = Z["-pls1-sep/final"][1:]   # separate heads (独立头)
probe = Z["-pls1-sep-bbfrozen/final"][1:]
plain = np.array([DEPTH[str(d)]["loss"] for d in range(2, 6)] + [B["loss"]])
SHARED, PROBE, PLAIN = tone('blue', 700), tone('red', 600), GREY_600
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
law = lambda c, E, A, a: E + A * (c / 1e18) ** (-a)

fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=2.3,
                   title='Per-layer training loss against backbone compute, 130m',
                   legend=[('Separate heads, fit', SHARED), ('Probe of the baseline', PROBE), ('Baseline by depth, fit', PLAIN)],
                   quantity='Cross-entropy (nats, log scale)', xlabel='Backbone compute up to the layer (FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
ax = axes.flat[0]
g = np.geomspace(x.min(), x.max(), 200)
for y, c in ((shared, SHARED), (plain, PLAIN)):
    p, _ = curve_fit(law, x, y, p0=(y.min() - 0.3, 0.3, 0.5), bounds=([0, 0, 0], [y.min(), 10, 5]), maxfev=20000)
    ax.plot(g, law(g, *p), color=c, lw=1.2, zorder=4)
    ax.scatter(x, y, s=22, color=c, zorder=6)
    print("fit", np.round(p, 3), "max |resid| %.4f" % np.abs(law(x, *p) - y).max())
pch = PchipInterpolator(np.log(x), np.log(probe))
ax.plot(g, np.exp(pch(np.log(g))), color=PROBE, lw=1.2, zorder=4)
ax.scatter(x, probe, s=22, color=PROBE, zorder=6)
ax.set_xscale('log')
ax.set_yscale('log')
for a in (ax.xaxis, ax.yaxis):
    a.set_minor_locator(matplotlib.ticker.NullLocator())
allx, ally = np.r_[x, x, x], np.r_[shared, probe, plain]
xmin, xmax = allx.min() / 4, allx.max() * 2
ylo, yhi = 0.97 * ally.min(), ally.max() * 1.15
xt = [v for v in (1e16, 1e17, 1e18, 1e19) if xmin <= v <= xmax]
ax.set_xticks(xt, [f'10{str(int(np.log10(v))).translate(SUP)}' for v in xt])
ax.set_xlim(xmin, xmax)
ax.set_ylim(ylo, yhi)
y_values(ax, [v for v in (3, 3.25, 3.5, 3.75, 4, 4.5, 5) if ylo <= v <= yhi / 1.05], '{:g}')
save(fig, HERE / 'layer_probe_130m')
print("separate", shared.round(3), "\nprobe ", probe.round(3), "\nplain ", plain.round(3))
print("probe - plain", (probe - plain).round(3), "\nseparate - plain", (shared - plain).round(3))
