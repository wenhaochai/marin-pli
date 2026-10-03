"""Per-layer training loss against backbone compute, 300m, at the latest step of the running 只训独立头 run (in progress).
Shared head = readouts L1..L11 of muonh-qwen3-300m-della4xh100-pls1 (flops.npz); probe = readouts L1..L11 of
-pls1-sep-bbfrozen (W&B, live); baseline = train/loss of muonh-qwen3-300m-della4xh100. All three are the mean over the
same 100-step window ending at the probe run's latest logged step. Backbone compute of readout k: 3*(k+1)/12*F_backbone
per token times tokens so far (embedding and LM head not counted). Shared head: fitted L = E + A (C/1e18)^-alpha;
probe: monotone PCHIP interpolation in log-log, not a fit. Output: layer_probe_300m_progress.*"""
import json
from pathlib import Path
import matplotlib.ticker
import numpy as np
import wandb
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from style import *
HERE = Path(__file__).resolve().parent
TOK, V, L = 128 * 4096, 128256, 12
D = dict(np.load(HERE / "flops.npz"))
F = D["300-base/gflops"][-1] * 1e9 / (D["300-base/step"][-1] * TOK) / 3 - 2 * 768 * V
api = wandb.Api(timeout=300)
keys = [f"train/pls/L{k}" for k in range(L)]
r = [x for x in api.run("reself/marin-della/muonh-qwen3-300m-della4xh100-pls1-sep-bbfrozen").scan_history(keys=["_step", *keys], page_size=5000)]
ps = np.array([x["_step"] for x in r]); pce = np.array([[x[k] for k in keys] for x in r])
now = int(ps.max()); lo = now - 99
win = lambda s: (s >= lo) & (s <= now)
probe = pce[win(ps)].mean(0)[1:]
shared = D["300-shared/ce"][win(D["300-shared/step"])].mean(0)[1:]
base = D["300-base/ce"][win(D["300-base/step"])].mean()
x = np.array([3 * (k + 1) / L * F * (now + 1) * TOK for k in range(1, L)])
print("step", now, "\nprobe ", probe.round(3), "\nshared", shared.round(3), "\nbaseline %.3f" % base)
apply_style()
SHARED, PROBE, BASE = tone('blue', 700), tone('red', 600), GREY_600
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=2.3,
                   title=f'Per-layer training loss at step {now} of 11,444, 300m (draft)',
                   legend=[('Shared head, fit', SHARED), ('Probe of the baseline', PROBE), ('Baseline', BASE, 'dot')],
                   quantity='Cross-entropy (nats, log scale)', xlabel='Backbone compute up to the layer (FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
ax = axes.flat[0]
g = np.geomspace(x.min(), x.max(), 200)
law = lambda c, E, A, a: E + A * (c / 1e18) ** (-a)
p, _ = curve_fit(law, x, shared, p0=(shared.min() - 0.3, 0.3, 0.5), bounds=([0, 0, 0], [shared.min(), 10, 5]), maxfev=20000)
print("shared fit", np.round(p, 3), "max |resid| %.4f" % np.abs(law(x, *p) - shared).max())
ax.plot(g, law(g, *p), color=SHARED, lw=1.2, zorder=4)
ax.scatter(x, shared, s=22, color=SHARED, zorder=6)
pch = PchipInterpolator(np.log(x), np.log(probe))
ax.plot(g, np.exp(pch(np.log(g))), color=PROBE, lw=1.2, zorder=4)
ax.scatter(x, probe, s=22, color=PROBE, zorder=6)
ax.scatter([x[-1]], [base], s=22, color=BASE, zorder=5)
ax.set_xscale('log'); ax.set_yscale('log')
for a in (ax.xaxis, ax.yaxis):
    a.set_minor_locator(matplotlib.ticker.NullLocator())
ally = np.r_[shared, probe, base]
xmin, xmax = x.min() / 4, x.max() * 2
ylo, yhi = 0.97 * ally.min(), ally.max() * 1.15
xt = [v for v in (1e16, 1e17, 1e18, 1e19) if xmin <= v <= xmax]
ax.set_xticks(xt, [f'10{str(int(np.log10(v))).translate(SUP)}' for v in xt])
ax.set_xlim(xmin, xmax); ax.set_ylim(ylo, yhi)
y_values(ax, [v for v in (3.25, 3.5, 3.75, 4, 4.5, 5, 5.5) if ylo <= v <= yhi / 1.05], '{:g}')
save(fig, HERE / 'layer_probe_300m_progress')
