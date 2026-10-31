"""Per-layer training loss against cumulative training FLOPs, 130m (four runs) and 300m (shared head).

Source: flops.npz (fetch_flops.py; W&B reself/marin-della, train/pls/L{k} or train/loss, and
throughput/total_gflops, levanter's cumulative count from the config's flops_per_token, which includes the extra
lm_head per trained readout). The off80 run's count is the post-switch one for every step; its first 80 steps are
corrected up to the always-on per-step cost. 50-step running means, from step 200. Output: layer_flops_130m.*,
layer_flops_300m.*."""
from pathlib import Path
import numpy as np
from style import *
HERE = Path(__file__).resolve().parent
D = dict(np.load(HERE / "flops.npz"))
apply_style()
COLS = [tone('blue', g) for g in (300, 400, 500, 600, 700, 900)]
E = 1e9 / 1e18   # gflops -> 1e18 FLOPs

# off80: add the always-on per-step surplus for steps < 80
s, g = D["130-off80/step"], D["130-off80/gflops"]
surplus = D["130-shared/gflops"][-1] / D["130-shared/step"][-1] - g[-1] / s[-1]
D["130-off80/gflops"] = g + np.minimum(s, 80) * surplus


def smooth(x, y, w=50):
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


def curves(ax, name, layers, cols):
    s, x, ce = D[name + "/step"], D[name + "/gflops"] * E, D[name + "/ce"]
    m = s >= 200
    hi = 0
    for k, c in zip(layers, cols):
        xs, y = smooth(x[m], ce[m, k])
        ax.plot(xs, y, color=c, lw=1.1, zorder=3)
        hi = max(hi, y.max())
    return hi


def base(ax, name):
    s, x, ce = D[name + "/step"], D[name + "/gflops"] * E, D[name + "/ce"][:, 0]
    m = s >= 200
    xs, y = smooth(x[m], ce[m])
    ax.plot(xs, y, color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=4)
    return y.min()


# 130m
fig, axes = canvas(rows=2, cols=2, width=WIDTH_POST, panel_height=1.3, title='Per-layer training loss against compute, 130m',
                   legend=[(f'L{k}', c) for k, c in enumerate(COLS)] + [('Baseline', GREY, 'dash')],
                   quantity='Cross-entropy (nats)', xlabel='Training compute (10¹⁸ FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
for ax, (n, label) in zip(axes.flat, [("130-shared", "Shared head"), ("130-ro", "Read-only shared head"),
                                      ("130-own", "Own heads"), ("130-off80", "Switched off at step 80")]):
    hi = curves(ax, n, range(6), COLS)
    lo = base(ax, "130-base")
    ax.set_xticks([0, 2, 4, 6], ['0', '2', '4', '6'])
    ax.set_xlim(0, 7.2)
    if hi > 6:
        ax.set_ylim(2, 10)
        y_values(ax, [2, 4, 6, 8, 10], '{:.0f}')
    else:
        nice_y(ax, lo, hi, fmt='{:.1f}')
    room(ax)
    panel_label(ax, label)
save(fig, HERE / 'layer_flops_130m')

# 300m
L300 = [0, 2, 4, 6, 8, 11]
fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=1.9, title='Per-layer training loss against compute, 300m',
                   legend=[(f'L{k}', c) for k, c in zip(L300, COLS)] + [('Baseline', GREY, 'dash')],
                   quantity='Cross-entropy (nats)', xlabel='Training compute (10¹⁸ FLOPs)',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
ax = axes.flat[0]
hi = curves(ax, "300-shared", L300, COLS)
lo = base(ax, "300-base")
ax.set_xticks([0, 10, 20, 30, 40, 50], ['0', '10', '20', '30', '40', '50'])
ax.set_xlim(0, 52)
nice_y(ax, lo, hi, fmt='{:.1f}')
room(ax)
panel_label(ax, 'Shared head')
save(fig, HERE / 'layer_flops_300m')

# numbers at matched compute
for n, b in [("130-shared", "130-base"), ("130-own", "130-base"), ("300-shared", "300-base")]:
    budget = D[b + "/gflops"][-1]
    i = np.searchsorted(D[n + "/gflops"], budget)
    print(f"{n}: at the baseline's total compute ({budget*E:.2f}e18) it is at step {int(D[n + '/step'][i])}, "
          f"last-layer loss {D[n + '/ce'][i-25:i+25, -1].mean():.3f} vs baseline final {D[b + '/ce'][-50:, 0].mean():.3f}")
