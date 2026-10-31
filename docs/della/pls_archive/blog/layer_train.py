"""Per-layer train CE (train/pls/L0..L5) of four 130m runs, with the baseline's train loss, 50-step running mean,
steps 200-4958. Source: layers.npz (fetch_layers.py, W&B reself/marin-della muonh-qwen3-130m-della4xh100{-pls1,
-pls1-dh,-pls1-sep,-pls1-off80} and the baseline run). Trained readouts: CE through the readout each layer is trained
with; after the switch at step 80 the off80 run logs the stop-gradient monitor (shared head, every 16th position).
Output: layer_train.pdf/png."""
from pathlib import Path
import numpy as np
from style import *
HERE = Path(__file__).resolve().parent
_raw = np.load(HERE / "layers.npz", allow_pickle=True)
D = {k: np.array([[np.nan if v is None else v for v in r] for r in _raw[k]], float) if _raw[k].ndim == 2 else np.array([np.nan if v is None else v for v in _raw[k]], float) for k in _raw.files}
apply_style()
COLS = [tone('blue', g) for g in (300, 400, 500, 600, 700, 900)]


def smooth(s, y, w=50):
    k = np.ones(w) / w
    return s[w - 1:], np.convolve(y, k, mode="valid")


runs = [("-pls1", "Shared head"), ("-pls1-dh", "Read-only shared head"), ("-pls1-sep", "Own heads"), ("-pls1-off80", "Switched off at step 80")]
fig, axes = canvas(rows=2, cols=2, width=WIDTH_POST, panel_height=1.3, title='Per-layer training loss, 130m',
                   legend=[(f'L{k}', c) for k, c in enumerate(COLS)] + [('Baseline', GREY, 'dash')],
                   quantity='Cross-entropy (nats)', title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
bs, bl = smooth(D["base/step"], D["base/ce"])
for ax, (n, name) in zip(axes.flat, runs):
    s, ce = D[n + "/step"], D[n + "/ce"]
    hi = 0
    for k in range(6):
        x, y = smooth(s, ce[:, k])
        m = x >= 200
        ax.plot(x[m], y[m], color=COLS[k], lw=1.1, zorder=3)
        hi = max(hi, y[m].max())
    m = bs >= 200
    ax.plot(bs[m], bl[m], color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.set_xticks([1000, 2000, 3000, 4000, 5000], ['1k', '2k', '3k', '4k', '5k'])
    ax.set_xlim(200, 5000)
    if hi > 6:
        ax.set_ylim(2, 10)
        y_values(ax, [2, 4, 6, 8, 10], '{:.0f}')
    else:
        nice_y(ax, bl[m].min(), hi, fmt='{:.1f}')
    room(ax)
    panel_label(ax, name)
save(fig, HERE / 'layer_train')
