"""Last-layer cost of separate heads: layer-48 loss of the separate-heads run minus that of the probes run (probes only,
backbone trained without layer losses), per 48-layer design. Finished designs: mean of the last 50 steps (train) and
the final eval, from /scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz (fetch_d48g4.py). Designs still training:
train loss at their current step from partial.npz (fetch_partial.py, 2026-10-13 ~21:00), status only, not final.
Output: l48gap.pdf / .png."""
from pathlib import Path
import numpy as np
from style import *
HERE = Path(__file__).resolve().parent
G = dict(np.load('/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz', allow_pickle=True))
P = dict(np.load(HERE / 'partial.npz'))
DONE = [('', 'Sandwich-LN'), ('apreln', 'Pre-LN'), ('alns', 'LayerNorm Scaling'), ('adeepnorm', 'DeepNorm'),
        ('akeel', 'KEEL'), ('aattnres_block', 'AttnRes (Block)'), ('aattnres', 'AttnRes (Full)')]
RUN = [('ahc', 'Hyper-Connections'), ('amhc', 'mHC'), ('amoda', 'MoDA')]
def t(a, s): return f'-sep{s}-{a}-vs-d48-g4' if a else f'-sep{s}-vs-d48-g4'
rows = []
for a, n in DONE:
    rows.append((n, G[t(a, '') + '/final'][-1] - G[t(a, '-bbfrozen') + '/final'][-1],
                 G[t(a, '') + '/eval'][-1] - G[t(a, '-bbfrozen') + '/eval'][-1]))
for a, n in RUN:
    s = int(min(P[t(a, '') + '/step'], P[t(a, '-bbfrozen') + '/step']))
    rows.append((f'{n}, step {s // 10 * 10}', P[t(a, '') + '/final'][-1] - P[t(a, '-bbfrozen') + '/final'][-1], None))
for r in rows: print(r)
apply_style()
fig, axes = canvas(1, 1, width=WIDTH_POST, panel_height=2.9, title='Last-layer cost of separate heads, 48 layers',
                   legend=[('Training loss', BLUE), ('Eval loss', LIGHT), ('Training loss, run not finished', GREY)],
                   quantity='Layer 48: separate heads minus probes (nats)', title_pt=9.2, tick_pt=TEXT_PT, side=0.10)
ax = np.ravel(axes)[0]
y = np.arange(len(rows))[::-1]; h = 0.36
for yi, (n, tr, ev) in zip(y, rows):
    run = ev is None
    ax.barh(yi + (h / 2 if ev is not None else 0), tr, height=h, color=GREY if run else BLUE)
    if ev is not None: ax.barh(yi - h / 2, ev, height=h, color=LIGHT)
ax.set_yticks(y, [r[0] for r in rows]); ax.tick_params(axis='y', length=0)
ax.set_xlim(-0.05, 0.2); ax.set_xticks([-0.05, 0, 0.05, 0.1, 0.15, 0.2])
ax.xaxis.set_major_formatter(lambda v, _: f'{v:.2f}'); ax.grid(axis='y', visible=False)
ax.axvline(0, color=AXIS, lw=0.8)
save(fig, HERE / 'l48gap')
