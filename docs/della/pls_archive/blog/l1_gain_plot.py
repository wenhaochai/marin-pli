"""Pre-norm scales and probe losses of the 48-layer Pre-LN runs (L1, Figure 36 data). Status figure for the owner.
Data: gains.json (fetch_gains.py: mean trained pre-norm scale per layer, LNS converted by 1/sqrt(l)) and d48g4.npz
(fetch_d48g4.py: <tag>/final = per-layer probe training loss, mean of the last 50 steps). Left: each run's mean scale
divided by Pre-LN's. Right: each run's final probe loss minus Pre-LN's. Output: l1_gain.{pdf,png}.
Run on a vis node in the blog data directory: nice -n 19 $PY l1_gain_plot.py"""
import json
from pathlib import Path
import numpy as np
from style import *

apply_style()
HERE = Path.cwd()
g = json.load(open(HERE / 'gains.json'))
d = np.load(HERE / 'd48g4.npz', allow_pickle=True)
fin = {k: d[f'-sep-bbfrozen-{t}/final'] for k, t in (('preln', 'apreln-vs-d48-g4'), ('lns', 'alns-vs-d48-g4'), ('gilns', 'apreln-gilns-vs-d48'))}
P = np.array(g['preln']['mean'])
x = np.arange(1, 49)
RED_, BLUE_, GREEN_ = STRONG[1], STRONG[0], STRONG[3]
fig, axes = canvas(1, 2, width=WIDTH_TEXT, panel_height=2.0, title='Pre-LN, LayerNorm Scaling and Pre-LN from 1/√ℓ, 48 layers',
                   legend=[('Pre-LN', RED_), ('LayerNorm Scaling, converted', BLUE_), ('Pre-LN, scales start at 1/√ℓ', GREEN_)],
                   xlabel='Layer')
a, b = axes[0]
a.plot(x, np.ones(48), color=RED_)
a.plot(x, np.array(g['lns']['mean']) / P, color=BLUE_)
a.plot(x, np.array(g['preln-gilns']['mean']) / P, color=GREEN_)
nice_y(a, 0, 1.2, zero=True, headroom=0.1, fmt='{:.1f}')
b.plot(x, np.zeros(48), color=RED_)
b.plot(x, fin['lns'] - fin['preln'], color=BLUE_)
b.plot(x, fin['gilns'] - fin['preln'], color=GREEN_)
nice_y(b, -0.6, 0.1, headroom=0.1, fmt='{:.1f}')
for ax, name in ((a, 'Mean trained pre-norm scale, relative to Pre-LN'), (b, 'Final probe training loss minus Pre-LN (nats)')):
    ax.set_xlim(0.5, 48.5); ax.set_xticks([1, 12, 24, 36, 48]); room(ax); panel_label(ax, name)
save(fig, HERE / 'l1_gain')
