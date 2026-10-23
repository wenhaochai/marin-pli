"""The largest AttnRes source and what it carries (A2 of the 2026-10-21 survey). Status figure for the owner.
Data: tmp/pls_status/attnres_massive/{full,full-s1,full-sep}.json (marin-pls-auto10 scripts/della/attnres_massive.py,
16 x 1024 paloma c4_en validation tokens, Full AttnRes 48 layers, step-4958): per model the source (sublayer output)
with the largest mean per-token RMS, divided by the median source's RMS; and the final cross-entropy increase when
that source is replaced, as it is written, by its mean over a second text (mean_held) or by zero.
Output: a2_massive.{pdf,png}. Run on a vis node: nice -n 19 $PY a2_massive_plot.py"""
import json
from pathlib import Path
import numpy as np
from style import *

apply_style()
D = Path('/scratch/gpfs/GROUP/USER/tmp/pls_status/attnres_massive')
M = [('full', 'Ordinary\nseed 0'), ('full-s1', 'Ordinary\nseed 1'), ('full-sep', 'Separate heads\nseed 0')]
r = {k: json.load(open(D / f'{k}.json')) for k, _ in M}
x = np.arange(len(M))
fig, axes = canvas(1, 2, width=WIDTH_TEXT, panel_height=2.0, title='Largest AttnRes source, Full AttnRes, 48 layers',
                   legend=[('Replaced by its mean over other text', BLUE, 'box'), ('Set to zero', LIGHT, 'box')])
a, b = axes[0]
a.bar(x, [r[k]['target']['ratio'] for k, _ in M], width=0.55, color=MEDIUM[0])
a.set_yscale('log'); a.set_ylim(1, 3e5); a.set_yticks([1, 10, 100, 1e3, 1e4, 1e5])
a.set_yticklabels(['1', '10', '100', '1,000', '10,000', '100,000']); a.minorticks_off()
w = 0.36
b.bar(x - w / 2, [max(r[k]['variants']['mean_held']['delta'], 0) for k, _ in M], width=w, color=BLUE)
b.bar(x + w / 2, [r[k]['variants']['zero']['delta'] for k, _ in M], width=w, color=LIGHT)
nice_y(b, 0, 10, zero=True, headroom=0.15, fmt='{:g}')
for ax, name in ((a, 'Size relative to the median source'), (b, 'Loss increase (nats)')):
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in M]); ax.grid(axis='x', visible=False); ax.set_xlim(-0.6, len(M) - 0.4)
    panel_label(ax, name)
save(fig, Path(__file__).parent / 'a2_massive')
for k, n in M:
    v = r[k]['variants']
    print(k, r[k]['target']['write'], f"ratio {r[k]['target']['ratio']:.0f}", f"cos to mean {r[k]['measures']['target_cos']['mean']:.3f}",
          f"rms cv {r[k]['measures']['target_rms']['cv']:.2f}", {kk: round(vv['delta'], 4) for kk, vv in v.items()})
