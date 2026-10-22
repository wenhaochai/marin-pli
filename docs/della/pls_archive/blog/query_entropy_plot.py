"""Read sharpness against logit scale, every AttnRes sublayer read (A1 of the 2026-10-21 survey). Status figure for the
owner, not a page figure. Data: checkup/*.json through query_entropy.rows (s = |query * key_gain| sqrt(H); normalised
entropy = mean per-token entropy of the read / log(number of sources)). Output: a1_query_entropy.{pdf,png}.
Run on a vis node: nice -n 19 python3 query_entropy_plot.py
"""
import json
import math
from pathlib import Path

from query_entropy import HERE, WIDTH, rows
from style import *

apply_style()
OUT = Path(__file__).parent
BLUE_, RED_, GREEN_ = STRONG[0], STRONG[1], STRONG[3]
fig, axes = canvas(1, 2, width=WIDTH_TEXT, panel_height=2.0, title='AttnRes read sharpness by logit scale',
                   legend=[('Probes on the ordinary model', BLUE_, 'dot'), ('Separate heads', RED_, 'dot'),
                           ('Kimi K3', GREEN_, 'dot')],
                   quantity='Normalised read entropy', xlabel='Logit scale s = |w| √H (log)')
for ax, name, files in ((axes[0][0], 'Full AttnRes, 48 layers', (('full', BLUE_), ('full-sep', RED_))),
                        (axes[0][1], 'Block sources: Block AttnRes and K3', (('block', BLUE_), ('block-sep', RED_), ('k3', GREEN_)))):
    for f, c in files:
        r = rows(json.load(open(HERE / f'{f}.json')), f, WIDTH.get(f, 512))
        dots(ax, [x['s'] for x in r], [x['hn'] for x in r], c, size=10)
    ax.set_xscale('log')
    ax.set_xlim(1.5, 300)
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(['2', '5', '10', '20', '50', '100', '200'])
    ax.minorticks_off()
    nice_y(ax, 0, 1, zero=True, headroom=0.12, fmt='{:.2f}')
    panel_label(ax, name)
save(fig, OUT / 'a1_query_entropy')
