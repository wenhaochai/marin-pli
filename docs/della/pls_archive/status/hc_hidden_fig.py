"""Zero-sum (readout-invisible) share of the HC and mHC stream state by layer, final 48-layer checkpoints.
Data: hc_hidden.json from scripts/della/pls_hc_hidden_cpu.py (job 15534022; paloma c4_en validation, 2 x 512 tokens).
share_k = sum_s ||H_s - mean H||^2 / sum_s ||H_s||^2 after layer k. Output hc_hidden.png/pdf."""
import json
from pathlib import Path
import numpy as np
from style import *
HERE = Path(__file__).parent
r = json.load(open(HERE / 'hc_hidden.json'))['runs']
apply_style()
fig, axes = canvas(1, 2, width=WIDTH_POST, panel_height=1.6, title='Readout-invisible share of the stream state, 48 layers',
                   legend=[('Separate heads', BLUE), ('Probes only', LIGHT)], quantity='Zero-sum share of stream energy', title_pt=9.2, tick_pt=7.35)
for ax, arch, lab in zip(axes.flat, ('hc', 'mhc'), ('Hyper-Connections', 'mHC')):
    k = np.arange(1, 49)
    ax.plot(k, r[f'{arch} separate heads']['hidden'], color=BLUE, lw=1.5)
    ax.plot(k, r[f'{arch} probes']['hidden'], color=LIGHT, lw=1.5)
    ax.set_xticks([1, 12, 24, 36, 48]); nice_y(ax, 0, 0.9, zero=True, fmt='{:.1f}'); room(ax); panel_label(ax, lab)
save(fig, HERE / 'hc_hidden')
