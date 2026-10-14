"""Layer-48 CE rise after removing the readout-invisible (zero-sum) part of the HC / mHC streams after layer c.
Data: job 15534928 log (scripts/della/pls_hc_hidden_ablate_cpu.py; final 48-layer checkpoints, paloma c4_en validation,
2 x 512 tokens). Output hc_ablate.png/pdf."""
import re
from pathlib import Path
from style import *
HERE = Path(__file__).parent
rows = {}
for line in open(HERE / 'jobs/hc_ablate_15534928.log'):
    m = re.match(r'(\w+) (separate heads|probes)\s+L48 CE [\d.]+; rise .* layer (.*)$', line.strip())
    if m:
        rows[(m.group(1), m.group(2))] = [(int(a), float(b)) for a, b in (t.split(':') for t in m.group(3).split())]
apply_style()
fig, axes = canvas(1, 2, width=WIDTH_POST, panel_height=1.6, title='Last-layer loss after removing what the readouts cannot see',
                   legend=[('Separate heads', BLUE), ('Probes only', LIGHT)], quantity='Rise in layer-48 loss (nats)', title_pt=9.2, tick_pt=7.35)
for ax, arch, lab in zip(axes.flat, ('hc', 'mhc'), ('Hyper-Connections', 'mHC')):
    for kind, col in (('separate heads', BLUE), ('probes', LIGHT)):
        xs, ys = zip(*rows[(arch, kind)])
        ax.plot(xs, ys, color=col, lw=1.5, marker='o', ms=3)
    ax.set_xticks([4, 12, 20, 28, 36, 44]); nice_y(ax, 0, 2.7, zero=True, fmt='{:.1f}'); room(ax); panel_label(ax, lab)
save(fig, HERE / 'hc_ablate')
