"""Final-layer c4_en bpb of each per-layer-supervision run minus the baseline, 130m and 300m (Kaiyue muonh_qwen3).

Source: W&B reself/marin-della, run.summary['eval/paloma/c4_en-marin-tokenizer/bpb'] at the last eval step.
  130m baseline = mean of 8 runs: muonh-qwen3-130m-della4xh100{'', -s1, -s2, -s3, -ema0.999, -ema0.999-s1,
                  -ema0.999-s2, -ema0.999-s3}
  300m baseline = muonh-qwen3-300m-della4xh100 (one run)
  arms: muonh-qwen3-130m-della4xh100-pls1 (shared head), -pls1-dh (read-only shared head), -pls1-sep (own heads),
        muonh-qwen3-300m-della4xh100-pls1 (shared head); switched off early: -pls1-off80, -pls1-off140,
        -pls1-off0-150 (130m, shared head).
Gap = arm - baseline. Output: final_gap.pdf and final_gap.png (WIDTH_POST, 1600 px).
"""
from pathlib import Path

import numpy as np
import wandb

from style import *

HERE = Path(__file__).resolve().parent
KEY = 'eval/paloma/c4_en-marin-tokenizer/bpb'
P = 'reself/marin-della/muonh-qwen3-'
api = wandb.Api(timeout=120)


def bpb(run_id):
    return float(api.run(P + run_id).summary[KEY])


pool = [bpb('130m-della4xh100' + s) for s in ('', '-s1', '-s2', '-s3', '-ema0.999', '-ema0.999-s1', '-ema0.999-s2', '-ema0.999-s3')]
base130, sd130 = float(np.mean(pool)), float(np.std(pool, ddof=1))
base300 = bpb('300m-della4xh100')
always = [('130m shared', bpb('130m-della4xh100-pls1') - base130),
          ('130m read-only', bpb('130m-della4xh100-pls1-dh') - base130),
          ('130m own heads', bpb('130m-della4xh100-pls1-sep') - base130),
          ('300m shared', bpb('300m-della4xh100-pls1') - base300)]
early = [('off at step 80', bpb('130m-della4xh100-pls1-off80') - base130),
         ('off at step 140', bpb('130m-della4xh100-pls1-off140') - base130),
         ('linear 0 to 150', bpb('130m-della4xh100-pls1-off0-150') - base130)]
print(f'130m pool {base130:.5f} +- {sd130:.5f} (n={len(pool)}, range {min(pool):.5f}-{max(pool):.5f}); 300m base {base300:.5f}')
for name, d in always + early:
    print(f'{name.replace(chr(10), " "):22s} {d:+.5f}')

apply_style()
fig, axes = canvas(rows=2, cols=1, width=WIDTH_POST, panel_height=1.15, gap_rows=0.42, title='Final-layer c4_en bpb minus the baseline',
                   quantity='Gap (bpb)', title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
for ax, (label, items, fmt) in zip(axes.flat, [('Always on', always, '{:.2f}'), ('Switched off early', early, '{:.4f}')]):
    x = np.arange(len(items))
    ax.bar(x, [d for _, d in items], width=0.62, color=MEDIUM[0], zorder=3)
    ax.set_xticks(x, [n for n, _ in items])
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.xaxis.grid(False)  # a categorical axis has no grid lines
    nice_y(ax, 0, max(d for _, d in items), zero=True, fmt=fmt)
    room(ax)
    panel_label(ax, label)
save(fig, HERE / 'final_gap')
