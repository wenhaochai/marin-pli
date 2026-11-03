"""Preflight of the float32-stream and move designs after the audit of 2026-11-03: small-model replays on a della-vis2 A40 GPU. Not a training result of the real model.

Data: /scratch/gpfs/GROUP/USER/tmp/pls_hcid/tf32_gpu/ (replay_hc6.py). final/*.jsonl: 3,000 steps on training tokens (Nemotron-CC hq_actual part 0). sweep/*.jsonl: validation tokens,
valid to step 940 (the array ends at step 944). Model: HC, 48 layers, 4 streams, width 96, batch 16, 64 tokens, vocabulary 2,048 (uniform loss 7.62 nats), probes run, DepthArchMuonHConfig with
the 130m numbers, bfloat16 compute. Each record is one batch, the same batches in every run of a folder, every 20 steps.
Top left: relative error of the stream sum after one layer of the real hyper_step (writes off, random mixing) against |differences| / |sum| of the input, from
scripts/della/depth_arch_hcsumf32_gpu_test.py: float32 dots at default precision (TF32, commit 52942aeb4e) and with Precision.HIGHEST (exact, commit 40f77cde85).
Top right: layer 48 cross-entropy (train/pls/L47) by step, rolling mean over 10 records (200 steps), because the batch noise is larger than the differences between runs.
Bottom left: |stream differences| / |stream sum| at layer 48 by step, rolling median over 10 records: mean token norm of H_s - mean_s H_s over mean token norm of the stream sum / 4. Lines stop at overflow.
Bottom right: positive off-diagonal mass of the static mixing by step (sweep/), mean over sublayers: sum over o != i of max(A[i, o], 0), mean over i. beta and shift: sp(a) = softplus(beta (a - shift)) / beta.
Output: hc_preflight_1103.{pdf,png}.
"""
import json
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
T = Path("/scratch/gpfs/GROUP/USER/tmp/pls_hcid/tf32_gpu")
apply_style()


def load(p):
    return [json.loads(x) for x in open(T / p)]


def rolling(y, w=10, f=np.mean):
    y = np.asarray(y, float)
    return np.array([f(y[max(0, i - w + 1): i + 1]) for i in range(len(y))])


RED, GREEN, PURPLE, YEL = tone("red", 600), tone("green", 600), tone("purple", 600), tone("yellow", 900)
KEYS = ["sumbf16", "sumf32_default", "sumf32_highest", "move", "identity", "learned"]
NAME = {"sumbf16": "Sum kept, bfloat16 streams", "sumf32_default": "Sum kept, float32 streams, TF32 dot", "sumf32_highest": "Sum kept, float32 streams, exact dot", "move": "Only moves, beta 50, shift 0.1", "identity": "Mixing = I", "learned": "HC"}
C = {"sumbf16": RED, "sumf32_default": YEL, "sumf32_highest": BLUE, "move": GREEN, "identity": PURPLE, "learned": GREY}
runs = {k: load(f"final/{k}.jsonl") for k in KEYS}
ERR = {"sumf32_default": ([1, 300, 1e4], [1.3e-3, 3.7e-1, 12.3]), "sumf32_highest": ([1, 300, 1e4], [3.9e-7, 1.1e-4, 3.8e-3])}
fig, axes = canvas(2, 2, width=WIDTH_TEXT, panel_height=1.5, title="Preflight replays on a small model", legend=[(NAME[k], C[k]) for k in KEYS])
(a, b), (c, e) = axes
for k in ("sumf32_default", "sumf32_highest"):
    a.plot(np.log10(ERR[k][0]), np.log10(ERR[k][1]), color=C[k], lw=1.2, marker="o", ms=3)
nice_y(a, -8, 2, fmt="{:.0f}")
a.set_xlim(-0.2, 4.2)
a.set_xticks([0, 2, 4])
room(a)
panel_label(a, "log10 sum error after one layer, by log10 |dif| / |sum|")
for k in KEYS:
    r = runs[k]
    st = np.array([x["step"] for x in r])
    b.plot(st, rolling([x["ce"]["48"] for x in r]), color=C[k], lw=1.0)
    ratio = np.array([x["dif"][47] / x["sum"][47] if x["sum"][47] > 0 else np.inf for x in r], float)
    fin = np.isfinite(ratio)
    n = int(np.argmin(fin)) if not fin.all() else len(ratio)
    c.plot(st[:n], np.log10(rolling(ratio[:n], f=np.median)), color=C[k], lw=1.0)
nice_y(b, 5.5, 8, fmt="{:.1f}")
b.set_xlim(0, 3000)
b.set_xticks([0, 1000, 2000, 3000])
room(b)
panel_label(b, "Layer 48 cross-entropy (nats) by step")
nice_y(c, -2, 6, fmt="{:.0f}")
c.set_xlim(0, 3000)
c.set_xticks([0, 1000, 2000, 3000])
room(c)
panel_label(c, "log10 |dif| / |sum| at layer 48 by step")
mv = [("learned", "HC", GREY, 1.2, 0.205), ("b20s00", "Beta 20, no shift", tone("green", 300), 1.0, 0.175), ("b50s10", "Beta 50, shift 0.1", GREEN, 1.4, 0.102), ("b100s05", "Beta 100, shift 0.05", tone("green", 800), 1.0, 0.070), ("b20s30", "Beta 20, shift 0.3", tone("green", 500), 1.0, 0.038)]
for tag, name, col, lw, yt in mv:
    r = [x for x in load(f"sweep/{tag}.jsonl") if x["step"] <= 940]
    e.plot([x["step"] for x in r], [x["offdiag"] for x in r], color=col, lw=lw)
    e.text(965, yt, name, color=col, fontsize=6.3, ha="left", va="center")
nice_y(e, 0, 0.2, zero=True, fmt="{:.2f}")
e.set_xlim(0, 1520)
e.set_xticks([0, 400, 800])
panel_label(e, "Positive off-diagonal mass by step")
save(fig, HERE / "hc_preflight_1103")
