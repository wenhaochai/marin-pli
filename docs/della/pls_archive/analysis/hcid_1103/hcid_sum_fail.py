"""HC variants whose mixing keeps the stream sum (wq w959) or is one learned number (wq w960), against HC and HC with
identity mixing (w811), all ordinary models (probes runs, PLS_DETACH_BB=1) on the fa4thd stack.

Data: /scratch/gpfs/GROUP/USER/tmp/pls_hcid/sumfail_data.json, from sumfail_data.py (W&B reself/marin-della
muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-ahc[-hcsum|-hcscalar|-hcid]-fa4thd-vs-d48; checkpoints at step 2,130 for
w959 and w960, step 4,958 for HC).
Top left: layer-48 train cross-entropy (train/pls/L47) by step, first 300 steps, every step.
Top right: train cross-entropy by layer, mean of steps 2,100-2,197.
Bottom left: largest |eigenvalue| of the static mixing each run applies, mean of the two sublayers per layer. HC: c_res +
res. w959: the same after the row-sum correction.
Bottom right: log10 of the cumulative static gain of the stream sum by layer: product over sublayers of the mean row sum
of c_res + res (HC) or of g (w960). w811 and w959 keep the sum, so their gain is 1.
w959 and w960 are still training (step 2,200 of 4,958). Output: hcid_sum_fail.{pdf,png}.
"""
import json
from pathlib import Path

import numpy as np
from style import *

HERE = Path(__file__).parent
apply_style()
d = json.load(open("/scratch/gpfs/GROUP/USER/tmp/pls_hcid/sumfail_data.json"))
RED = tone("red", 600)
GREEN = tone("green", 600)
C = {"HC": BLUE, "w811": GREY, "w959": RED, "w960": GREEN}
NAME = {"HC": "HC", "w811": "HC, mixing = I", "w959": "HC, mixing keeps the sum", "w960": "HC, mixing = a scalar"}
ORDER = ["HC", "w811", "w960", "w959"]
fig, axes = canvas(2, 2, width=WIDTH_TEXT, panel_height=1.35, title="Ordinary models with changed HC mixing",
                   legend=[(NAME[k], C[k]) for k in ORDER])
(a, b), (c, e) = axes
for k in ORDER:
    s = np.array(d["l48"][k])
    m = s[:, 0] <= 300
    a.plot(s[m, 0], s[m, 1], color=C[k], lw=1.2)
    b.plot(np.arange(1, 49), d["ce"][k], color=C[k], lw=1.2)
nice_y(a, 4, 14, fmt="{:.0f}")
a.set_xlim(0, 300)
a.set_xticks([0, 100, 200, 300])
room(a)
panel_label(a, "Layer 48 cross-entropy (nats) by step")
nice_y(b, 2, 14, fmt="{:.0f}")
b.set_xlim(1, 48)
b.set_xticks([1, 12, 24, 36, 48])
room(b)
panel_label(b, "Cross-entropy (nats) by layer, steps 2,100-2,197")
for k in ("HC", "w959"):
    r = d["mix"][k]["rows"]
    v = np.array([x["maxeig"] for x in r]).reshape(48, 2).mean(1)
    c.plot(np.arange(1, 49), v, color=C[k], lw=1.2)
nice_y(c, 0.75, 2.25, fmt="{:.2f}")
c.set_xlim(1, 48)
c.set_xticks([1, 12, 24, 36, 48])
room(c)
panel_label(c, "Largest |eigenvalue| of the static mixing by layer")
g = np.cumsum(np.log10([x["rowsum"] for x in d["mix"]["HC"]["rows"]])).reshape(48, 2)[:, 1]
h = np.cumsum(np.log10([x["g"] for x in d["mix"]["w960"]["rows"]])).reshape(48, 2)[:, 1]
e.plot(np.arange(1, 49), g, color=C["HC"], lw=1.2)
e.plot(np.arange(1, 49), h, color=C["w960"], lw=1.2)
e.plot([1, 48], [0, 0], color=C["w811"], lw=1.2)
e.plot([1, 48], [0, 0], color=C["w959"], lw=1.2, ls=(0, (3, 3)))
nice_y(e, -8, 4, fmt="{:.0f}")
e.set_xlim(1, 48)
e.set_xticks([1, 12, 24, 36, 48])
room(e)
panel_label(e, "log10 cumulative gain of the stream sum by layer")
save(fig, HERE / "hcid_sum_fail")
