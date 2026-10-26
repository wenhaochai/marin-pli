"""Flood-level curves for VARIANT=flood (experiments/references/flood_qwen3.py; owner popup 2026-10-25).

For each tokenizer K: u(s) = the unique-token run's W&B train/loss at step s, r(s) = the plain 8-pass run's, both seed 0,
in nats per token of that tokenizer, each smoothed by a centred moving average of WINDOW steps (shorter at the ends).
The level for an allowed share f of the plain run's memorisation is b(s) = u(s) - f (u(s) - r(s)): f = 0 pins the 8-pass
run's batch loss to the unique run's, f = 0.5 lets it fall halfway to the plain 8-pass run's. Output:
logs/vocab_flood/level_<K>_f<f>.npy (entry s = b(s), length = the runs' last step + 1) and level_<K>.json (run ids, steps,
the per-eighth means of u, r and b). Run on a vis node (W&B access): python scripts/della/flood_levels.py
"""
import json
from pathlib import Path

import numpy as np
import wandb

P = "reself/marin-della/muonh-qwen3-300m-della4xh100"
RUNS = {128000: ("", "-rep8"), 32000: ("-v32k", "-v32k-rep8"), 8000: ("-v8k", "-v8k-rep8")}
FRACTIONS = {128000: (0.0, 0.5), 32000: (0.0,), 8000: (0.0,)}
WINDOW = 101
OUT = Path(__file__).resolve().parents[2] / "logs" / "vocab_flood"


def history(api, tag):
    run = api.run(P + tag)
    h = {int(x["_step"]): float(x["train/loss"]) for x in run.scan_history(keys=["_step", "train/loss"]) if x.get("train/loss") is not None}
    if run.state != "finished":
        raise RuntimeError(f"{tag}: run state {run.state}")
    n = max(h) + 1
    s = np.array(sorted(h)); v = np.array([h[i] for i in s])
    return np.interp(np.arange(n), s, v), len(h)


def smooth(x, w=WINDOW):
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x)); lo = np.maximum(0, i - w // 2); hi = np.minimum(len(x), i + w // 2 + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def eighths(x):
    n = len(x)
    return [round(float(x[i * n // 8:(i + 1) * n // 8].mean()), 4) for i in range(8)]


OUT.mkdir(parents=True, exist_ok=True)
api = wandb.Api(timeout=180)
for k, (tu, tr) in RUNS.items():
    (u, nu), (r, nr) = history(api, tu), history(api, tr)
    if len(u) != len(r):
        raise RuntimeError(f"K={k}: unique run has {len(u)} steps, 8-pass run {len(r)}")
    us, rs = smooth(u), smooth(r)
    meta = dict(k=k, unique=P + tu, rep8=P + tr, steps=len(u), logged=(nu, nr), window=WINDOW,
                unique_eighths=eighths(us), rep8_eighths=eighths(rs), levels={})
    for f in FRACTIONS[k]:
        b = (us - f * (us - rs)).astype(np.float32)
        np.save(OUT / f"level_{k}_f{f:g}.npy", b)
        meta["levels"][f"{f:g}"] = eighths(b)
    (OUT / f"level_{k}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
