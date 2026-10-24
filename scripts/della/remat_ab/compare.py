"""Compare the REMAT A/B arms from W&B (run on a vis node: it needs api.wandb.ai).

    python compare.py --a <run id> --b <run id> [--ref <run id> ...] [--max-step 2000] [--window 100]

For every pair (a, b) and (a, ref): train loss difference per step and of --window-step means, eval losses at the steps
both logged, and the mean step duration (throughput/duration, steps >= 20). b counts as "same curve" when its gaps to a are
no larger than those of the references (the same run on other hardware or a seed replicate). Prints one JSON line per pair.
"""
import argparse
import json

import numpy as np
import wandb

p = argparse.ArgumentParser()
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--ref", action="append", default=[])
p.add_argument("--max-step", type=int, default=2000)
p.add_argument("--window", type=int, default=100)
p.add_argument("--project", default="reself/marin-della")
args = p.parse_args()
api = wandb.Api(timeout=120)


def history(run_id):
    run = api.run(f"{args.project}/{run_id}")
    train, dur, ev = {}, {}, {}
    for r in run.scan_history(page_size=5000, min_step=0, max_step=args.max_step + 1):
        s = r.get("_step")
        if s is None or s > args.max_step:
            continue
        if r.get("train/loss") is not None:
            train[s] = r["train/loss"]
        if r.get("throughput/duration") is not None:
            dur[s] = r["throughput/duration"]
        for k, v in r.items():
            if k.startswith("eval/") and k.endswith("loss") and v is not None:
                ev.setdefault(k, {})[s] = v
    return train, dur, ev


H = {r: history(r) for r in dict.fromkeys([args.a, args.b, *args.ref])}


def pair(x, y):
    tx, dx, ex = H[x]
    ty, dy, ey = H[y]
    steps = sorted(set(tx) & set(ty))
    if not steps:
        return dict(a=x, b=y, error="no common steps")
    d = np.array([ty[s] - tx[s] for s in steps])
    w = args.window
    wins = [(s0, float(np.mean([ty[s] - tx[s] for s in steps if s0 <= s < s0 + w]))) for s0 in range(0, steps[-1] + 1, w)
            if any(s0 <= s < s0 + w for s in steps)]
    late = [s for s in steps if s >= steps[-1] // 2]
    evals = {k: {s: round(ey[k][s] - ex[k][s], 5) for s in sorted(set(ex[k]) & set(ey.get(k, {})))} for k in ex if k in ey}
    evals = {k: v for k, v in evals.items() if v and ("macro" in k or k == "eval/loss")}
    return dict(a=x, b=y, steps=[steps[0], steps[-1]], n=len(steps),
                mean_abs_dloss=float(np.mean(np.abs(d))), mean_dloss_second_half=float(np.mean([ty[s] - tx[s] for s in late])),
                max_abs_window_dloss=max(abs(v) for _, v in wins), last_window_dloss=wins[-1][1],
                eval_dloss=evals,
                step_s_a=float(np.mean([v for s, v in dx.items() if s >= 20])) if dx else None,
                step_s_b=float(np.mean([v for s, v in dy.items() if s >= 20])) if dy else None)


for y in [args.b, *args.ref]:
    print("PAIR", json.dumps(pair(args.a, y)), flush=True)
