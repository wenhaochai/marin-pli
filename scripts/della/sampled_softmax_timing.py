"""Step-time comparison of sampled-softmax runs against baseline runs, from W&B ``throughput/duration``.

    .venv/bin/python scripts/della/sampled_softmax_timing.py --base=<run_id>[,...] --ss=<run_id>[,...]

For every run: the mean step time over steady-state steps (step >= 10, dropping steps that ran an eval or a checkpoint:
anything above 3x the run's median), split by the candidate count logged in train/ss/candidates for ss runs, and the
summed training time. The ss/baseline ratio is taken per stage against the pooled baseline mean, and for the whole run
as summed ss step time over summed baseline step time at equal steps. Runs are admitted only when finished.
"""
import argparse

import numpy as np
import wandb

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--ss", required=True)
p.add_argument("--project", default="marin-della")
a = p.parse_args()
api = wandb.Api(timeout=120)


def steady(run):
    rows = [r for r in run.scan_history(keys=["_step", "throughput/duration"]) if r["_step"] >= 10]
    d = np.array([r["throughput/duration"] for r in rows])
    keep = d < 3 * np.median(d)
    return {r["_step"]: x for r, x, k in zip(rows, d, keep) if k}, len(rows) - int(keep.sum())


base_times, base_total = [], []
for rid in a.base.split(","):
    run = api.run(f"{a.project}/{rid}")
    assert run.state == "finished", (rid, run.state)
    t, dropped = steady(run)
    base_times.append(np.mean(list(t.values())))
    base_total.append(np.mean(list(t.values())) * run.config["trainer"]["num_train_steps"])
    print(f"base {rid}: {np.mean(list(t.values())) * 1e3:.1f} ms/step over {len(t)} steps ({dropped} eval/ckpt steps dropped)")
base_mean = float(np.mean(base_times))
print(f"BASELINE pool: {base_mean * 1e3:.1f} ms/step (sd {np.std(base_times, ddof=1) * 1e3 if len(base_times) > 1 else 0:.1f} across {len(base_times)} runs)")

for rid in a.ss.split(","):
    run = api.run(f"{a.project}/{rid}")
    assert run.state == "finished", (rid, run.state)
    t, dropped = steady(run)
    cand = {r["_step"]: int(r["train/ss/candidates"]) for r in run.scan_history(keys=["_step", "train/ss/candidates"])}
    by_p: dict[int, list[float]] = {}
    for s, x in t.items():
        by_p.setdefault(cand.get(s, -1), []).append(x)
    parts = " | ".join(f"P={pp}: {np.mean(v) * 1e3:.1f} ms x{len(v)} ({base_mean / np.mean(v):.2f}x)" for pp, v in sorted(by_p.items()))
    n_steps = run.config["trainer"]["num_train_steps"]
    # whole-run time at equal steps: each stage's mean step time times its step count (dropped steps filled at that mean)
    stage_steps: dict[int, int] = {}
    for s, pp in cand.items():
        stage_steps[pp] = stage_steps.get(pp, 0) + 1
    ss_total = sum(np.mean(by_p[pp]) * c for pp, c in stage_steps.items() if pp in by_p)
    print(f"ss {rid}: {parts}")
    print(f"  whole run at equal steps ({n_steps}): ss {ss_total / 60:.1f} min vs baseline {base_mean * n_steps / 60:.1f} min -> {base_mean * n_steps / ss_total:.2f}x ({dropped} eval/ckpt steps dropped)")
