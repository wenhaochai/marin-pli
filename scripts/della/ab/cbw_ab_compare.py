"""Compare the XLA_CMD_BUFFER_WHILE A/B runs on W&B (run on a vis node; reads marin-della).

  cbw_ab_compare.py [--ref muonh-qwen3-300m-della4xh100-v64k-sl4224] [--stem same] [--from 50]

Runs: ref (the real run, w31), <stem>-abbase (flag off), <stem>-abcbw (flag on); same seed, data and schedule.
Prints, over the steps all three logged (from --from on):
  - step time: median and mean throughput/duration per run (eval steps excluded by the median);
  - train/loss: max and mean |diff| per step, and the mean diff of 100-step means, for base-ref (the run-to-run band),
    cbw-ref and cbw-base;
  - eval/loss at the steps all three logged.
Verdict rule (harmless): every cbw-* loss statistic is no larger than ~1.5x the base-ref one, and no drift in the 100-step
means (sign changes, no growing trend).
"""
import argparse
import json

import numpy as np
import wandb

p = argparse.ArgumentParser()
p.add_argument("--ref", default="muonh-qwen3-300m-della4xh100-v64k-sl4224")
p.add_argument("--stem", default="")
p.add_argument("--from", dest="start", type=int, default=50)
p.add_argument("--project", default="marin-della")
p.add_argument("--base", default="")   # run-id overrides (testing)
p.add_argument("--cbw", default="")
a = p.parse_args()
stem = a.stem or a.ref
api = wandb.Api(timeout=120)


def hist(run_id, keys):
    r = api.run(f"{a.project}/{run_id}")
    out = {}
    for row in r.scan_history(keys=["_step", *keys]):
        out[row["_step"]] = row
    return out


names = {"ref": a.ref, "base": a.base or stem + "-abbase", "cbw": a.cbw or stem + "-abcbw"}
tr = {k: hist(v, ["train/loss"]) for k, v in names.items()}
th = {k: hist(v, ["throughput/duration"]) for k, v in names.items() if k != "ref"}
ev = {k: hist(v, ["eval/loss"]) for k, v in names.items()}
steps = sorted(set.intersection(*(set(s for s in h if s >= a.start) for h in tr.values())))
res = {"runs": names, "steps_compared": [steps[0], steps[-1], len(steps)] if steps else None}
for k, h in th.items():
    d = np.array([h[s]["throughput/duration"] for s in h if s >= a.start])
    res[f"step_s_{k}"] = dict(median=float(np.median(d)), mean=float(d.mean()), n=len(d)) if len(d) else None
if res.get("step_s_base") and res.get("step_s_cbw"):
    res["speedup_median"] = res["step_s_base"]["median"] / res["step_s_cbw"]["median"]
L = {k: np.array([tr[k][s]["train/loss"] for s in steps]) for k in tr}
for x, y in (("base", "ref"), ("cbw", "ref"), ("cbw", "base")):
    d = L[x] - L[y]
    n = len(d) // 100 * 100
    blocks = d[:n].reshape(-1, 100).mean(1) if n else np.array([])
    res[f"loss_{x}_minus_{y}"] = dict(max_abs=float(np.abs(d).max()), mean_abs=float(np.abs(d).mean()), mean=float(d.mean()),
                                     block100_means=[round(float(b), 5) for b in blocks])
es = sorted(set.intersection(*(set(h) for h in ev.values())))
res["eval_loss"] = {int(s): {k: ev[k][s]["eval/loss"] for k in ev} for s in es}
print(json.dumps(res, indent=1))
