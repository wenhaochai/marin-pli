"""Replay check of the trec-runs smoke (scripts/della/trec-runs-smoke.sbatch): with the final weights, the loss of each
of a run's last training batches must equal the train loss W&B logged at that step, since every schedule ends at
learning rate 0, so the last updates barely move the weights (w447 matched to 0.0001 nats). A mismatch means the
replayed batches are not the ones the run read. The EMA weights (<run>-ema) are not checked: they differ from the
weights that logged the loss; they pass when their raw run passes.

    python experiments/trec/smoke_check.py OUT_ROOT                  # writes OUT_ROOT/check.json, exit 1 if any run fails
    python experiments/trec/smoke_check.py --passed OUT_ROOT < list  # the "run step subpath" lines whose run passed
"""

import glob
import json
import os
import sys

import numpy as np

CACHE = "/scratch/gpfs/GROUP/USER/project/trec/cache/wandb"
TOL = 0.002   # nats, on the last training batch


def batch_losses(d):
    idx, loss = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        idx.append(z["index"])
        loss.append(z["seq_loss"].sum(1) / z["seq_weight"].sum(1))
        total = int(z["total_steps"])
    return np.concatenate(idx), np.concatenate(loss), total


def check(root):
    out, ok = {}, True
    for d in sorted(glob.glob(os.path.join(root, "muonh-*"))):
        name = os.path.basename(d)
        idx, loss, total = batch_losses(d)
        if name.endswith("-ema"):
            out[name] = {"pass": None, "batches": len(idx)}
            continue
        logged = dict((int(s), v) for s, v in json.load(open(os.path.join(CACHE, f"{name}.json")))["train/loss"])
        last = [(int(i), float(l), logged[int(i)]) for i, l in zip(idx, loss) if total - 12 <= i < total and int(i) in logged]
        diffs = [abs(a - b) for _, a, b in last]
        passed = bool(last) and last[-1][0] == total - 1 and diffs[-1] < TOL
        out[name] = {"pass": passed, "last_diff": diffs[-1] if diffs else None, "max_diff_last12": max(diffs) if diffs else None,
                     "steps": total, "batches": len(idx)}
        ok &= passed
        print(f"{'ok  ' if passed else 'FAIL'} {name}: last batch {diffs[-1] if diffs else float('nan'):.4f} nats from W&B, "
              f"last 12 at most {max(diffs) if diffs else float('nan'):.4f}", flush=True)
    for name, r in out.items():
        if r["pass"] is None:
            r["pass"] = out.get(name[:-len("-ema")], {}).get("pass", False)
            print(f"{'ok  ' if r['pass'] else 'FAIL'} {name}: EMA weights, {r['batches']} batches scored (not compared)")
    json.dump(out, open(os.path.join(root, "check.json"), "w"), indent=1)
    return ok


def passed(root):
    res = json.load(open(os.path.join(root, "check.json")))
    for line in sys.stdin:
        if not line.split():
            continue
        run, _, sub = line.split()
        if res.get(run + ("" if sub == "model" else "-ema"), {}).get("pass"):
            print(line.strip())
        else:
            print(f"skipped (smoke failed or missing): {line.strip()}", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1] == "--passed":
        passed(sys.argv[2])
    else:
        sys.exit(0 if check(sys.argv[1]) else 1)
