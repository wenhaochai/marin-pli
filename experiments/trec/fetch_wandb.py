"""Per-step train loss and learning rates of the TREC runs from W&B (reself/marin-della), cached on disk: a finished
run never changes, so only missing or unfinished runs are fetched (run on a vis node).

    python experiments/trec/fetch_wandb.py RUN_ID... [--out DIR]
"""

import argparse
import json
import os

import wandb

KEYS = ["train/loss", "optim/learning_rate", "optim/adam_lr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="/scratch/gpfs/GROUP/USER/project/trec/cache/wandb")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    api = wandb.Api(timeout=120)
    for rid in a.runs:
        path = os.path.join(a.out, f"{rid}.json")
        if os.path.exists(path) and json.load(open(path)).get("state") == "finished":
            continue
        r = api.run(f"reself/marin-della/{rid}")
        rows = {k: {} for k in KEYS}
        for row in r.scan_history(keys=["_step"] + KEYS, page_size=10000):
            for k in KEYS:
                if row.get(k) is not None:
                    rows[k][int(row["_step"])] = float(row[k])
        out = {"run": rid, "state": r.state, "config": r.config,
               **{k: [[s, v] for s, v in sorted(d.items())] for k, d in rows.items()}}
        json.dump(out, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        print(rid, r.state, {k: len(d) for k, d in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
