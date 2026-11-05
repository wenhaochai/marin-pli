"""Final train and eval cross-entropy of w1000 (HC mixing, only moves, ordinary model) at selected layers. Train = mean of the last 50 logged steps, as on the page.
Usage: w1000_final.py [TAG ...]"""
import sys, numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
for tag in sys.argv[1:] or ["-sep-bbfrozen-ahc-hcmove-fa4thd-vs-d48"]:
    r = api.run(P + tag)
    print(tag, "state", r.state, "steps", r.summary.get("_step"))
    keys = [f"train/pls/L{k}" for k in (0, 1, 3, 11, 23, 35, 45, 46, 47)]
    rows = {}
    for x in r.scan_history(keys=["_step", *keys], page_size=2000, min_step=4800):
        rows[x["_step"]] = x
    st = sorted(rows)
    print(" last step", st[-1], "rows", len(st))
    last = st[-50:]
    print(" train:", " ".join(f"L{int(k.rsplit('L',1)[1]) + 1}={np.mean([rows[s][k] for s in last]):.3f}" for k in keys))
    print(" eval:", " ".join(f"L{k + 1}={r.summary.get(f'eval/L{k}/loss', float('nan')):.3f}" for k in (11, 23, 35, 47)))
