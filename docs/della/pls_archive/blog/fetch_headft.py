"""Figure 10 data: held-out loss by layer (eval/L{k}/loss, k = 0..11, W&B summaries) of the 300m separate-heads and
probes runs before (the finished runs) and after their heads were retrained on the frozen backbone for 2000 steps
(-headft2000). Also the final training loss by layer (mean of the last 50 logged steps). -> headft.json"""
import json, numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-300m-della4xh100"
keys = [f"train/pls/L{k}" for k in range(12)]
out = {}
for src in ("-pls1-sep", "-pls1-sep-bbfrozen"):
    for tag in (src, src + "-headft2000"):
        r = api.run(P + tag)
        ev = [r.summary.get(f"eval/L{k}/loss") for k in range(12)]
        rows = [x for x in r.scan_history(keys=["_step", *keys], page_size=5000) if all(x.get(k) is not None for k in keys)]
        tr = np.array([[x[k] for k in keys] for x in rows[-50:]]).mean(0)
        out[tag] = dict(state=r.state, eval=ev, train=[round(float(v), 4) for v in tr], last_step=rows[-1]["_step"])
        print(f"{tag:32s} {r.state:9s} step {rows[-1]['_step']:5d}\n  eval  {np.round(np.array(ev, float), 3).tolist()}\n  train {np.round(tr, 3).tolist()}")
json.dump(out, open("/scratch/gpfs/GROUP/USER/tmp/pls_blog/headft.json", "w"), indent=1)
