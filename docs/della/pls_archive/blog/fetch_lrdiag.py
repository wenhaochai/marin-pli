"""Depth / learning-rate diagnostic (depth_lr_diag.sbatch, array 15039135): 130m baseline, no readout heads, 600-step
schedule; arms d6/d12/d24/d48 at the recipe LR and d48 at LR x 1/2, 1/4, 1/8. Prints train/loss (50-step means) at
steps 50..600 per finished or running run; writes lrdiag.json."""
import json, numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
ARMS = [("d6", ""), ("d12", "-d12"), ("d24", "-d24"), ("d48", "-d48"), ("d48 lr/2", "-d48-lrm0.5"), ("d48 lr/4", "-d48-lrm0.25"), ("d48 lr/8", "-d48-lrm0.125")]
out = {}
for name, tag in ARMS:
    try:
        r = api.run(P + tag + "-diag600")
    except Exception:
        print(f"{name:10s} not on W&B yet"); continue
    h = [(x["_step"], x["train/loss"]) for x in r.scan_history(keys=["_step", "train/loss"], page_size=2000) if x.get("train/loss") is not None]
    if not h:
        print(f"{name:10s} {r.state} no steps yet"); continue
    st, l = map(np.array, zip(*h))
    pts = {int(t): round(float(l[(st > t - 50) & (st <= t)].mean()), 3) for t in range(50, 601, 50) if ((st > t - 50) & (st <= t)).any()}
    out[name] = dict(state=r.state, last_step=int(st[-1]), loss=pts)
    print(f"{name:10s} {r.state:8s} step {int(st[-1]):4d}  " + "  ".join(f"{t}:{v}" for t, v in pts.items()))
json.dump(out, open("/scratch/gpfs/GROUP/USER/tmp/pls_blog/lrdiag.json", "w"), indent=1)
