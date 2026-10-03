"""Health check: last-layer CE of the running 300m 只训独立头 run vs the 300m baseline's train/loss at the same steps."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-300m-della4xh100"
keys = [f"train/pls/L{k}" for k in range(12)]
r = [x for x in api.run(P + "-pls1-sep-bbfrozen").scan_history(keys=["_step", *keys], page_size=5000)]
b = {x["_step"]: x["train/loss"] for x in api.run(P).scan_history(keys=["_step", "train/loss"], page_size=5000)}
s = np.array([x["_step"] for x in r]); ce = np.array([[x[k] for k in keys] for x in r])
for lo in (500, 1000, 2000, s.max() - 99):
    m = (s >= lo) & (s < lo + 100)
    base = np.mean([b[t] for t in s[m] if t in b])
    print(f"steps {lo}-{lo+99}: L11 {ce[m, 11].mean():.4f} baseline {base:.4f} diff {ce[m, 11].mean() - base:+.4f}; L0..L11 {np.round(ce[m].mean(0), 2)}")
