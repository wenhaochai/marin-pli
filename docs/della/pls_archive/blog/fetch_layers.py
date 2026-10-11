"""Per-layer train CE (train/pls/L{k}) of the 130m pls runs from W&B reself/marin-della -> layers.npz."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
keys = [f"train/pls/L{k}" for k in range(6)]
out = {}
for n in ["-pls1", "-pls1-dh", "-pls1-sep", "-pls1-off80", "-pls1-off140", "-pls1-off0-150"]:
    rows = list(api.run(P + n).scan_history(keys=["_step", *keys], page_size=5000))
    out[n + "/step"] = np.array([r["_step"] for r in rows])
    out[n + "/ce"] = np.array([[r[k] for k in keys] for r in rows])
    print(n, len(rows), out[n + "/ce"][-1].round(3))
b = list(api.run(P).scan_history(keys=["_step", "train/loss"], page_size=5000))
out["base/step"] = np.array([r["_step"] for r in b]); out["base/ce"] = np.array([r["train/loss"] for r in b])
print("base", len(b), out["base/ce"][-1])
np.savez("/scratch/gpfs/GROUP/USER/tmp/pls_blog/layers.npz", **out)
