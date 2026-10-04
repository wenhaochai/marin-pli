"""Per-layer train CE (train/pls/L{k}, k = 0..11) of the 300m separate-heads run (W&B reself/marin-della
muonh-qwen3-300m-della4xh100-pls1-sep, job 14940604) -> sep300.npz (step, ce). Prints the run state and the final values
(mean of the last 50 logged steps)."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
r = api.run("reself/marin-della/muonh-qwen3-300m-della4xh100-pls1-sep")
keys = [f"train/pls/L{k}" for k in range(12)]
rows = [x for x in r.scan_history(keys=["_step", *keys], page_size=5000) if all(x.get(k) is not None for k in keys)]
step = np.array([x["_step"] for x in rows]); ce = np.array([[x[k] for k in keys] for x in rows])
print(r.state, "step", step.max(), "rows", len(step), "dups", len(step) - len(np.unique(step)), "nonfinite", int((~np.isfinite(ce)).sum()))
print("final L0..L11", np.round(ce[-50:].mean(0), 4))
np.savez("/scratch/gpfs/GROUP/USER/tmp/pls_blog/sep300.npz", step=step, ce=ce, state=r.state)
