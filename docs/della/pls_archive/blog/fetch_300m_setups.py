"""Per-layer final training loss (mean of the last 50 logged steps of train/pls/L{k}, k = 0..11) of the 300m runs with
separate heads, the shared head and probes only (W&B reself/marin-della muonh-qwen3-300m-della4xh100-pls1-sep,
-pls1, -pls1-sep-bbfrozen), plus the 300m baseline's final train/loss. -> s300.npz"""
import numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-300m-della4xh100"
keys = [f"train/pls/L{k}" for k in range(12)]
out = {}
for tag, n in (("sep", "-pls1-sep"), ("shared", "-pls1"), ("probes", "-pls1-sep-bbfrozen")):
    r = api.run(P + n)
    rows = [x for x in r.scan_history(keys=["_step", *keys], page_size=5000) if all(x.get(k) is not None for k in keys)]
    ce = np.array([[x[k] for k in keys] for x in rows]); st = np.array([x["_step"] for x in rows])
    out[tag] = ce[-50:].mean(0)
    print(tag, r.state, "last step", st.max(), "nonfinite", int((~np.isfinite(ce)).sum()), "finals", np.round(out[tag], 4))
b = [x["train/loss"] for x in api.run(P).scan_history(keys=["_step", "train/loss"], page_size=5000)]
out["baseline"] = np.array(float(np.mean(b[-50:])))
print("baseline", round(float(out["baseline"]), 4))
np.savez("/scratch/gpfs/GROUP/USER/tmp/pls_blog/s300.npz", **out)
