"""Per-layer train CE and cumulative training FLOPs (throughput/total_gflops) per step, W&B reself/marin-della -> flops.npz."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-"
runs = {"130-base": ("130m-della4xh100", 0), "130-shared": ("130m-della4xh100-pls1", 6), "130-ro": ("130m-della4xh100-pls1-dh", 6),
        "130-own": ("130m-della4xh100-pls1-sep", 6), "130-off80": ("130m-della4xh100-pls1-off80", 6),
        "300-base": ("300m-della4xh100", 0), "300-shared": ("300m-della4xh100-pls1", 12)}
out = {}
for name, (rid, L) in runs.items():
    keys = ["throughput/total_gflops"] + ([f"train/pls/L{k}" for k in range(L)] if L else ["train/loss"])
    rows = [r for r in api.run(P + rid).scan_history(keys=["_step", *keys], page_size=5000) if all(r.get(k) is not None for k in keys)]
    out[name + "/step"] = np.array([r["_step"] for r in rows], float)
    out[name + "/gflops"] = np.array([r["throughput/total_gflops"] for r in rows], float)
    out[name + "/ce"] = np.array([[r[k] for k in keys[1:]] for r in rows], float)
    print(name, len(rows), "steps", out[name + "/step"][[0, -1]], "PFLOP total %.3g" % (out[name + "/gflops"][-1] / 1e6), "final", out[name + "/ce"][-1].round(3))
np.savez("/scratch/gpfs/GROUP/USER/tmp/pls_blog/flops.npz", **out)
