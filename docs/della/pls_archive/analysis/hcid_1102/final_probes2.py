"""Final values of w811 (live W&B) against d48g4.npz references, ordinary model (probes) only.
final = mean of the last 50 logged steps of train/pls/L47; eval = eval/L47/loss. Same-run HC seed values if present."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
r = api.run("reself/marin-della/muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48")
ev = np.array([r.summary.get(f"eval/L{k}/loss", np.nan) for k in range(48)], float)
by = {}
for lo in range(0, 5000, 1000):
    for x in r.scan_history(keys=["_step", *[f"train/pls/L{k}" for k in (0, 3, 7, 11, 23, 35, 46, 47)]], min_step=lo, max_step=lo + 1000, page_size=2000):
        by[x["_step"]] = x
st = sorted(by)[-50:]
F = {k: np.mean([by[s][f"train/pls/L{k}"] for s in st]) for k in (0, 3, 7, 11, 23, 35, 46, 47)}
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
print("w811 final", {k + 1: round(float(v), 3) for k, v in F.items()}, "eval L48", round(float(ev[47]), 4))
for k in sorted(x[:-6] for x in z.files if x.endswith("/final") and x.startswith("-sep-bbfrozen") and "fa4thd" in x):
    f, e = z[k + "/final"], z[k + "/eval"]
    print(f"{k:60s} final {f[47]:.4f} eval {e[47]:.4f} d_final(w811-this) {F[47]-f[47]:+.4f} d_eval {ev[47]-e[47]:+.4f}")
for k in sorted(x[:-6] for x in z.files if x.endswith("/final") and x.startswith("-sep-") and not x.startswith("-sep-bbfrozen") and "fa4thd" in x and ("ahc-" in x or "mhcid" in x)):
    print(f"sep {k:56s} final {z[k+'/final'][47]:.4f} eval {z[k+'/eval'][47]:.4f}")
