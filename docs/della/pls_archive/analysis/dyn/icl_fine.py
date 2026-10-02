"""In-context score (final readout, L5) and final-layer train CE in 20-step bins for the three probe runs, plus the
paired loss gap of the full runs (pls - baseline) from dyn/*.npz."""
import numpy as np
import wandb

P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
runs = {"base": "-pls0-probe-j14808814", "shared": "-pls1-probe-j14808812", "sep": "-pls1-sep-probe-j14808813"}
keys = ["train/pls_probe/icl_L5", "train/pls_probe/icl_L2", "train/pls/L5", "train/pls_probe/cos_aux"]
api = wandb.Api(timeout=300)
D = {}
for n, s in runs.items():
    rows = list(api.run(P + s).scan_history(keys=["_step", *keys], page_size=2000))
    D[n] = {"step": np.array([r["_step"] for r in rows])} | {k: np.array([r[k] for r in rows], float) for k in keys}
    print(n, "steps", D[n]["step"].min(), "-", D[n]["step"].max())
full = {n: np.load(f"/scratch/gpfs/GROUP/USER/tmp/pls/dyn/{f}.npz") for n, f in [("base", "base"), ("shared", "-pls1"), ("sep", "-pls1-sep")]}
print("\nbin      | icl_L5 base shared sep | icl_L2 base shared sep | CE_L5(probe run) base shared sep | full-run gap shared sep")
for a in range(60, 460, 20):
    b = a + 20
    def m(n, k):
        d = D[n]; sel = (d["step"] >= a) & (d["step"] < b)
        return np.nanmean(d[k][sel]) if sel.any() else np.nan
    fb = full["base"]; sel = (fb["step"] >= a) & (fb["step"] < b)
    gaps = [np.mean(full[n]["train/pls/L5"][sel] - fb["train/loss"][sel]) if sel.any() else np.nan for n in ("shared", "sep")]
    print(f"{a:3d}-{b:3d}  | " + " ".join(f"{m(n, 'train/pls_probe/icl_L5'):+.3f}" for n in runs) + " | "
          + " ".join(f"{m(n, 'train/pls_probe/icl_L2'):+.3f}" for n in runs) + " | "
          + " ".join(f"{m(n, 'train/pls/L5'):.3f}" for n in runs) + " | " + " ".join(f"{g:+.3f}" for g in gaps))
