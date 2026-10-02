"""Probe diagnostics (PLS_PROBE=1) for the 130m arms: per-layer gradient cosine with the final loss, norm ratios, cos of
the summed aux gradient, per-readout in-context scores, next to the paired final-layer gap d(t) of the full runs.
Writes figs/probe_*.png and prints windowed means."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

D = "/scratch/gpfs/GROUP/USER/tmp/pls/dyn/"
P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
JOBS = dict(x.split("=", 1) for x in sys.argv[1:])  # e.g. shared=14808812 sep=14808813 base=14808814
SFX = {"shared": "-pls1-probe-j{}", "sep": "-pls1-sep-probe-j{}", "base": "-pls0-probe-j{}"}
L = 6
keys = [f"train/pls_probe/cos_L{k}" for k in range(L - 1)] + [f"train/pls_probe/gratio_L{k}" for k in range(L - 1)] + ["train/pls_probe/cos_aux"]
keys += [f"train/pls_probe/icl_L{k}" for k in range(L)] + ["train/pls/L5"]
api = wandb.Api(timeout=300)
data = {}
for arm, jid in JOBS.items():
    r = api.run(P + SFX[arm].format(jid))
    rows = list(r.scan_history(keys=["_step", *keys], page_size=2000))
    data[arm] = {"step": np.array([x["_step"] for x in rows])} | {k: np.array([x[k] for x in rows], float) for k in keys}
    print(arm, r.state, "steps", data[arm]["step"].min(), "-", data[arm]["step"].max())

edges = [0, 50, 100, 130, 160, 200, 300, 400, 500, 700]
for name in ["cos_aux", *[f"cos_L{k}" for k in range(L - 1)], *[f"icl_L{k}" for k in range(L)]]:
    print(f"\n{name}: window " + "".join(f"{a}-{b:<6}" for a, b in zip(edges[:-1], edges[1:])))
    for arm, d in data.items():
        v = d[f"train/pls_probe/{name}"]
        row = [np.nanmean(v[(d["step"] > a) & (d["step"] <= b)]) if np.any((d["step"] > a) & (d["step"] <= b)) else np.nan for a, b in zip(edges[:-1], edges[1:])]
        print(f"  {arm:7s}" + "".join(f"{x:+9.3f}" for x in row))

fig, axs = plt.subplots(1, len(data), figsize=(5.7 * len(data), 4.5), squeeze=False)
axs = axs[0]
cmap = plt.get_cmap("viridis")
for ax, arm in zip(axs, data):
    d = data[arm]
    for k in range(L - 1):
        ax.plot(d["step"], d[f"train/pls_probe/cos_L{k}"], color=cmap(k / (L - 1)), lw=1.1, label=f"cos L{k}")
    ax.plot(d["step"], d["train/pls_probe/cos_aux"], "k--", lw=1.4, label="cos aux (sum)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvspan(130, 160, color="red", alpha=0.12, label="crossover (paired d(t)=0)")
    ax.set_title(f"{arm}: cos(grad CE_k, grad CE_final) on trunk")
    ax.set_xlabel("step")
    ax.set_xlim(0, d["step"].max())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("/scratch/gpfs/GROUP/USER/tmp/pls/figs/probe_cos.png", dpi=140)
fig, axs = plt.subplots(1, len(data), figsize=(5.7 * len(data), 4.5), squeeze=False)
axs = axs[0]
for ax, arm in zip(axs, data):
    d = data[arm]
    for k in range(L):
        ax.plot(d["step"], d[f"train/pls_probe/icl_L{k}"], color=cmap(k / (L - 1)), lw=1.1, label=f"icl L{k}")
    ax.axvspan(130, 160, color="red", alpha=0.12)
    ax.set_title(f"{arm}: in-context score (late - early CE)")
    ax.set_xlabel("step")
    ax.set_xlim(0, d["step"].max())
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("/scratch/gpfs/GROUP/USER/tmp/pls/figs/probe_icl.png", dpi=140)
print("wrote figs/probe_cos.png figs/probe_icl.png")
