"""One figure for the dynamic-schedule question (130m): (a) final-layer train CE gap of the full runs vs the same-init
baseline run, with the baseline's seed spread; (b) in-context score of the final readout (probe runs); (c) cosine of the
summed per-layer gradient with the final-loss gradient on the trunk (probe runs; baseline = what the logit-lens losses
would do). 20-step running means."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

D = "/scratch/gpfs/GROUP/USER/tmp/pls/dyn/"
P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
runs = {"baseline": "-pls0-probe-j14808814", "shared head": "-pls1-probe-j14808812", "own heads": "-pls1-sep-probe-j14808813"}
colors = {"baseline": "black", "shared head": "tab:red", "own heads": "tab:blue"}
keys = ["train/pls_probe/icl_L5", "train/pls_probe/cos_aux"]
api = wandb.Api(timeout=300)
pr = {}
for n, s in runs.items():
    rows = list(api.run(P + s).scan_history(keys=["_step", *keys], page_size=2000))
    pr[n] = {"step": np.array([r["_step"] for r in rows])} | {k: np.array([r[k] for r in rows], float) for k in keys}


def smooth(x, y, w=20):
    o = np.argsort(x)
    x, y = x[o], y[o]
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


full = {n: np.load(D + f + ".npz") for n, f in [("baseline", "base"), ("shared head", "-pls1"), ("own heads", "-pls1-sep")]}
seeds = np.load(D + "seeds_trainloss.npy", allow_pickle=True).item()

fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
b = full["baseline"]
st = b["step"]
for n in ("shared head", "own heads"):
    m = st <= 600
    x, y = smooth(st[m], full[n]["train/pls/L5"][m] - b["train/loss"][m], 3)
    axs[0].plot(x, y, color=colors[n], label=n)
ss = sorted(seeds["base"])
for s in ("-s1", "-s2", "-s3"):
    xs = np.array([t for t in ss if t <= 600 and t in seeds[s]])
    x, y = smooth(xs, np.array([seeds[s][t] - seeds["base"][t] for t in xs]), 3)
    axs[0].plot(x, y, color="gray", lw=0.8, alpha=0.7, label="baseline other seeds" if s == "-s1" else None)
axs[0].axhline(0, color="k", lw=0.6)
axs[0].set_ylabel("final-layer train CE\nminus same-init baseline")
axs[0].set_title("(a) per-layer supervision helps for ~140 steps, then hurts")
for n, d in pr.items():
    x, y = smooth(d["step"], d["train/pls_probe/icl_L5"])
    axs[1].plot(x, y, color=colors[n], label=n)
    x, y = smooth(d["step"], d["train/pls_probe/cos_aux"])
    axs[2].plot(x, y, color=colors[n], label=n)
axs[1].set_ylabel("in-context score\n(late - early position CE)")
axs[1].set_title("(b) the baseline starts using context first; supervised runs lag ~30-50 steps")
axs[2].set_ylabel("cos(sum of per-layer grads,\nfinal-loss grad) on trunk")
axs[2].set_title("(c) gradient alignment stays high in supervised runs: no signal at the crossover")
axs[2].axhline(0, color="k", lw=0.6)
for ax in axs:
    ax.axvspan(140, 160, color="orange", alpha=0.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axs[2].set_xlabel("step (130m, constant LR, no warmup; shaded = crossover of panel a)")
axs[2].set_xlim(0, 580)
fig.tight_layout()
out = "/scratch/gpfs/GROUP/USER/project/marin-pls/docs/della/pls_figs/130m_dynamic_signals.png"
fig.savefig(out, dpi=150)
print("wrote", out)
