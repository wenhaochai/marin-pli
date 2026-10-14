"""The 12-layer ordinary model (W&B muonh-qwen3-130m-della4xh100-d12) against layer 12 of the 48-layer Sandwich-LN runs
(separate heads, probes; d48g4.npz): final train loss (mean of the last 50 steps) and final eval loss."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
r = api.run("reself/marin-della/muonh-qwen3-130m-della4xh100-d12")
h = [x for x in r.scan_history(keys=["_step", "train/loss"], page_size=5000)]
st = np.array([x["_step"] for x in h]); tl = np.array([x["train/loss"] for x in h])
print("d12", r.state, "steps", st.min(), st.max(), len(st), "train last50", round(float(tl[-50:].mean()), 4), "eval/loss", round(float(r.summary.get("eval/loss")), 4))
G = dict(np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True))
for t in ("-sep-vs-d48-g4", "-sep-bbfrozen-vs-d48-g4"):
    print(t, "L12 train", round(float(G[t + "/final"][11]), 4), "L12 eval", round(float(G[t + "/eval"][11]), 4), "| L48 train", round(float(G[t + "/final"][47]), 4))
for a in ("apreln", "alns", "adeepnorm", "akeel", "ahc", "aattnres_block"):
    print(a, "sep L12 train", round(float(G[f"-sep-{a}-vs-d48-g4/final"][11]), 4))
