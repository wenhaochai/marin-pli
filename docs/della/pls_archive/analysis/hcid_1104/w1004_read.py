"""w1004 (Pre-LN fixed gains, ordinary) at its end against Pre-LN new code at the same steps, and the entropy of the token frequencies of the training data, in nats."""
import sys, numpy as np, wandb
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid")
from fw_tokens import load
r = wandb.Api(timeout=300).run("reself/marin-della/muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-apreln-plleak-fa4thd-vs-d48")
by = {x["_step"]: x for x in r.scan_history(keys=["_step", "train/pls/L47", "train/pls/L11", "train/pls/L1"], page_size=2000)}
st = np.array(sorted(by))
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
rs, rc = z["-sep-bbfrozen-apreln-fa4thd-vs-d48/step"], z["-sep-bbfrozen-apreln-fa4thd-vs-d48/ce"]
print("W&B state", r.state, "last step", st[-1], "rows", len(st))
for lo, hi in ((100, 200), (500, 600), (1000, 1100), (1500, 1600), (st[-1] - 99, st[-1] + 1)):
    m = (st >= lo) & (st < hi); q = (rs >= lo) & (rs < hi)
    print(f"steps {lo}-{hi - 1}: L48 {np.mean([by[s]['train/pls/L47'] for s in st[m]]):.3f}, L12 {np.mean([by[s]['train/pls/L11'] for s in st[m]]):.3f}, L2 {np.mean([by[s]['train/pls/L1'] for s in st[m]]):.3f} | Pre-LN L48 {rc[q, 47].mean():.3f}, L12 {rc[q, 11].mean():.3f}, L2 {rc[q, 1].mean():.3f}")
x = load(1, 4096, 2048, seed=1)
v, c = np.unique(x, return_counts=True)
p = c / c.sum()
print(f"entropy of token frequencies over {x.size} tokens: {-(p * np.log(p)).sum():.3f} nats; distinct tokens {v.size}")
