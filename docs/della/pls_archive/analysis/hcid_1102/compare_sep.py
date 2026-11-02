"""w810 (HC identity mixing, separate heads, live) against HC (ahc-fa4thd-vs-d48, separate heads, finished), per
250-step window: layer-48 train CE difference, global grad norm (median, max), the learning rate, and robust loss spikes
(z > 6 on log CE against a 100-step rolling median, the training-monitor score). Source: W&B reself/marin-della."""
import sys, numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1-" + (sys.argv[1] if len(sys.argv) > 1 else "sep") + "-"   # argv: sep | sep-bbfrozen
K = ["_step", "train/pls/L47", "grad/norm/total", "optim/learning_rate", "train/loss"]
def get(tag):
    r = api.run(P + tag); by = {}
    for lo in range(0, 5000, 1000):
        for x in r.scan_history(keys=K, min_step=lo, max_step=lo + 1000, page_size=2000):
            by[x["_step"]] = x
    return by
a, b = get("ahc-hcid-fa4thd-vs-d48"), get("ahc-fa4thd-vs-d48")
st = np.array(sorted(set(a) & set(b)))
def arr(d, k): return np.array([d[s][k] for s in st], float)
ce_a, ce_b, g_a, g_b, lr = arr(a, "train/pls/L47"), arr(b, "train/pls/L47"), arr(a, "grad/norm/total"), arr(b, "grad/norm/total"), arr(a, "optim/learning_rate")
tl_a, tl_b = arr(a, "train/loss"), arr(b, "train/loss")
def spikes(v):
    u = np.log(v); z = np.zeros_like(u)
    for i in range(100, len(u)):
        w = u[i - 100:i]; q = np.median(w); D = np.median(np.abs(w - q))
        z[i] = 0.6745 * abs(u[i] - q) / (D + 1e-8)
    return z > 6
sa, sb = spikes(ce_a), spikes(ce_b)
print("common steps", st[0], "-", st[-1], len(st))
print("window      dCE48   dTotal  gnorm_med a/b    gnorm_max a/b    lr        spikes a/b")
for lo in range(0, int(st[-1]) + 1, 250):
    m = (st >= lo) & (st < lo + 250)
    if not m.any(): continue
    print(f"{lo:5d}-{lo+249:5d} {np.mean(ce_a[m]-ce_b[m]):+.3f}  {np.mean(tl_a[m]-tl_b[m]):+7.2f}  {np.median(g_a[m]):6.2f}/{np.median(g_b[m]):6.2f}  {g_a[m].max():7.2f}/{g_b[m].max():7.2f}  {lr[m].mean():.5f}  {sa[m].sum():3d}/{sb[m].sum():3d}")
