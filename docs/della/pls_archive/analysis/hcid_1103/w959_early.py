"""Early history of w959 (HC, mixing keeps the stream sum, probes): CE at L2, L12, L48, total loss, grad norm, finite flags; every step up to 300, then coarse. Also lists which metric families exist."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
r = api.run("reself/marin-della/muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-ahc-hcsum-fa4thd-vs-d48")
keys = [k for k in r.summary.keys() if not k.startswith("_")]
print(len(keys), "summary keys; non-pls families:", sorted({k.split("/")[0] + "/" + (k.split("/")[1] if "/" in k else "") for k in keys})[:40])
print({k: r.summary[k] for k in keys if any(s in k for s in ("hc/", "hyper", "res", "norm/total", "lr", "learning"))} if False else "")
rows = []
for x in r.scan_history(keys=["_step", "train/loss", "grad/norm/total", "train/pls/L1", "train/pls/L11", "train/pls/L47"], page_size=2000):
    rows.append((x["_step"], x["train/loss"], x["grad/norm/total"], x["train/pls/L1"], x["train/pls/L11"], x["train/pls/L47"]))
a = np.array(sorted(rows))
print("step loss grad L2 L12 L48")
for s in list(range(0, 40, 4)) + list(range(40, 300, 20)) + list(range(300, 2200, 150)):
    m = a[:, 0] == s
    if m.any():
        print(*[f"{v:.3f}" if i else f"{int(v)}" for i, v in enumerate(a[m][0])])
