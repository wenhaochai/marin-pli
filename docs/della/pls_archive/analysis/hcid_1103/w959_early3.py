"""Gradient norms by parameter family at steps 10-60: w959 (mixing keeps the sum) and HC probes. W&B logs every 10 steps."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
R = {"w959": "-sep-bbfrozen-ahc-hcsum-fa4thd-vs-d48", "HC": "-sep-bbfrozen-ahc-fa4thd-vs-d48"}
fam = ["hyper.0.res", "hyper.1.res", "hyper.0.w_res", "hyper.1.w_res", "hyper.0.scale", "hyper.1.scale", "hyper.0.write", "hyper.1.write", "hyper.0.read", "hyper.1.read", "hyper.0.w_read", "hyper.1.w_write"]
K = ["grad/norm/transformer.layers.0." + f for f in fam]
for n, t in R.items():
    r = api.run(P + t)
    rows = {}
    for x in r.scan_history(keys=["_step", "optim/learning_rate", *K], min_step=0, max_step=125, page_size=500):
        rows[x["_step"]] = x
    print(n)
    print(" step lr " + " ".join(f"{f.replace('hyper.','')[:9]:>9}" for f in fam))
    for s in sorted(rows):
        print(f" {s:4d} {rows[s]['optim/learning_rate']:.5f} " + " ".join(f"{rows[s][k]:9.3f}" for k in K))
