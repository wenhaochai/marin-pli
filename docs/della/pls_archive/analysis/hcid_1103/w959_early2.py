"""First 120 steps of w959 (HC, mixing keeps the sum, probes) next to w960 (mixing = scalar), HC probes and w811 (mixing = I, probes): L2, L12, L24, L48 CE and total gradient norm at each step. Also the non-eval metric families."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
R = {"w959": "-sep-bbfrozen-ahc-hcsum-fa4thd-vs-d48", "w960": "-sep-bbfrozen-ahc-hcscalar-fa4thd-vs-d48", "HC": "-sep-bbfrozen-ahc-fa4thd-vs-d48", "w811": "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48"}
K = ["train/pls/L1", "train/pls/L11", "train/pls/L23", "train/pls/L47", "grad/norm/total"]
out = {}
for n, t in R.items():
    r = api.run(P + t)
    if n == "w959":
        keys = [k for k in r.summary.keys() if not k.startswith(("_", "eval/", "train/pls/"))]
        print(len(keys), "other keys, families:", sorted({"/".join(k.split("/")[:2]) for k in keys})[:60])
        print("sample:", [k for k in keys if any(s in k.lower() for s in ("hyper", "res", "write", "ln_"))][:40])
        print("lr keys:", {k: r.summary[k] for k in keys if "learning" in k.lower() or k.endswith("/lr") or "lr" == k.split("/")[-1]})
    rows = {}
    for x in r.scan_history(keys=["_step", *K], min_step=0, max_step=125, page_size=500):
        rows[x["_step"]] = [x[k] for k in K]
    out[n] = rows
print("step | " + " | ".join(f"{n}: L2 L12 L24 L48 grad" for n in R))
for s in list(range(0, 60, 2)) + list(range(60, 125, 8)):
    print(f"{s:4d} | " + " | ".join(" ".join(f"{v:7.2f}" for v in out[n][s]) if s in out[n] else "-" for n in R))
