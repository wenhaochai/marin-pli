import json, wandb
api = wandb.Api(timeout=300)
r = api.run("reself/marin-della/muonh-qwen3-1_2b-della8xh100-pls0-noev-probes1000at22887")
n = int(r.config["trainer"]["num_train_steps"]); L = 16
keys = [f"train/pls/L{k}" for k in range(L)] + [f"train/pls2/L{k}" for k in range(L)]
h = list(r.scan_history(keys=["_step", *keys], min_step=max(0, n - 200), max_step=n, page_size=1000))
m = {k: sum(v := [x[k] for x in h if x.get(k) is not None][-50:]) / len(v) for k in keys}
print(r.state, r.summary.get("_step"), n)
print("next ", [round(m[f"train/pls/L{k}"], 3) for k in range(L)])
print("after", [round(m[f"train/pls2/L{k}"], 3) for k in range(L)])
j = json.load(open("/scratch/gpfs/GROUP/USER/project/marin-pls/docs/della/pls_archive/blog/mtpl.json"))
for s in j: print(s, {a: (v["ntp"]["next"] if isinstance(v, dict) and "ntp" in v else None) for a, v in j[s].items() if a == "probes"})
