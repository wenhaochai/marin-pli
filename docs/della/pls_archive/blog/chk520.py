import wandb
api = wandb.Api(timeout=300)
for rid in ["520m-della8xh100-pls0-noev-probes1000at9917", "300m-della8xh100-pls0-noev-probes2000at11443"]:
    r = api.run("reself/marin-della/muonh-qwen3-" + rid)
    L = 24 if "520m" in rid else 12
    ks = ["_step", f"train/pls/L{L-1}", "train/pls/L0", f"train/pls2/L{L-1}", "train/loss"]
    h = list(r.scan_history(keys=ks, page_size=2000))
    print(rid, r.state, r.summary.get("_step"), "rows", len(h), "created", r.created_at)
    for x in h[:3] + h[len(h)//2:len(h)//2+2] + h[-3:]: print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
