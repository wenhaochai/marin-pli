"""Per-layer train CE (train/pls/L{k}) of the 130m 只训独立头 run (separate heads, backbone detached) and the end of
the other 130m arms, plus Paloma macro and c4_en bpb summaries. W&B reself/marin-della -> bbfrozen.npz, prints a table.
Final value = mean of the last 50 logged steps, same as baselines.json and depth.json."""
import json, numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
keys = [f"train/pls/L{k}" for k in range(6)]
out = {}
for n, name in [("-pls1-sep-bbfrozen", "只训独立头"), ("-pls1", "共享头"), ("-pls1-dh", "只读共享头"), ("-pls1-sep", "独立头")]:
    r = api.run(P + n)
    rows = [x for x in r.scan_history(keys=["_step", "train/loss", *keys], page_size=5000) if all(x.get(k) is not None for k in keys)]
    ce = np.array([[x[k] for k in keys] for x in rows])
    out[n + "/step"] = np.array([x["_step"] for x in rows]); out[n + "/ce"] = ce
    fin = ce[-50:].mean(0)
    print(f"{name:6s} {r.state} step {rows[-1]['_step']} final L0..L5 {np.round(fin, 3)} nonfinite {int((~np.isfinite(ce)).sum())}"
          f" macro {r.summary.get('eval/paloma/macro_loss')} c4bpb {r.summary.get('eval/paloma/c4_en-marin-tokenizer/bpb')}")
    out[n + "/final"] = fin
b = api.run(P)
bl = [x["train/loss"] for x in b.scan_history(keys=["_step", "train/loss"], page_size=5000)]
print("baseline final", round(float(np.mean(bl[-50:])), 4), "macro", b.summary.get("eval/paloma/macro_loss"), "c4bpb", b.summary.get("eval/paloma/c4_en-marin-tokenizer/bpb"))
d = json.load(open("depth.json"))
print("depth controls d1..d5", [round(d[str(k)]["loss"], 4) for k in range(1, 6)])
np.savez("/scratch/gpfs/GROUP/USER/tmp/pls_blog/bbfrozen.npz", **out)
