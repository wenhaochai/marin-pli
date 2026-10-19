"""Final per-layer training loss of the ten more attempts at the last-layer cost (Figure 15, fig-auto10; branch
pls-auto10). Per run: train/pls/L0..L5, mean of the last 50 logged steps (as Figure 3, fetch_bbfrozen.py); a run counts
only when W&B says finished and its last logged step is num_train_steps - 1. For 9 (HC) and 10 (AttnRes (Full)) the
probes run of the same design is kept too (key + "-probes"): its layer 6 is that design's ordinary 6-layer model.
W&B reself/marin-della -> auto10.json {key: [L1..L6]}. Usage: python fetch_auto10.py. Exits 2 on a W&B error."""
import json, sys, numpy as np, wandb
from pathlib import Path
OUT = Path(__file__).resolve().parent / "auto10.json"
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1-sep"
RUNS = {"os": "-os", "mix": "-mix0.2", "kd": "-kd", "conf": "-conf", "ad": "-ad1024", "mlpo": "-mlpo", "first": "-first256", "randl": "-randl",
        "hc": "-ahc", "hc-probes": "-bbfrozen-ahc", "attnres": "-aattnres", "attnres-probes": "-bbfrozen-aattnres",
        "upto3": "-upto3", "upto2": "-upto2", "upto1": "-upto1", "upto2s1": "-upto2-s1"}
keys = [f"train/pls/L{k}" for k in range(6)]
api = wandb.Api(timeout=300)
out, errors = {}, 0
for name, tag in RUNS.items():
    try:
        r = api.run(P + tag)
    except Exception as e:
        if "Could not find run" in str(e) or "404" in str(e):
            continue
        print(name, "W&B ERROR", str(e)[:200]); errors += 1; continue
    n, last = int(r.config["trainer"]["num_train_steps"]), int(r.summary.get("_step", -1))
    if r.state != "finished" or last != n - 1:
        print(name, r.state, "step", last, "of", n, "skipped"); continue
    rows = [x for x in r.scan_history(keys=["_step", *keys], min_step=n - 400, max_step=n, page_size=1000) if all(x.get(k) is not None for k in keys)]
    ce = np.array([[x[k] for k in keys] for x in rows])
    if len(rows) < 50 or rows[-1]["_step"] != n - 1 or not np.isfinite(ce).all():
        print(name, "history short or non-finite near the end:", len(rows)); errors += 1; continue
    out[name] = [round(float(v), 5) for v in ce[-50:].mean(0)]
    print(name, out[name])
OUT.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", OUT, sorted(out))
sys.exit(2 if errors else 0)
