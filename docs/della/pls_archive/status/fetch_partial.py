"""Per-layer train CE of the 48-layer designs still running (or stopped mid-chain), mean of the last 50 logged steps at
the time of the fetch. Status only: these runs have not finished, so the values are not final. W&B reself/marin-della
-> partial.npz with <tag>/final, <tag>/step (last step)."""
import numpy as np, wandb
from pathlib import Path
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
keys = [f"train/pls/L{k}" for k in range(48)]
out = {}
for arch in ("moda", "hc", "mhc"):
    for setup in ("-sep", "-sep-bbfrozen"):
        tag = f"{setup}-a{arch}-vs-d48-g4"
        r = api.run(P + tag); last = int(r.summary["_step"])
        by = {x["_step"]: x for x in r.scan_history(keys=["_step", *keys], min_step=last - 80, max_step=last + 1, page_size=500)
              if all(x.get(k) is not None for k in keys)}
        st = sorted(by)[-50:]
        out[tag + "/final"] = np.array([[by[s][k] for k in keys] for s in st]).mean(0); out[tag + "/step"] = np.int64(st[-1])
        print(tag, r.state, st[0], st[-1], len(st))
np.savez(Path(__file__).resolve().parent / "partial.npz", **out)
