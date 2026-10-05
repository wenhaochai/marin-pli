"""Per-layer train CE (train/pls/L0..L5) of the 130m separate-heads variants behind Figures 8 and 9: layer-loss weight
0.2, 0.1, 0.05 (-pls0.2-sep, -pls0.1-sep, -pls0.05-sep), own-layer heads (-pls1-sep-local), both (-pls0.2-sep-local), and the Figure 9 runs once finished
(-pls0.2-sep-shdepth, -pls0.2-sep-pcg, -pls0.2-sep-shdepth-pcg). W&B reself/marin-della -> fix.npz with <tag>/final =
mean of the last 50 logged steps (as everywhere on the page), <tag>/step, <tag>/ce; only finished runs are written."""
import numpy as np, wandb
from pathlib import Path
OUT = Path(__file__).resolve().parent / "fix.npz"
api = wandb.Api(timeout=300)
keys = [f"train/pls/L{k}" for k in range(6)]
out = {}
for tag in ("-pls0.1-sep", "-pls0.05-sep", "-pls0.2-sep", "-pls1-sep-local", "-pls0.2-sep-local", "-pls0.2-sep-shdepth", "-pls0.2-sep-pcg", "-pls0.2-sep-shdepth-pcg"):
    try:
        r = api.run("reself/marin-della/muonh-qwen3-130m-della4xh100" + tag)
    except Exception:
        print(tag, "not on W&B yet"); continue
    if r.state != "finished":
        print(tag, r.state, "skipped"); continue
    rows = [x for x in r.scan_history(keys=["_step", *keys], page_size=5000) if all(x.get(k) is not None for k in keys)]
    ce = np.array([[x[k] for k in keys] for x in rows])
    out[tag + "/step"], out[tag + "/ce"], out[tag + "/final"] = np.array([x["_step"] for x in rows]), ce, ce[-50:].mean(0)
    print(tag, "step", rows[-1]["_step"], "final", np.round(ce[-50:].mean(0), 3), "nonfinite", int((~np.isfinite(ce)).sum()))
np.savez(OUT, **out)
