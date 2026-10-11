"""Per-layer train CE (train/pls/L0..L5) of the 130m separate-heads variants behind Figures 8 and 9: layer-loss weight
0.2, 0.1, 0.05 (-pls0.2-sep, -pls0.1-sep, -pls0.05-sep), own-layer heads (-pls1-sep-local), both (-pls0.2-sep-local), and the Figure 9 runs once finished
(-pls0.2-sep-shdepth, -pls0.2-sep-pcg, -pls0.2-sep-shdepth-pcg, -pls1-sep-pcg), and Figure 11's late starts
(-pls1-sep-on300-400, -pls1-sep-on1500-1600). W&B reself/marin-della -> fix.npz with <tag>/final =
mean of the last 50 logged steps (as everywhere on the page), <tag>/step, <tag>/ce; only complete runs are written
(W&B finished and history at step num_train_steps - 1). Results merge into the existing fix.npz: a W&B error other than
"run not found" keeps the old entry and makes the script exit 2, instead of writing the file without that run (audit 2026-10-11)."""
import sys, numpy as np, wandb
from pathlib import Path
OUT = Path(__file__).resolve().parent / "fix.npz"
api = wandb.Api(timeout=300)
keys = [f"train/pls/L{k}" for k in range(6)]
out = dict(np.load(OUT, allow_pickle=True)) if OUT.exists() else {}
errors = 0
for tag in ("-pls0.1-sep", "-pls0.05-sep", "-pls0.2-sep", "-pls1-sep-local", "-pls0.2-sep-local", "-pls0.2-sep-shdepth", "-pls0.2-sep-pcg", "-pls0.2-sep-shdepth-pcg", "-pls1-sep-pcg", "-pls1-sep-on300-400", "-pls1-sep-on1500-1600"):
    try:
        r = api.run("reself/marin-della/muonh-qwen3-130m-della4xh100" + tag)
    except Exception as e:
        if "not found" in str(e).lower() or "could not find" in str(e).lower():
            print(tag, "not on W&B yet")
        else:
            print(tag, "W&B ERROR, old entry kept:", type(e).__name__, str(e)[:200]); errors += 1
        continue
    if r.state != "finished":
        print(tag, r.state, "skipped"); continue
    n = int(r.config["trainer"]["num_train_steps"])
    by = {}
    for lo in range(0, n, 1000):   # windows: one long scan drops and repeats a row at each page boundary
        for x in r.scan_history(keys=["_step", *keys], min_step=lo, max_step=lo + 1000, page_size=2000):
            if all(x.get(k) is not None for k in keys):
                by[x["_step"]] = x
    st = np.array(sorted(by))
    if not len(st) or st[-1] != n - 1:
        print(tag, "INCOMPLETE: ends at step", st[-1] if len(st) else None, "of", n - 1, "- not written")
        for k in [k for k in out if k.startswith(tag + "/")]: out.pop(k)
        continue
    ce = np.array([[by[s][k] for k in keys] for s in st])
    out[tag + "/step"], out[tag + "/ce"], out[tag + "/final"] = st, ce, ce[-50:].mean(0)
    print(tag, "step", st[-1], "gaps", int(st[-1] - st[0] + 1 - len(st)), "final", np.round(ce[-50:].mean(0), 3), "nonfinite", int((~np.isfinite(ce)).sum()))
np.savez(OUT, **out)
sys.exit(2 if errors else 0)
