"""Per-layer probe losses of the MTP pair (job 15523669, branch pls-mtp) behind the last figure (fig-mtp). 130m, 6 layers,
both runs with detached probes on every layer (PLS_DETACH_BB=1, PLS_OPT_SPLIT=1) for x_{t+1} (train/pls/L{k}) and for
x_{t+2} (train/pls2/L{k}, PLS_FUT=1): NTP (-sep-bbfrozen-os-fut) and NTP + one DeepSeek-V3 MTP module of weight 0.3 on
the last layer (-sep-bbfrozen-os-mtp0.3-fut, also train/pls/mtp). W&B reself/marin-della -> mtp.json with, per run,
next / after (mean of the last 50 logged steps, as everywhere on the page), mtp (same, NTP + MTP only), eval_next
(eval/L{k}/loss summaries), state. A run counts only when W&B says finished AND its history reaches step
num_train_steps - 1 with no gap (fetch_d48g4.py rule); history read in 1000-step windows.
Usage: python fetch_mtp.py [--partial] (--partial prints running runs too and writes nothing). Exits 2 on a W&B error."""
import json, sys, numpy as np, wandb
from pathlib import Path
OUT = Path(__file__).resolve().parent / "mtp.json"
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
RUNS = {"ntp": "-sep-bbfrozen-os-fut", "mtp": "-sep-bbfrozen-os-mtp0.3-fut"}
partial = "--partial" in sys.argv
api = wandb.Api(timeout=300)
nxt, aft = [f"train/pls/L{k}" for k in range(6)], [f"train/pls2/L{k}" for k in range(6)]
out = json.loads(OUT.read_text()) if OUT.exists() else {}
errors = 0
for name, tag in RUNS.items():
    try:
        r = api.run(P + tag)
    except Exception as e:
        print(tag, "W&B ERROR", type(e).__name__, str(e)[:200]); errors += 1; continue
    n = int(r.config["trainer"]["num_train_steps"]); last = int(r.summary.get("_step", -1))
    if r.state != "finished" and not partial:
        print(tag, r.state, "step", last, "skipped"); continue
    keys = nxt + aft + (["train/pls/mtp"] if name == "mtp" else [])
    by = {}
    for lo in range(0, last + 1, 1000):
        for x in r.scan_history(keys=["_step", *keys], min_step=lo, max_step=lo + 1000, page_size=2000):
            if all(x.get(k) is not None for k in keys):
                by[x["_step"]] = x
    if not by:
        print(tag, r.state, "no rows"); continue
    st = np.array(sorted(by)); v = np.array([[by[s][k] for k in keys] for s in st], float)
    gaps = len(set(range(int(st[0]), int(st[-1]) + 1)) - set(st.tolist()))
    complete = r.state == "finished" and st[-1] == n - 1 and last == n - 1 and gaps == 0
    fin = v[-50:].mean(0)
    ev = [r.summary.get(f"eval/L{k}/loss") for k in range(6)]
    print(f"{tag} {r.state} step {st[-1]} of {n - 1} unique {len(st)} gaps {gaps} nonfinite {int((~np.isfinite(v)).sum())} complete {complete}\n"
          f"  next  {np.round(fin[:6], 3).tolist()}\n  after {np.round(fin[6:12], 3).tolist()}"
          + (f"\n  mtp   {fin[12]:.3f}" if name == "mtp" else "") + f"\n  eval next {[round(e, 3) if e is not None else None for e in ev]}")
    if partial:
        continue
    if complete:
        out[name] = dict(tag=tag, next=fin[:6].tolist(), after=fin[6:12].tolist(), mtp=float(fin[12]) if name == "mtp" else None,
                         eval_next=ev, state=r.state)
    else:
        out.pop(name, None); print("  INCOMPLETE: not written")
if not partial:
    OUT.write_text(json.dumps(out, indent=1))
sys.exit(2 if errors else 0)
