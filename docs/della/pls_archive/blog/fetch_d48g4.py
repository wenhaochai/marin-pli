"""Per-layer train CE (train/pls/L{k}, k = 0..47) of the 48-layer runs (width 512, the 130m layer) behind Figures 6 and 7,
the -vs-d48-g4 set (PARAM_MAP=vocab: only vocab-sized tensors sharded, 4 GPUs per run, 2026-10-09). The earlier FSDP
runs in d48.npz decode the readouts wrongly at 48 layers (15255032) and are not used. Separate heads (-sep) and probes
(-sep-bbfrozen), Sandwich-LN (no arch tag) and the DepthBench designs (-a<arch>). W&B reself/marin-della -> d48g4.npz
with <tag>/step, <tag>/ce, <tag>/final (mean of the last 50 logged steps, as everywhere on the page), <tag>/eval
(eval/L{k}/loss summaries), <tag>/params (W&B parameter_count; build_figures_data.py places each design at its own
per-layer size), <tag>/state.
A run counts as complete only when W&B says finished AND its history reaches step num_train_steps - 1 with no gap: a run
that crashes early can also end as 'finished' (the hc run that died at step 259, audit 2026-10-11). An incomplete run's
old keys are removed, so a design never mixes an old half with a new one. History is read in 1000-step windows: one
scan over the whole run drops and repeats a row at each page boundary (audit 2026-10-11).
Usage: python fetch_d48g4.py [--partial] [arch ...] ('' or sandwich = Sandwich-LN; default: all ten). --partial prints
running runs too and writes nothing. Exits 2 if a W&B call failed (the other results are still written)."""
import sys, numpy as np, wandb
from pathlib import Path
OUT = Path(__file__).resolve().parent / "d48g4.npz"
ARCHS = ["", "preln", "lns", "deepnorm", "keel", "hc", "mhc", "attnres", "attnres_block", "moda"]
partial = "--partial" in sys.argv
args = [a for a in sys.argv[1:] if a != "--partial"]
archs = [("" if a == "sandwich" else a) for a in args] or ARCHS
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
keys = [f"train/pls/L{k}" for k in range(48)]
out = dict(np.load(OUT, allow_pickle=True)) if OUT.exists() else {}
errors = 0

def history(r, last):
    by = {}
    for lo in range(0, last + 1, 1000):
        for x in r.scan_history(keys=["_step", *keys], min_step=lo, max_step=lo + 1000, page_size=2000):
            if all(x.get(k) is not None for k in keys):
                by[x["_step"]] = x   # a resumed run logs the steps after its last checkpoint twice; keep the last value
    return by

for arch in archs:
    for setup in ("-sep", "-sep-bbfrozen"):
        tag = setup + (f"-a{arch}" if arch else "") + "-vs-d48-g4"
        try:
            r = api.run(P + tag)
        except Exception as e:
            if "not found" in str(e).lower() or "could not find" in str(e).lower():
                print(tag, "not on W&B")
            else:
                print(tag, "W&B ERROR", type(e).__name__, str(e)[:200]); errors += 1
            continue
        n = int(r.config["trainer"]["num_train_steps"]); last = int(r.summary.get("_step", -1))
        if r.state != "finished" and not partial:
            print(tag, r.state, "step", last, "skipped")
            continue
        by = history(r, last)
        if not by:
            print(tag, r.state, "no rows"); continue
        st = np.array(sorted(by)); ce = np.array([[by[s][k] for k in keys] for s in st])
        gaps = int(st[-1] - st[0] + 1 - len(st))
        complete = r.state == "finished" and st[-1] == n - 1 and last == n - 1 and gaps == 0
        fin = ce[-50:].mean(0)
        ev = np.array([r.summary.get(f"eval/L{k}/loss", np.nan) for k in range(48)], float)
        print(f"{tag} {r.state} step {st[-1]} of {n - 1} unique {len(st)} gaps {gaps} nonfinite {int((~np.isfinite(ce)).sum())} complete {complete}\n"
              f"  final L1 L12 L24 L36 L47 L48 {np.round(fin[[0, 11, 23, 35, 46, 47]], 3).tolist()}\n"
              f"  eval  L1 L12 L24 L36 L47 L48 {np.round(ev[[0, 11, 23, 35, 46, 47]], 3).tolist()} eval/loss {r.summary.get('eval/loss')} params {r.summary.get('parameter_count')}")
        if partial:
            continue
        if complete:
            out.update({tag + "/step": st, tag + "/ce": ce, tag + "/final": fin, tag + "/eval": ev,
                        tag + "/params": np.int64(r.summary["parameter_count"]), tag + "/state": r.state})
        else:
            dropped = [k for k in list(out) if k.startswith(tag + "/")]
            for k in dropped: out.pop(k)
            print(f"  INCOMPLETE: not written{'; removed ' + str(len(dropped)) + ' old keys' if dropped else ''}")
if not partial:
    np.savez(OUT, **out)
sys.exit(2 if errors else 0)
