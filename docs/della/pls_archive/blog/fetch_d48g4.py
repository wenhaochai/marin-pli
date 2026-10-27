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
The Lean L1 / L2 arms (pls_arch2_fetchable.sbatch, wq w15 w27 w28) are probes only, -vs-d48 with no -g4, on 8 or 4 GPUs:
name them arch2:<tag> (arch2:preln-gilns, arch2:hc-rstatic, arch2:attnres-rstatic,
arch2:preln-gilnsbf-slns, arch2:preln-slns; not in the default set).
Usage: python fetch_d48g4.py [--partial] [arch[@seed] ...] ('' or sandwich = Sandwich-LN; attnres@s1 = the seed-1 pair; default: all ten). --partial prints
running runs too and writes nothing. Exits 2 if a W&B call failed (the other results are still written)."""
import sys, numpy as np, wandb
from pathlib import Path
OUT = Path(__file__).resolve().parent / "d48g4.npz"
ARCHS = ["", "preln", "lns", "deepnorm", "keel", "hc", "mhc", "attnres", "attnres_block", "moda"]
partial = "--partial" in sys.argv
args = [a for a in sys.argv[1:] if a != "--partial"]
archs = [("" if a == "sandwich" else a) for a in args] or ARCHS
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della{}xh100-pls1"
ARCH2_GPUS = {"preln-gilns": 8, "hc-rstatic": 4, "attnres-rstatic": 4, "preln-gilnsbf-slns": 8, "preln-slns": 8}
keys48 = [f"train/pls/L{k}" for k in range(48)]
out = dict(np.load(OUT, allow_pickle=True)) if OUT.exists() else {}
errors = 0
# Steps whose W&B row was lost although training ran through them, checked in the Slurm logs before listing. 4595: job
# 15266160 saved checkpoint step-4595 at its training limit and stopped before W&B took that step's row; 15266161 loaded
# step-4595 and went on from 4596 (2026-10-13). The final value (last 50 steps) does not touch it.
# 4631: the same for the seed-1 probes run (15520126 saved step-4631 at its limit; 15538869 loaded it, 2026-10-17).
LOGGING_GAPS = {"-sep-aattnres-vs-d48-g4": {4595}, "-sep-bbfrozen-aattnres-vs-d48-s1-g4": {4631}}

def history(r, last, keys):
    by = {}
    for lo in range(0, last + 1, 1000):
        for x in r.scan_history(keys=["_step", *keys], min_step=lo, max_step=lo + 1000, page_size=2000):
            if all(x.get(k) is not None for k in keys):
                by[x["_step"]] = x   # a resumed run logs the steps after its last checkpoint twice; keep the last value
    return by

# The A3 / E1 arms (pls_arch3_fetchable.sbatch, wq w76-w90) are 4-GPU runs named -vs-d{L} with no -g4:
# "arch3:<middle>:<L>[@seed]", e.g. arch3:ahc:24 -> -sep-ahc-vs-d24, arch3:apreln:48@s1 -> -sep-apreln-vs-d48-s1.
for arch in archs:
    arch2 = arch.removeprefix("arch2:") if arch.startswith("arch2:") else None
    arch3 = arch.removeprefix("arch3:") if arch.startswith("arch3:") else None
    nl = int(arch3.split(":")[1].partition("@")[0]) if arch3 else 48
    keys = keys48[:nl]
    sel = sorted({0, nl // 4 - 1, nl // 2 - 1, 3 * nl // 4 - 1, nl - 2, nl - 1})
    for setup in (("-sep-bbfrozen",) if arch2 else ("-sep", "-sep-bbfrozen")):
        a_, _, seed = arch.partition("@")   # "attnres@s1": the seed-1 copy (-vs-d48-s1-g4; Figure 8)
        if arch3:
            mid, _, rest = arch3.partition(":")
            tag = f"{setup}-{mid}-vs-d{nl}" + (f"-{rest.partition('@')[2]}" if "@" in rest else "")
        else:
            tag = f"{setup}-a{arch2}-vs-d48" if arch2 else setup + (f"-a{a_}" if a_ else "") + "-vs-d48" + (f"-{seed}" if seed else "") + "-g4"
        try:
            r = api.run(P.format(ARCH2_GPUS[arch2] if arch2 else 4) + tag)
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
        by = history(r, last, keys)
        if not by:
            print(tag, r.state, "no rows"); continue
        st = np.array(sorted(by)); ce = np.array([[by[s][k] for k in keys] for s in st])
        allowed = LOGGING_GAPS.get(tag, set())
        missing = set(range(int(st[0]), int(st[-1]) + 1)) - set(st.tolist())
        gaps = len(missing - allowed)
        complete = r.state == "finished" and st[-1] == n - 1 and last == n - 1 and gaps == 0
        fin = ce[-50:].mean(0)
        ev = np.array([r.summary.get(f"eval/L{k}/loss", np.nan) for k in range(nl)], float)
        print(f"{tag} {r.state} step {st[-1]} of {n - 1} unique {len(st)} gaps {gaps} (allowed {sorted(missing & allowed)}) nonfinite {int((~np.isfinite(ce)).sum())} complete {complete}\n"
              f"  final L{[k + 1 for k in sel]} {np.round(fin[sel], 3).tolist()}\n"
              f"  eval  L{[k + 1 for k in sel]} {np.round(ev[sel], 3).tolist()} eval/loss {r.summary.get('eval/loss')} params {r.summary.get('parameter_count')}")
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
