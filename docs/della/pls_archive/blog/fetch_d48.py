"""Per-layer train CE (train/pls/L{k}, k = 0..47) of the 48-layer runs (width 512, the 130m layer): separate heads (-sep)
and probes (-sep-bbfrozen), Sandwich-LN (no arch tag) and the DepthBench designs (-a<arch>). W&B reself/marin-della ->
d48.npz with <tag>/step, <tag>/ce, <tag>/final (mean of the last 50 logged steps, as everywhere on the page), <tag>/state,
<tag>/eval (eval/L{k}/loss summaries). Only finished runs are written; usage: python fetch_d48.py [arch ...] ('' = Sandwich-LN)."""
import sys, numpy as np, wandb
from pathlib import Path
OUT = Path(__file__).resolve().parent / "d48.npz"
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
keys = [f"train/pls/L{k}" for k in range(48)]
out = dict(np.load(OUT, allow_pickle=True)) if OUT.exists() else {}
for arch in (sys.argv[1:] or [""]):
    for setup in ("-sep", "-sep-bbfrozen"):
        tag = setup + (f"-a{arch}" if arch else "") + "-d48"
        try:
            r = api.run(P + tag)
        except Exception as e:
            print(tag, "missing", e); continue
        if r.state != "finished":
            print(tag, r.state, "skipped"); continue
        rows = [x for x in r.scan_history(keys=["_step", *keys], page_size=2000) if all(x.get(k) is not None for k in keys)]
        ce = np.array([[x[k] for k in keys] for x in rows]); st = np.array([x["_step"] for x in rows])
        fin = ce[-50:].mean(0)
        ev = np.array([r.summary.get(f"eval/L{k}/loss", np.nan) for k in range(48)], float)
        out.update({tag + "/step": st, tag + "/ce": ce, tag + "/final": fin, tag + "/eval": ev, tag + "/state": r.state})
        print(f"{tag} step {st[-1]} rows {len(st)} nonfinite {int((~np.isfinite(ce)).sum())} train/loss {r.summary.get('train/loss')}\n"
              f"  final L1..L48 {np.round(fin, 3).tolist()}\n  eval  {np.round(ev, 3).tolist()}")
np.savez(OUT, **out)
