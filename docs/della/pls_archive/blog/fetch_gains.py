"""Pre-norm scales of the 48-layer Pre-LN runs behind Figure 36 (L1, owner popup 2026-10-21): Pre-LN, LayerNorm Scaling and
Pre-LN whose scales start at LNS's 1/sqrt(l) (wq w15), all ordinary models with probes (-sep-bbfrozen). Reads the two
pre-norm weights (ln_1 before attention, ln_2 before the MLP; shape layers x 512) from each final checkpoint (step-4958,
OCDBT, read-only). LNS computes h + F(RN(h) g / sqrt(l)), l the 1-based layer, so its converted scale is g / sqrt(l):
the Pre-LN scale with the same function. Per run and layer: the mean over the 512 dims of each scale, and their mean
("mean"). Output: gains.json {tag: {ln_1, ln_2, mean}} in the working directory.
Run on a vis node: nice -n 19 $PY fetch_gains.py"""
import json
import numpy as np
import tensorstore as ts

ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della{}xh100-pls1-sep-bbfrozen-{}/2026.09.13/checkpoints/step-4958"
RUNS = {"preln": (4, "apreln-vs-d48-g4"), "lns": (4, "alns-vs-d48-g4"), "preln-gilns": (8, "apreln-gilns-vs-d48")}
out = {}
for name, (g, tag) in RUNS.items():
    kv = {"driver": "ocdbt", "base": f"file://{ROOT.format(g, tag)}/"}
    w = {k: np.asarray(ts.open({"driver": "zarr3", "kvstore": {**kv, "path": f"model/transformer/layers/stacked/{k}/weight/"}}).result().read().result(), np.float64)
         for k in ("ln_1", "ln_2")}
    assert all(v.shape == (48, 512) for v in w.values()), {k: v.shape for k, v in w.items()}
    if name == "lns":
        w = {k: v / np.sqrt(np.arange(1, 49))[:, None] for k, v in w.items()}
    r = {k: v.mean(1).tolist() for k, v in w.items()}
    r["mean"] = ((w["ln_1"].mean(1) + w["ln_2"].mean(1)) / 2).tolist()
    out[name] = r
    print(name, "layers 1 12 24 36 48 mean scale", np.round(np.array(r["mean"])[[0, 11, 23, 35, 47]], 3), "min dim", round(float(min(v.min() for v in w.values())), 4))
P = np.array(out["preln"]["mean"])
for name in ("lns", "preln-gilns"):
    q = np.array(out[name]["mean"]) / P
    print(f"{name} / preln: layers 1 12 24 36 48 {np.round(q[[0, 11, 23, 35, 47]], 3)}  range {q.min():.3f}-{q.max():.3f}")
json.dump(out, open("gains.json", "w"))
