"""Parameter and gradient norms of the AttnRes (Full) and AttnRes (Block) 48-layer runs (separate heads, probes) from W&B
reself/marin-della -> attnres_params.npz (<tag>/<key> arrays, <tag>/_step)."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
K = ["params/norm/transformer.final_query", "params/norm/transformer.final_key_gain", "params/norm/transformer.layers.stacked.query",
     "params/norm/transformer.layers.stacked.key_gain", "params/norm/transformer.norm.weight", "params/norm/lm_head.weight",
     "grad/norm/total", "grad/norm/transformer.final_query", "grad/norm/transformer.final_key_gain", "grad/norm/lm_head.weight",
     "grad/norm/transformer.layers.stacked.query", "train/loss", "train/pls/L47"]
out = {}
for a in ("aattnres", "aattnres_block"):
    for s in ("-sep", "-sep-bbfrozen"):
        tag = f"{s}-{a}-vs-d48-g4"; r = api.run(P + tag)
        rows = list(r.scan_history(keys=["_step", "grad/norm/total"], page_size=2000))
        print(tag, len(rows))
        hist = {}
        for k in K:
            xs = [(x["_step"], x[k]) for x in r.scan_history(keys=["_step", k], page_size=2000) if x.get(k) is not None]
            out[f"{tag}/{k}/step"] = np.array([a for a, _ in xs]); out[f"{tag}/{k}"] = np.array([b for _, b in xs])
            print("  ", k, len(xs))
np.savez("attnres_params.npz", **out)
