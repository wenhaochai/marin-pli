"""Pull per-step train metrics (per-layer CE, grad norms by block) for the 130m pls arms and the paired baseline into
dyn/<suffix>.npz, for the dynamic-schedule analysis."""
import sys
import numpy as np
import wandb

P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
PARAMS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
api = wandb.Api(timeout=300)
for sfx in sys.argv[1:]:
    sfx = "" if sfx == "base" else sfx
    r = api.run(P + sfx)
    keys = ["train/loss", "grad/norm/total", "grad/norm/lm_head.weight", "grad/norm/embeddings.token_embeddings.weight", "grad/norm/transformer.norm.weight"]
    keys += [f"train/pls/L{k}" for k in range(6)] if sfx else []
    keys += [f"grad/norm/transformer.layers.{i}.{p}.weight" for i in range(6) for p in PARAMS]
    rows = list(r.scan_history(keys=["_step", *keys], page_size=2000))
    steps = np.array([row["_step"] for row in rows])
    out = {"step": steps}
    for k in keys:
        out[k] = np.array([np.nan if row.get(k) is None else row[k] for row in rows], dtype=float)
    blk = np.stack([np.sqrt(sum(out[f"grad/norm/transformer.layers.{i}.{p}.weight"] ** 2 for p in PARAMS)) for i in range(6)], 1)
    out["block_grad"] = blk
    np.savez(f"/scratch/gpfs/GROUP/USER/tmp/pls/dyn/{sfx or 'base'}.npz", **out)
    print(sfx or "base", len(steps), steps.min(), steps.max())
