"""pls1 vs the baseline pool at every shared eval step: macro loss, c4_en loss/bpb, micro loss; plus pls1's per-layer macro."""
import sys
import numpy as np
import wandb

size = sys.argv[1] if len(sys.argv) > 1 else "130m"
P = f"reself/marin-della/muonh-qwen3-{size}-della4xh100"
pool_sfx = ["", "-s1", "-s2", "-s3", "-ema0.999", "-ema0.999-s1", "-ema0.999-s2", "-ema0.999-s3"] if size == "130m" else [""]
K = {"macro": "eval/paloma/macro_loss", "c4_loss": "eval/paloma/c4_en-marin-tokenizer/loss", "c4_bpb": "eval/paloma/c4_en-marin-tokenizer/bpb", "micro": "eval/loss"}
api = wandb.Api(timeout=120)


def hist(run, keys):
    out = {}
    for row in run.scan_history(keys=["_step", *keys]):
        out[int(row["_step"])] = row
    return out


pool = {}
for s in pool_sfx:
    try:
        r = api.run(P + s)
        pool[s] = hist(r, list(K.values()))
    except Exception as e:
        print("skip", s, repr(e)[:80])
pls = api.run(P + (sys.argv[2] if len(sys.argv) > 2 else "-pls1"))
L = 6 if size == "130m" else 12
layer_keys = [f"eval/L{k}/paloma/macro_loss" for k in range(L)]
ph = hist(pls, list(K.values()) + layer_keys)
print(f"{size}: pool n={len(pool)} runs; pls1 state={pls.state}, eval steps {sorted(ph)}")
for st in sorted(ph):
    line = [f"step {st:5d}"]
    for name, key in K.items():
        vals = [h[st][key] for h in pool.values() if st in h and h[st].get(key) is not None]
        v = ph[st].get(key)
        if vals and v is not None:
            m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            line.append(f"{name}: base {m:.4f}±{sd:.4f}(n={len(vals)}) pls1 {v:.4f} Δ{v - m:+.4f}")
    print(" | ".join(line))
    lm = [ph[st].get(k) for k in layer_keys]
    if all(x is not None for x in lm):
        print("   pls1 macro by layer: " + " ".join(f"L{k}={x:.3f}" for k, x in enumerate(lm)))
