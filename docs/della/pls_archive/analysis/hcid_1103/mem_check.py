"""Peak GPU memory of an HC run on the fa4thd stack, from W&B summary and system metrics: how much room is there for a float32 stream state?"""
import wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
for t in ("-sep-bbfrozen-ahc-hcscalar-fa4thd-vs-d48", "-sep-ahc-fa4thd-vs-d48", "-sep-bbfrozen-ahc-fa4thd-vs-d48"):
    r = api.run(P + t)
    ks = [k for k in r.summary.keys() if any(s in k.lower() for s in ("mem", "peak", "bytes")) and not k.startswith("eval/")]
    print(t, {k: r.summary[k] for k in ks[:30]})
    cfg = dict(r.config)
    print("  batch", cfg.get("trainer", {}).get("train_batch_size"), "seq", cfg.get("model", {}).get("max_seq_len"), "grad ckpt", cfg.get("model", {}).get("gradient_checkpointing"), "mesh", {k: v for k, v in cfg.get("trainer", {}).get("mesh", {}).items() if k in ("axes", "shared_mapping")} if isinstance(cfg.get("trainer", {}).get("mesh"), dict) else None)
    s = r.history(stream="system", samples=200)
    cols = [c for c in s.columns if "gpu.0" in c and any(x in c for x in ("memoryAllocated", "memoryAllocatedBytes", "memory"))]
    print("  system", {c: float(s[c].max()) for c in cols})
