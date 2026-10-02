import wandb
api = wandb.Api(timeout=120)
for rid in ("muonh-qwen3-130m-della4xh100", "muonh-qwen3-300m-della4xh100"):
    r = api.run(f"reself/marin-della/{rid}")
    s = r.summary
    print(rid, {k: s.get(k) for k in ("_runtime", "throughput/tokens_per_second", "throughput/mfu", "throughput/duration", "eval/loss", "eval/paloma/macro_loss", "eval/total_time", "eval/loading_time")})
    print("  eval keys sample:", sorted(k for k in s.keys() if k.startswith("eval/"))[:14])
    print("  config trainer steps_per_eval/max_eval_batches:", r.config.get("trainer", {}).get("steps_per_eval"), r.config.get("trainer", {}).get("max_eval_batches"), "per_device_eval_parallelism", r.config.get("trainer", {}).get("per_device_eval_parallelism"))
