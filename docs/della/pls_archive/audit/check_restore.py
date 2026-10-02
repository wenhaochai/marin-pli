import wandb
api = wandb.Api(timeout=120)
for rid in ("muonh-qwen3-130m-della4xh100-restore", "muonh-qwen3-130m-della4xh100-s1-restore"):
    try:
        r = api.run(f"reself/marin-della/{rid}")
        m = r.config.get("model", {})
        print(rid, r.state, r.group, "model type:", m.get("type"), "layers", m.get("num_layers"), "steps", r.config.get("trainer", {}).get("num_train_steps"),
              "init_from", r.config.get("trainer", {}).get("initialize_from"), "c4bpb", r.summary.get("eval/paloma/c4_en-marin-tokenizer/bpb"), "tags", r.tags)
    except Exception as e:
        print(rid, "absent", type(e).__name__)
