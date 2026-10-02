import wandb
api = wandb.Api(timeout=120)
for rid in ("muonh-qwen3-130m-della4xh100-ov12.8m", "muonh-qwen3-130m-della4xh100", "muonh-qwen3-300m-della4xh100"):
    try:
        r = api.run(f"reself/marin-della/{rid}")
        print(f"EXISTS {rid}: state={r.state} group={r.group} step={r.summary.get('_step')} keys_eval={len([k for k in r.summary.keys() if k.startswith('eval/')])}")
    except Exception as e:  # noqa: BLE001
        print(f"absent {rid} ({type(e).__name__}: {str(e)[:120]})")
