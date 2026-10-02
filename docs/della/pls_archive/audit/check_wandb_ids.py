"""Audit: do the planned pls run ids already exist in W&B (reself/marin-della)? (resume="allow" would append to them)."""
import wandb

api = wandb.Api(timeout=120)
want = [f"muonh-qwen3-{s}-della4xh100-pls{w}{sm}" for s in ("130m", "300m") for w in ("1", "0") for sm in ("", "-smoke40")]
for rid in want:
    try:
        r = api.run(f"reself/marin-della/{rid}")
        print(f"EXISTS {rid}: state={r.state} created={r.created_at} group={r.group} step={r.summary.get('_step')}")
    except Exception as e:  # noqa: BLE001
        print(f"absent {rid} ({type(e).__name__})")
runs = api.runs("reself/marin-della", filters={"display_name": {"$regex": "pls"}}, per_page=100)
print("runs whose name matches 'pls':", [(r.id, r.state) for r in runs][:50])
runs = api.runs("reself/marin-della", filters={"group": "muonh-qwen3-pls-della"}, per_page=100)
print("runs in group muonh-qwen3-pls-della:", [(r.id, r.state) for r in runs][:50])
