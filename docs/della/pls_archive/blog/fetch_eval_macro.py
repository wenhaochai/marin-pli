"""Per-layer Paloma macro eval loss (eval/L{k}/paloma/macro_loss) of the shared-head pls runs and the same-size
baselines' eval/paloma/macro_loss, at every eval step. W&B reself/marin-della. -> eval_macro.json"""
import json, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-"
out = {}
for size, L in (("130m", 6), ("300m", 12)):
    keys = [f"eval/L{k}/paloma/macro_loss" for k in range(L)]
    rows = [r for r in api.run(P + f"{size}-della4xh100-pls1").scan_history(keys=["_step", *keys]) if all(r.get(k) is not None for k in keys)]
    out[f"{size}-shared"] = {"step": [r["_step"] for r in rows], "macro": [[r[k] for k in keys] for r in rows]}
    rows = [r for r in api.run(P + f"{size}-della4xh100").scan_history(keys=["_step", "eval/paloma/macro_loss"]) if r.get("eval/paloma/macro_loss") is not None]
    out[f"{size}-base"] = {"step": [r["_step"] for r in rows], "macro": [r["eval/paloma/macro_loss"] for r in rows]}
    print(size, "pls steps", out[f"{size}-shared"]["step"], "final", [round(v, 3) for v in out[f"{size}-shared"]["macro"][-1]])
    print(size, "base steps", out[f"{size}-base"]["step"][-3:], "final", round(out[f"{size}-base"]["macro"][-1], 4))
json.dump(out, open("/scratch/gpfs/GROUP/USER/tmp/pls_blog/eval_macro.json", "w"))
