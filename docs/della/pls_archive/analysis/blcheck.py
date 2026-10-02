import wandb
api = wandb.Api(timeout=120)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
a = api.run(P + "-restore").summary
b = api.run(P + "-pls0-blreadout").summary
for k in ["eval/paloma/c4_en-marin-tokenizer/bpb", "eval/paloma/macro_loss", "eval/loss", "eval/paloma/code-marin-tokenizer/loss" if False else "eval/bpb"]:
    print(f"{k:45s} restore={a.get(k)} blreadout={b.get(k)}")
print("blreadout global_step", b.get("global_step"), "_step", b.get("_step"), "train/loss", b.get("train/loss"))
