import sys, json, math, wandb
api = wandb.Api(timeout=180); P = "reself/marin-della/muonh-qwen3-300m-della4xh100"
k = int(sys.argv[1]); R = 1.0 if k == 128000 else json.load(open(f"/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/convert_v{k}.json"))["fineweb-edu-10B"]["ratio"]
bpt = 4.744406985227457 / R
for t in sys.argv[2:]:
    h = [x for x in api.run(P + t).scan_history(keys=["_step", "train/loss"]) if x.get("train/loss") is not None]
    n = max(x["_step"] for x in h); tail = [x["train/loss"] for x in h if x["_step"] >= 0.97 * n]
    print(k, t, round(sum(tail) / len(tail) / math.log(2) / bpt, 4))
