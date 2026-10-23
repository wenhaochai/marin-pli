"""Train loss (on the training stream) at the end of training, unique tokens vs 8 passes, per tokenizer and for the
setups that flip sign across tokenizers. Source: W&B reself/marin-della train/loss (nats per token), last 3% of steps;
bits per byte = loss / ln2 / bytes per token, bytes per token = 4.7444 (128K, fineweb sample) / token ratio of the
truncation (convert_v<k>.json). Output: train_gap.json."""
import json, math, wandb
api = wandb.Api(timeout=180); P = "reself/marin-della/muonh-qwen3-300m-della4xh100"
R = {128000: 1.0}
R.update({k: json.load(open(f"/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/convert_v{k}.json"))["fineweb-edu-10B"]["ratio"] for k in (64000, 32000, 16000, 8000, 256)})
runs = {128000: ["", "-rep8", "-rep8-rs", "-inrand0.1-fp2-rep8", "-drop0.1-fp2-rep8-rs", "-ema0.999", "-rep8-ema0.999"],
        64000: ["-v64k", "-v64k-rep8", "-v64k-rep8-rs"], 32000: ["-v32k", "-v32k-rep8", "-v32k-rep8-rs"],
        16000: ["-v16k", "-v16k-rep8", "-v16k-rep8-rs"],
        8000: ["-v8k", "-v8k-rep8", "-v8k-rep8-rs", "-inrand0.1-fp2-v8k-rep8", "-drop0.1-fp2-v8k-rep8-rs", "-lrx2-fp2-v8k-rep8", "-lrx15-fp2-v8k-rep8", "-inzero0.1-fp2-v8k-rep8", "-v8k-rep8-repack"],
        256: ["-bytes", "-bytes-rep8"]}
out = {}
for k, tags in runs.items():
    bpt = 4.744406985227457 / R[k]
    for t in tags:
        try:
            r = api.run(P + t)
        except Exception as e:
            print("MISSING", t); continue
        h = [x for x in r.scan_history(keys=["_step", "train/loss"]) if x.get("train/loss") is not None]
        n = max(x["_step"] for x in h); tail = [x["train/loss"] for x in h if x["_step"] >= 0.97 * n]
        # per-pass train loss at the end of each eighth (rep8 = pass boundaries)
        per8 = []
        for i in range(1, 9):
            seg = [x["train/loss"] for x in h if (i - 1) * n / 8 + 0.9 * n / 8 <= x["_step"] < i * n / 8]
            per8.append(round(sum(seg) / len(seg) / math.log(2) / bpt, 4) if seg else None)
        out[t or "128k"] = dict(k=k, steps=n, train_bpb_end=round(sum(tail) / len(tail) / math.log(2) / bpt, 4), per_eighth=per8)
        print(k, t, out[t or "128k"], flush=True)
json.dump(out, open("train_gap.json", "w"), indent=1)
