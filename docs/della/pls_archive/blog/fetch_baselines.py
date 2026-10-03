"""Final training loss (mean of the last 50 logged steps) and backbone FLOPs of the baseline runs at four sizes.
Backbone FLOPs = logged throughput/total_gflops (levanter, fwd+bwd) minus the LM head's share, 3 * 2*hidden*vocab per
token, tokens = steps * batch * 4096. Source: W&B reself/marin-della muonh-qwen3-{size}-della4xh100. -> baselines.json"""
import json, numpy as np, wandb
api = wandb.Api(timeout=300)
V, SEQ = 128256, 4096
SIZES = {"130m": (512, 128), "300m": (768, 128), "520m": (1024, 256), "1_2b": (2048, 256)}
out = {}
for size, (hid, batch) in SIZES.items():
    rid = f"reself/marin-della/muonh-qwen3-{size}-della4xh100"
    try:
        r = api.run(rid)
    except Exception as e:
        print(size, "MISSING", repr(e)[:80]); continue
    rows = [x for x in r.scan_history(keys=["_step", "train/loss", "throughput/total_gflops"], page_size=5000)
            if x.get("train/loss") is not None and x.get("throughput/total_gflops") is not None]
    step = rows[-1]["_step"]; total = rows[-1]["throughput/total_gflops"] * 1e9
    tokens = (step + 1) * batch * SEQ
    head = 3 * 2 * hid * V * tokens
    loss = float(np.mean([x["train/loss"] for x in rows[-50:]]))
    out[size] = dict(state=r.state, step=step, loss=loss, total_flops=total, backbone_flops=total - head, head_share=head / total)
    print(size, r.state, "step", step, "loss %.4f" % loss, "total %.3g backbone %.3g head share %.2f" % (total, total - head, head / total))
json.dump(out, open("/scratch/gpfs/GROUP/USER/tmp/pls_blog/baselines.json", "w"), indent=1)
