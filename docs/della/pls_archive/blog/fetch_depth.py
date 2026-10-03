"""Final training loss (mean of the last 50 logged steps) of the depth-matched 130m controls with 1..5 layers
(W&B reself/marin-della muonh-qwen3-130m-della4xh100-d{1..5}), plus their logged total FLOPs. -> depth.json"""
import json, numpy as np, wandb
api = wandb.Api(timeout=300)
out = {}
for d in range(1, 6):
    r = api.run(f"reself/marin-della/muonh-qwen3-130m-della4xh100-d{d}")
    rows = [x for x in r.scan_history(keys=["_step", "train/loss", "throughput/total_gflops"], page_size=5000)
            if x.get("train/loss") is not None and x.get("throughput/total_gflops") is not None]
    loss = [x["train/loss"] for x in rows]
    out[d] = dict(state=r.state, step=rows[-1]["_step"], loss=float(np.mean(loss[-50:])),
                  total_flops=rows[-1]["throughput/total_gflops"] * 1e9,
                  steps=[x["_step"] for x in rows], ce=loss,
                  macro=r.summary.get("eval/paloma/macro_loss"), c4=r.summary.get("eval/paloma/c4_en-marin-tokenizer/bpb"),
                  max_loss_after_500=float(max(loss[500:])), nonfinite=int(sum(not np.isfinite(v) for v in loss)))
    print(d, r.state, "step", out[d]["step"], "final %.4f" % out[d]["loss"], "macro", out[d]["macro"], "c4bpb", out[d]["c4"], "nonfinite", out[d]["nonfinite"])
json.dump(out, open("/scratch/gpfs/GROUP/USER/tmp/pls_blog/depth.json", "w"))
