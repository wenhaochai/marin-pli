"""The ordinary 130m-recipe models of 12, 24 and 36 layers (W&B reself/marin-della muonh-qwen3-130m-della4xh100-d{12,24,36},
Slurm array 15519767), same-depth references for Figure 6: final training loss (mean of the last 50 logged steps) and
final eval loss, for runs that finished every step -> d48ref.json {depth: {train, eval, step}}."""
import json, numpy as np, wandb
from pathlib import Path
api = wandb.Api(timeout=300)
out = {}
for d in (12, 24, 36):
    try:
        r = api.run(f"reself/marin-della/muonh-qwen3-130m-della4xh100-d{d}")
    except Exception:
        print(d, "no run yet"); continue
    h = [x for x in r.scan_history(keys=["_step", "train/loss"], page_size=5000)]
    st = np.array([x["_step"] for x in h]); tl = np.array([x["train/loss"] for x in h])
    last = int(r.config["trainer"]["num_train_steps"]) - 1
    ok = r.state == "finished" and len(st) and int(st.max()) == last
    print(d, r.state, "last step", int(st.max()) if len(st) else None, "of", last, "complete", ok)
    if ok:
        out[str(d)] = dict(train=float(tl[-50:].mean()), eval=float(r.summary["eval/loss"]), step=int(st.max()))
Path(__file__).with_name("d48ref.json").write_text(json.dumps(out, indent=1))
print(out)
