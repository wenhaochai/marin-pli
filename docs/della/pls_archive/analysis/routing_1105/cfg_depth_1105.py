"""Save the W&B configs of the 12- and 24-layer runs for res_gain_depth.py."""
import json, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
for t in ("-sep-bbfrozen-ahc-fa4thd-vs-d12", "-sep-bbfrozen-ahc-fa4thd-vs-d24", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d12", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d24"):
    json.dump(dict(api.run(P + t).config), open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{t}.json", "w"))
    print("ok", t)
