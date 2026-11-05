"""Save the W&B configs of the 48-layer 2 x 2 arms for res_gain_arms.py."""
import json, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
for t in ("-sep-bbfrozen-ahc-hcshrink-fa4thd-vs-d48", "-sep-bbfrozen-ahc-hcshrink-sp0.5-fa4thd-vs-d48", "-sep-bbfrozen-ahc-hcmove-fa4thd-vs-d48", "-sep-bbfrozen-ahc-hcsum-hcf32-fa4thd-vs-d48"):
    json.dump(dict(api.run(P + t).config), open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{t}.json", "w"))
    print("ok", t)
